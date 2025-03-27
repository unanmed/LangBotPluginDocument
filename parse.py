import hashlib
import os
import traceback
import json
from tqdm import tqdm
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_community.vectorstores import FAISS, DistanceStrategy
from .loader import CodeAwareMDLoader, CodeLoader
from .splitter import DocumentSplitter
from .retriever import HybridRetriever
from .watcher import DocumentWatcher
from .extensions.classification import Classification

def is_path_in_directory(path, directory):
    # 计算公共前缀
    common_path = os.path.commonpath([os.path.abspath(path), os.path.abspath(directory)])
    return common_path == os.path.abspath(directory)

class DocumentParser:
    text_model: HuggingFaceEmbeddings = None
    code_model: HuggingFaceEmbeddings = None
    
    splitter: DocumentSplitter = None
    docs: list[Document] = []
    
    text_store: FAISS = None
    code_store: FAISS = None
    code_comment_store: FAISS = None
    
    retriever: HybridRetriever = None
    
    root_path: str = None
    indices_path: str = None
    config: object = None
    indices_cache: dict[str, object] = None
    doc_ids: dict[str, tuple[list[str], list[str], list[str]]] = dict()
    doc_text_indices: list[FAISS] = list()
    doc_code_indices: list[FAISS] = list()
    doc_comment_indices: list[FAISS] = list()
    
    deleted_docs: set[str] = set()
    max_id: int = 0
    
    from_cache: int = 0
    modified: int = 0
    new_doc: int = 0
    indexed: int = 0
    
    watcher: DocumentWatcher
    
    def __init__(self, config, indices_cache: dict[str, object], root: str, indices_path: str):
        for i, path in enumerate(config['files']):
            config['files'][i] = os.path.normpath(path)
        
        self.config = config
        self.indices_cache = indices_cache
        self.indices_path = indices_path
        
        # 检查索引缓存的格式
        self.check_indices_cache()
        self.doc_ids = self.indices_cache['doc_ids']
        
        self.root_path = root
        self.deleted_docs = { os.path.join(root, 'docs', path) for path in config["files"] }
        if len(self.indices_cache['data']) == 0:
            self.max_id = 0
        else:
            self.max_id = max([int(index["id"]) for index in self.indices_cache['data'].values()])
        self.splitter = DocumentSplitter(
            code_context_length=config["code_context_length"],
            chunk_size=config["chunk_size"],
            chunk_overlap=config["chunk_overlap"]
        )
        self.watcher = DocumentWatcher(root, os.path.join(root, 'docs'), self)
        
    def check_indices_cache(self):
        cache = self.indices_cache
        
        if cache.get('data') is None:
            # 如果 data 是 None 的话，说明是旧版缓存，但是由于新版更改了内容，没办法沿用旧版缓存，因此直接删除
            cache = dict()
            cache['data'] = dict()
            cache['doc_ids'] = dict()
        
        # 给缓存加个版本信息，方便后续更新
        cache['version'] = 1
        self.indices_cache = cache
    
    def fetch_models(self):
        """根据配置信息获取需要的模型"""
        modes = {self.config["mode"]}

        # 提取所有文件中的模式
        for file in self.config["files"]:
            if not isinstance(file, str):
                modes.add(file["mode"])
            elif not file.endswith(".md"):
                modes.add("code-only")
        
        # 判断是否需要文本或代码模型
        need_text = "text-only" in modes or "text-code" in modes
        need_code = "code-only" in modes or "text-code" in modes
        
        if need_text and need_code:
            print("Using text model and code model to parse documents.")
        elif need_text and not need_code:
            print("Using text model to parse documents.")
        elif not need_text and need_code:
            print("Using code model to parse documents.")
            
        # 加载模型，暂时两个模型都加载，之后再优化为按需加载
        if need_text:
            self.text_model = HuggingFaceEmbeddings(model_name=self.config["text_model"])
        if need_code:
            self.code_model = HuggingFaceEmbeddings(model_name=self.config["code_model"])
            
    def check_cache(self, doc_path: str) -> bool:
        """检查一个文档的缓存是否存在，以及是否需要重新索引"""
        indices = self.indices_cache['data'].get(doc_path)
        
        if not indices:
            self.new_doc += 1
            return False
        
        if not os.path.exists(doc_path):
            return False
        
        with open(doc_path, 'r', encoding='utf-8') as doc:
            file_hash = hashlib.sha256(doc.read().encode()).hexdigest()
        
        if file_hash == indices["hash"]:
            self.from_cache += 1
            return True
        
        self.modified += 1
        return False
    
    def cache_index(self, doc_path: str, text_index: FAISS, code_index: FAISS, comment_index: FAISS):
        """将一个索引写入缓存"""
        
        indices = self.indices_cache['data'].get(doc_path)
        if indices:
            index_id = indices["id"]
        else:
            self.max_id += 1
            index_id = self.max_id
            
        text_path = os.path.join(self.root_path, "data\\text", f"doc_{index_id}")
        code_path = os.path.join(self.root_path, "data\\code", f"doc_{index_id}")
        comment_path = os.path.join(self.root_path, "data\\comment", f"doc_{index_id}")
        
        with open(doc_path, 'r', encoding='utf-8') as doc:
            file_hash = hashlib.sha256(doc.read().encode()).hexdigest()
            
        self.indices_cache['data'][doc_path] = {
            "text_path": text_path if text_index else None,
            "comment_path": comment_path if comment_index else None,
            "code_path": code_path if code_index else None,
            "id": str(index_id),
            "hash": file_hash
        }
        
        if text_index:
            FAISS.save_local(text_index, text_path)
        if code_index:
            FAISS.save_local(code_index, code_path)
        if comment_index:
            FAISS.save_local(comment_index, comment_path)
            
    def load_document(self, doc_path: str, mode: str, nocache=False):
        path = Path(doc_path)
        ext = path.suffix
        if not os.path.exists(doc_path):
            tqdm.write(f'Warn: File {doc_path} does not exists.')
            return
        
        if self.check_cache(doc_path) and not nocache:
            # 有缓存，直接从缓存加载
            indices = self.indices_cache['data'].get(doc_path)
            text = None
            code = None
            comment = None
            if indices["text_path"] and os.path.exists(indices["text_path"]):
                text = FAISS.load_local(
                    indices["text_path"], self.text_model, "index", allow_dangerous_deserialization=True,
                    distance_strategy=DistanceStrategy.COSINE
                )
                self.doc_text_indices.append(text)
            if indices["code_path"] and os.path.exists(indices["code_path"]):
                code = FAISS.load_local(
                    indices["code_path"], self.code_model, "index", allow_dangerous_deserialization=True,
                    distance_strategy=DistanceStrategy.COSINE
                )
                self.doc_code_indices.append(code)
            if indices["comment_path"] and os.path.exists(indices["comment_path"]):
                comment = FAISS.load_local(
                    indices["comment_path"], self.text_model, "index", allow_dangerous_deserialization=True,
                    distance_strategy=DistanceStrategy.COSINE
                )
                self.doc_comment_indices.append(comment)
            self.deleted_docs.remove(doc_path)
            return text, code, comment
            
        else:
            # 没有缓存，加载文档并索引
            if ext == ".md":
                loader = CodeAwareMDLoader(path)
                docs = self.splitter.split_documents(loader.load(), mode)
            
            else:
                loader = CodeLoader(path)
                docs = self.splitter.split_documents(loader.load(), "code-only")
                
            text, code, comment = self.parse_one_document(docs, doc_path)
            if text:
                self.doc_text_indices.append(text)
            if code:
                self.doc_code_indices.append(code)
            if comment:
                self.doc_comment_indices.append(comment)
            self.cache_index(doc_path, text, code, comment)
            self.indexed += 1
            if doc_path in self.deleted_docs:
                self.deleted_docs.remove(doc_path)
                
            return text, code, comment

    def parse_one_document(self, docs: list[Document], path: str):
        text_docs = [doc for doc in docs if not doc.metadata.get("is_code", False)]
        code_docs = [doc for doc in docs if doc.metadata.get("is_code", False)]
        
        code_comment_docs: list[Document] = []
        for doc in code_docs:
            comment = doc.metadata.get("comments")
            if comment and comment.strip():
                metadata = { **doc.metadata, "code": doc.page_content }
                metadata.pop("comments")
                code_comment_docs.append(Document(page_content=comment, metadata=metadata))
        
        path = os.path.normpath(os.path.relpath(path, self.root_path))
        self.doc_ids[path] = (list(), list(), list())
        i = 0
        for type, docs in enumerate([text_docs, code_docs, code_comment_docs]):
            for doc in docs:
                doc.id = f"{path}-{i}"
                self.doc_ids[path][type].append(doc.id)
                i += 1
        
        text_store = FAISS.from_documents(text_docs, self.text_model, distance_strategy=DistanceStrategy.COSINE) if text_docs else None
        code_store = FAISS.from_documents(code_docs, self.code_model, distance_strategy=DistanceStrategy.COSINE) if code_docs else None
        comment_store = FAISS.from_documents(code_comment_docs, self.text_model, distance_strategy=DistanceStrategy.COSINE) if code_comment_docs else None
        
        return text_store, code_store, comment_store
    
    def reindex(self, data: list[tuple[str, str]]):
        for path, mode in tqdm(data, leave=False, desc='Reindexing'):
            try:
                self.reindex_document(path, mode)
            except Exception as e:
                print(f'Reindex document "{path}" failed. exception: {e}')
                traceback.print_exc()
        print(f"✅ Reindexed {len(data)} documents.")
        
        with open(os.path.join(self.root_path, 'config.json'), 'w') as f:
            json.dump(self.config, f, indent=4)
        with open(self.indices_path, 'w') as f:
            json.dump(self.indices_cache, f, indent=4)
        
        self.doc_code_indices.clear()
        self.doc_text_indices.clear()
        self.doc_comment_indices.clear()
    
    def reindex_document(self, doc_path: str, mode: str):
        path = os.path.normpath(os.path.relpath(doc_path, self.root_path))
        doc_rel_path = os.path.normpath(os.path.relpath(doc_path, os.path.join(self.root_path, 'docs')))
        ids = self.doc_ids.get(path)
        if not is_path_in_directory(doc_path, os.path.join(self.root_path, 'docs')):
            return

        if mode == 'add':
            # 添加新文档
            text, code, comment = self.load_document(os.path.join(self.root_path, doc_path), self.config['mode'], True)
            if text:
                self.text_store.merge_from(text)
            if code:
                self.code_store.merge_from(code)
            if comment:
                self.code_comment_store.merge_from(comment)
            self.config['files'].append(os.path.normpath(os.path.relpath(doc_path, os.path.join(self.root_path, 'docs'))))
            
        elif mode == 'delete':
            # 删除文档
            if ids is None:
                print(f'Warn: Cannot get document identifier for "{doc_path}". This may be a bug for LangBotPluginDocument. Please open an issue with a screenshot for call stack.')
                traceback.print_stack()
                return
                
            text_ids, code_ids, comment_ids = ids
            if len(text_ids) > 0:
                self.text_store.delete(text_ids)
            if len(code_ids) > 0:
                self.code_store.delete(code_ids)
            if len(comment_ids) > 0:
                self.code_comment_store.delete(comment_ids)
            # 缓存也得删
            abs_path = os.path.join(self.root_path, path)
            if self.indices_cache['data'].get(abs_path):
                self.indices_cache['data'].pop(abs_path)
            if self.doc_ids.get(path):
                self.doc_ids.pop(path)
            if doc_rel_path in self.config['files']:
                self.config['files'].remove(doc_rel_path)
            
        elif mode == 'modify':
            # 修改文档，先删除再添加
            if ids is not None:
                self.reindex_document(doc_path, 'delete')
            self.reindex_document(doc_path, 'add')
    
    def merge_documents_one(self, indices: list[FAISS]):
        if not indices:
            return None
        store = indices[0]
        if not store:
            return None
        for one in indices[1:]:
            store.merge_from(one)
        return store
            
    def merge_documents(self):
        """将所有数据库合并"""
        self.text_store = self.merge_documents_one(self.doc_text_indices)
        self.code_store = self.merge_documents_one(self.doc_code_indices)
        self.code_comment_store = self.merge_documents_one(self.doc_comment_indices)

        self.retriever = HybridRetriever(
            text_store=self.text_store,
            code_store=self.code_store,
            code_comment_store=self.code_comment_store,
            classification=Classification(self.root_path, self.config["extensions"]["classification"])
        )
        
        for deleted in self.deleted_docs:
            if self.indices_cache['data'].get(deleted):
                self.indices_cache['data'].pop(deleted)
        
        if self.new_doc > 0:
            print(f"✅ Find {self.new_doc} new documents.")
        if self.modified > 0:
            print(f"✅ Find {self.modified} documents modified.")
        if len(self.deleted_docs) > 0:
            print(f"✅ Find {len(self.deleted_docs)} documents deleted.")
        if self.from_cache > 0:
            print(f"✅ Loaded {self.from_cache} stores from cache.")
        if self.indexed > 0:
            print(f"✅ Index and cached {self.indexed} stores.")
            
        self.doc_code_indices.clear()
        self.doc_text_indices.clear()
        self.doc_comment_indices.clear()
            
        return self.indices_cache

    def search(self, message: str) -> list[Document]:
        return self.retriever.search(message)
    