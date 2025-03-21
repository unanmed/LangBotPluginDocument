import hashlib
import os
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_community.vectorstores import FAISS, DistanceStrategy
from .loader import CodeAwareMDLoader, CodeLoader
from .splitter import DocumentSplitter
from .retriever import HybridRetriever
from .extensions.classification import Classification

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
    config: object = None
    indices_cache: dict[str, object] = None
    doc_text_indices: list[FAISS] = list()
    doc_code_indices: list[FAISS] = list()
    doc_comment_indices: list[FAISS] = list()
    
    deleted_docs: set[str] = set()
    max_id: int = 0
    
    from_cache: int = 0
    modified: int = 0
    new_doc: int = 0
    indexed: int = 0
    
    def __init__(self, config, indices_cache: dict[str, object], root: str):
        self.config = config
        self.indices_cache = indices_cache
        self.root_path = root
        self.deleted_docs = { os.path.join(root, 'docs', path) for path in config["files"] }
        if len(indices_cache) == 0:
            self.max_id = 0
        else:
            self.max_id = max([int(index["id"]) for index in indices_cache.values()])
        self.splitter = DocumentSplitter(
            code_context_length=config["code_context_length"],
            chunk_size=config["chunk_size"],
            chunk_overlap=config["chunk_overlap"]
        )
    
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
        indices = self.indices_cache.get(doc_path)
        
        if not indices:
            self.new_doc += 1
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
        
        indices = self.indices_cache.get(doc_path)
        if indices:
            index_id = indices["id"]
        else:
            self.max_id += 1
            index_id = self.max_id
            
        text_path = os.path.join(self.root_path, "data/text", f"doc_{index_id}")
        code_path = os.path.join(self.root_path, "data/code", f"doc_{index_id}")
        comment_path = os.path.join(self.root_path, "data/comment", f"doc_{index_id}")
        
        with open(doc_path, 'r', encoding='utf-8') as doc:
            file_hash = hashlib.sha256(doc.read().encode()).hexdigest()
            
        self.indices_cache[doc_path] = {
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
            
    def load_document(self, doc_path: str, mode: str):
        path = Path(doc_path)
        ext = path.suffix
        
        if self.check_cache(doc_path):
            # 有缓存，直接从缓存加载
            indices = self.indices_cache.get(doc_path)
            if indices["text_path"] and os.path.exists(indices["text_path"]):
                self.doc_text_indices.append(FAISS.load_local(
                    indices["text_path"], self.text_model, "index", allow_dangerous_deserialization=True,
                    distance_strategy=DistanceStrategy.COSINE
                ))
            if indices["code_path"] and os.path.exists(indices["code_path"]):
                self.doc_code_indices.append(FAISS.load_local(
                    indices["code_path"], self.code_model, "index", allow_dangerous_deserialization=True,
                    distance_strategy=DistanceStrategy.COSINE
                ))
            if indices["comment_path"] and os.path.exists(indices["comment_path"]):
                self.doc_comment_indices.append(FAISS.load_local(
                    indices["comment_path"], self.text_model, "index", allow_dangerous_deserialization=True,
                    distance_strategy=DistanceStrategy.COSINE
                ))
            self.deleted_docs.remove(doc_path)
            
        else:
            # 没有缓存，加载文档并索引
            if ext == ".md":
                loader = CodeAwareMDLoader(path)
                docs = self.splitter.split_documents(loader.load(), mode)
            
            else:
                loader = CodeLoader(path)
                docs = self.splitter.split_documents(loader.load(), "code-only")
                
            text, code, comment = self.parse_one_document(docs)
            if text:
                self.doc_text_indices.append(text)
            if code:
                self.doc_code_indices.append(code)
            if comment:
                self.doc_comment_indices.append(comment)
            self.cache_index(doc_path, text, code, comment)
            self.indexed += 1
            self.deleted_docs.remove(doc_path)

    def parse_one_document(self, docs: list[Document]):
        text_docs = [doc for doc in docs if not doc.metadata.get("is_code", False)]
        code_docs = [doc for doc in docs if doc.metadata.get("is_code", False)]
        
        code_comment_docs: list[Document] = []
        for doc in code_docs:
            comment = doc.metadata.get("comments")
            if comment and comment.strip():
                metadata = { **doc.metadata, "code": doc.page_content }
                metadata.pop("comments")
                code_comment_docs.append(Document(page_content=comment, metadata=metadata))
            
        text_store = FAISS.from_documents(text_docs, self.text_model, distance_strategy=DistanceStrategy.COSINE) if text_docs else None
        code_store = FAISS.from_documents(code_docs, self.code_model, distance_strategy=DistanceStrategy.COSINE) if code_docs else None
        comment_store = FAISS.from_documents(code_comment_docs, self.text_model, distance_strategy=DistanceStrategy.COSINE) if code_comment_docs else None
        
        return text_store, code_store, comment_store
    
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
            
        text_retriever = self.text_store.as_retriever() if self.text_store else None
        code_retriever = self.code_store.as_retriever() if self.code_store else None
        code_comment_retriever = self.code_comment_store.as_retriever() if self.code_comment_store else None

        self.retriever = HybridRetriever(
            text_retriever=text_retriever,
            code_retriever=code_retriever,
            code_comment_retriever=code_comment_retriever,
            classification=Classification(self.root_path, self.config["extensions"]["classification"])
        )
        
        for deleted in self.deleted_docs:
            self.indices_cache.pop(deleted)
        
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
    