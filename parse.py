from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from .loader import CodeAwareMDLoader, CodeLoader
from .splitter import DocumentSplitter
from .retriever import HybridRetriever

class DocumentParser:
    text_model: HuggingFaceEmbeddings = None
    code_model: HuggingFaceEmbeddings = None
    
    splitter: DocumentSplitter = None
    docs: list[Document] = []
    
    text_store: FAISS = None
    code_store: FAISS = None
    code_comment_store: FAISS = None
    
    retriever: HybridRetriever = None
    
    def __init__(self, config):
        self.config = config
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
        
        # 判断是否需要文本或代码模型
        need_text = "text-only" in modes or "text-code" in modes
        need_code = "code-only" in modes or "text-code" in modes

        # 加载模型
        if need_text:
            self.text_model = HuggingFaceEmbeddings(model_name=self.config["text_model"])
        if need_code:
            self.code_model = HuggingFaceEmbeddings(model_name=self.config["code_model"])
            
    def load_document(self, doc_path: str, mode: str):
        path = Path(doc_path)
        ext = path.suffix
        
        if ext == ".md":
            loader = CodeAwareMDLoader(path)
            self.docs.extend(self.splitter.split_documents(loader.load(), mode))
        
        else:
            loader = CodeLoader(path)
            self.docs.extend(self.splitter.split_documents(loader.load(), "code-only"))
            
    def parse_documents(self):
        text_docs = [doc for doc in self.docs if not doc.metadata.get("is_code", False)]
        code_docs = [doc for doc in self.docs if doc.metadata.get("is_code", False)]
        
        code_comment_docs: list[Document] = []
        for doc in code_docs:
            comment = doc.metadata.get("comments")
            if comment and comment.strip():
                metadata = { **doc.metadata, "code": doc.page_content }
                metadata.pop("comments")
                code_comment_docs.append(Document(page_content=comment, metadata=metadata))
        
        # print("\n===================================\n".join([f"{doc.page_content}\n-------------\n{doc.metadata.get("code")}" for doc in code_comment_docs]))
        
        if not text_docs:
            text_docs = [Document(page_content="")]
        if not code_docs:
            code_docs = [Document(page_content="")]
        if not code_comment_docs:
            code_comment_docs = [Document(page_content="")]
        
        self.text_store = FAISS.from_documents(text_docs, self.text_model)
        self.code_store = FAISS.from_documents(code_docs, self.code_model)
        self.code_comment_store = FAISS.from_documents(code_comment_docs, self.text_model)

        text_retriever = self.text_store.as_retriever() if self.text_store else None
        code_retriever = self.code_store.as_retriever() if self.code_store else None
        code_comment_retriever = self.code_comment_store.as_retriever() if self.code_comment_store else None

        self.retriever = HybridRetriever(
            text_retriever=text_retriever,
            code_retriever=code_retriever,
            code_comment_retriever=code_comment_retriever
        )

    def search(self, message: str) -> list[Document]:
        return self.retriever.invoke(message)
    