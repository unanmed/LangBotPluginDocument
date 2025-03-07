from pathlib import Path
from tqdm import tqdm
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from .loader import CodeAwareMDLoader
from .splitter import DocumentSplitter
from .retriever import HybridRetriever

class DocumentParser:
    text_model: HuggingFaceEmbeddings = None
    code_model: HuggingFaceEmbeddings = None
    
    splitter: DocumentSplitter = None
    docs: list[Document] = []
    
    text_store: FAISS = None
    code_store: FAISS = None
    
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
            
    def parse_documents(self):
        self.text_store = FAISS.from_documents([doc for doc in self.docs if not doc.metadata.get("is_code", False)], self.text_model)
        self.code_store = FAISS.from_documents([doc for doc in self.docs if doc.metadata.get("is_code", False)], self.code_model)

        self.retriever = HybridRetriever(text_retriever=self.text_store.as_retriever(), code_retriever=self.code_store.as_retriever())

    def search(self, message: str) -> list[Document]:
        return self.retriever.invoke(message)
    