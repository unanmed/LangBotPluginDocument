import json
import os
from tqdm import tqdm
from pkg.plugin.context import register, handler, llm_func, BasePlugin, APIHost, EventContext
from pkg.plugin.events import *  # 导入事件类
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

@register(name="LangBotPluginDocument", description="提供文档检索增强（RAG）功能，可以将机器人部署为文档机器人", version="0.1", author="AncTe(unanmed)")
class LangBotPluginDocument(BasePlugin):
    def __init__(self, host: APIHost):
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        
        with open(os.path.join(self.current_dir, "config.json"), 'r', encoding='utf-8') as file:
            data = json.load(file)
            
        self.files = data["files"]
        self.reference_prompt = data["reference_prompt"]
        self.question_prompt = data["question_prompt"]
        self.debug = data["debug"]
        self.texts = []
        
        print("Fetching models...")
        self.embeddings = HuggingFaceEmbeddings(model_name='BAAI/bge-large-zh-v1.5')
        
        self.parse_documents()
        
        print("Vectoring...")
        self.vector_store = FAISS.from_documents(self.texts, self.embeddings)
        self.retriever = self.vector_store.as_retriever(search_kwargs={ "k": 3 })
    
    def parse_documents(self):
        chunker = SemanticChunker(self.embeddings, sentence_split_regex=r'(?<=[。！？])')
        for file in tqdm(self.files, desc="Parsing documents"):
            path = os.path.join(self.current_dir, 'docs', file)
            loader = UnstructuredMarkdownLoader(path)
            documents = loader.load()
            texts = chunker.split_documents(documents)
            self.texts.extend(texts)

    async def initialize(self):
        pass
    
    def handle_RAG(self, message):
        print("Processing RAG")
        docs = self.vector_store.similarity_search(message)
        context = "\n".join([doc.page_content for doc in docs])
        return context

    @handler(PersonNormalMessageReceived)
    async def person_normal_message_received(self, ctx: EventContext):
        msg = ctx.event.text_message
        context = self.handle_RAG(msg)
        handled = f"{self.reference_prompt}\n{context}\n{self.question_prompt}{msg}"
        ctx.event.alter = handled
        if self.debug:
            print(handled)
        

    @handler(GroupNormalMessageReceived)
    async def group_normal_message_received(self, ctx: EventContext):
        msg = ctx.event.text_message
        context = self.handle_RAG(msg)
        handled = f"{self.reference_prompt}\n{context}\n{self.question_prompt}{msg}"
        ctx.event.alter = handled
        if self.debug:
            print(handled)

    def __del__(self):
        self.texts.clear()
        pass
