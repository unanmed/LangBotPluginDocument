import json
import os
from tqdm import tqdm
from pkg.plugin.context import register, handler, llm_func, BasePlugin, APIHost, EventContext
from pkg.plugin.events import *  # 导入事件类
from .parse import DocumentParser

@register(name="LangBotPluginDocument", description="提供文档检索增强（RAG）功能，可以将机器人部署为文档机器人", version="0.1", author="AncTe(unanmed)")
class LangBotPluginDocument(BasePlugin):
    parser: DocumentParser = None
    """文档解析器"""
    current_dir: str = os.path.dirname(os.path.abspath(__file__))
    """当前文件路径，用于获取文档路径"""
    
    def __init__(self, host: APIHost):
        print("=============== Loading LangBot Document Plugin ===============")
        os.makedirs(os.path.join(self.current_dir, "data"), exist_ok=True)
        os.makedirs(os.path.join(self.current_dir, "data/text"), exist_ok=True)
        os.makedirs(os.path.join(self.current_dir, "data/code"), exist_ok=True)
        os.makedirs(os.path.join(self.current_dir, "data/comment"), exist_ok=True)
        indices_path = os.path.join(self.current_dir, "indices.json")
        if not os.path.exists(indices_path):
            with open(indices_path, 'w') as f:
                f.write("{}")
        
        with open(os.path.join(self.current_dir, "config.json"), 'r', encoding='utf-8') as file:
            data = json.load(file)
        with open(indices_path, 'r', encoding='utf-8') as indices:
            indices_cache = json.load(indices)
            
        self.parser = DocumentParser(data, indices_cache, self.current_dir)
        
        self.reference_prompt = data["reference_prompt"]
        self.question_prompt = data["question_prompt"]
        self.debug = data["debug"]
        
        print("Fetching models...")
        
        self.parser.fetch_models()
        
        for path in tqdm(data["files"], desc="Loading and parsing documents"):
            if isinstance(path, str):
                self.parser.load_document(os.path.join(self.current_dir, "docs", path), data["mode"])
            else:
                self.parser.load_document(os.path.join(self.current_dir, "docs", path["path"]), path["mode"])
        
        cache = self.parser.merge_documents()
        
        with open(indices_path, 'w', encoding='utf-8') as indices:
            json.dump(cache, indices, ensure_ascii=False, indent=4)

        print("=============== Loaded LangBot Document Plugin ===============")

    async def initialize(self):
        pass
    
    def handle_RAG(self, message):
        print("Processing RAG")
        docs = self.parser.search(message)
        text = "\n---\n".join(
            f"{doc.metadata.get('prev_context', '')}\n"
            f"{doc.metadata.get("code", doc.page_content)}\n"
            f"{doc.metadata.get('next_context', '')}"
            for doc in docs
        )

        return text

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
        pass
