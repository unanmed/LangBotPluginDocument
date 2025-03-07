import re
from langchain_community.document_loaders.base import BaseLoader
from langchain.schema import Document
from typing import List

def extract_code_blocks(md_text):
    # 匹配带语言标签的代码块
    pattern = r'```([a-zA-Z0-9+]+)\n(.*?)```'
    matches = re.findall(pattern, md_text, re.DOTALL)
    return [(lang.lower().strip(), code.strip()) for lang, code in matches]

class CodeAwareMDLoader(BaseLoader):
    def __init__(self, file_path: str):
        self.file_path = file_path
    
    def load(self) -> List[Document]:
        with open(self.file_path, 'r', encoding='utf-8') as f:
            md_content = f.read()
        
        code_blocks = extract_code_blocks(md_content)
        docs = []
        
        # 将代码块转换为独立文档
        for lang, code in code_blocks:
            metadata = {
                "code_language": lang,
                "source": self.file_path,
                "is_code": True
            }
            docs.append(Document(page_content=code, metadata=metadata))
        
        # 处理非代码部分
        text_blocks = re.split(r'```.*?```', md_content, flags=re.DOTALL)
        for text in text_blocks:
            stripped = text.strip()
            if stripped:
                docs.append(Document(
                    page_content=stripped,
                    metadata={"is_code": False, "source": self.file_path}
                ))
        
        return docs