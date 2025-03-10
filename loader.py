import re
from pathlib import Path
from langchain_community.document_loaders.base import BaseLoader
from langchain.schema import Document
from typing import List
from .splitter import languages_map

def extract_language(line: str) -> str:
    match = re.match(r"^```(\w+)", line.strip())
    return match.group(1) if match else ""

class CodeAwareMDLoader(BaseLoader):
    def __init__(self, file_path: str):
        self.file_path = file_path
    
    def load(self) -> List[Document]:
        with open(self.file_path, 'r', encoding='utf-8') as f:
            md_content = f.read()
        
        line_data = md_content.splitlines()
        in_code = False
        now_lang = "python"
        now_lines: list[str] = []
        docs = []
        
        def append_document():
            content = "\n".join(now_lines).strip()
            if in_code:
                metadata = {
                    "code_language": now_lang,
                    "source": self.file_path,
                    "is_code": True
                }
                docs.append(Document(page_content=content, metadata=metadata))
            
            else:
                metadata = {
                    "source": self.file_path,
                    "is_code": False
                }
                docs.append(Document(page_content=content, metadata=metadata))
            
            now_lines.clear()
        
        for line in line_data:
            stripped = line.strip()
            if stripped.startswith("```"):
                if in_code:
                    append_document()
                    in_code = False
                
                else:
                    now_lang = extract_language(stripped).strip()
                    append_document()
                    in_code = True
                
            else:
                now_lines.append(line)
        
        if now_lines:
            append_document()
        
        return docs
    
class CodeLoader(BaseLoader):
    def __init__(self, file_path: str):
        self.file_path = file_path
        
    def load(self) -> List[Document]:
        with open(self.file_path, 'r', encoding='utf-8') as f:
            code_content = f.read()
            
        path = Path(self.file_path)
        ext = path.suffix[1:]
        
        if ext in languages_map:
            metadata = {"is_code": True, "source": self.file_path, "code_language": languages_map[ext]}
            return [Document(page_content=code_content, metadata=metadata)]
        else:
            print(f"Warn: Unknown file extension '{ext}', please refer to README and splitter.py to find solutions.")
            metadata = {"is_code": True, "source": self.file_path, "code_language": languages_map["text"]}
            return [Document(page_content=code_content, metadata=metadata)]
