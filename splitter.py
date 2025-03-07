from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter
from tree_sitter_javascript import language as language_javascript
from tree_sitter_typescript import language_tsx, language_typescript
from tree_sitter_python import language as language_python
from tree_sitter import Language, Parser

def language_text():
    pass

def language_antlr():
    pass

# 将不同语言及其缩写映射为语言名称
languages_map = {
    "js": "javascript",
    "javascript": "javascript",
    "jsx": "javascript",
    "mjs": "javascript",
    "cjs": "javascript",
    "ts": "typescript",
    "typescript": "typescript",
    "tsx": "tsx",
    "cts": "typescript",
    "mts": "typescript",
    "python": "python",
    "py": "python",
    "pyi": "python",
    "text": "text",
    "txt": "text",
    "antlr": "antlr"
}

# 不同语言名称会使用的解析器
parser_map = {
    "javascript": language_javascript,
    "typescript": language_typescript,
    "tsx": language_tsx,
    "python": language_python,
    "text": language_text,
    "antlr": language_antlr
}

class DocumentSplitter(TextSplitter):
    parser: dict[str, Parser] = {}
    """对应语言当前的解析器实例"""
    
    markdown_splitter: RecursiveCharacterTextSplitter
    """md 文档的粗分割器"""
    
    code_context_length: int = 1
    """在解析代码块时，联系上下文的文本长度"""
    
    def __init__(self, code_context_length=1, chunk_size=500, chunk_overlap=100, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.code_context_length = code_context_length
        self.markdown_splitter = RecursiveCharacterTextSplitter(
            separators=[
                "\n# ",      # 标题
                "\n\n",      # 段落
            ],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    def split_code(self, code: Document) -> list[Document]:
        """分割代码部分"""
        text = code.page_content
        if len(text) <= 500:
            return [code]
        
        # TODO 代码分割
        
        return [code]
    
    def split_text_content(self, text: list[Document]) -> list[Document]:
        """分割文本部分"""
        return self.markdown_splitter.split_documents([text])
    
    def split_documents(self, documents: list[Document], mode: str) -> list[Document]:
        chunks: list[Document] = []
        
        # 定义模式到处理函数的映射
        mode_to_splitter = {
            "text-only": self.split_text_content,
            "code-only": self.split_code,
        }

        # 处理 "text-only" 和 "code-only" 模式
        if mode in mode_to_splitter:
            splitter = mode_to_splitter[mode]
            for doc in documents:
                chunks.extend(splitter(doc))
            return chunks

        # 处理 "text-code" 模式
        if mode == "text-code":
            chunk_map: dict[int, list[Document]] = {}
            
            # 先确定每个文档的拆分函数
            for i, doc in enumerate(documents):
                if doc.metadata.get("is_code"):
                    lang = doc.metadata.get("code_language", "").lower().strip()
                    if languages_map.get(lang):
                        splitter = self.split_code
                    else:
                        splitter = self.split_text_content
                        print(f"Warn: Unknown code language: {lang}. Please refer to READEME to find solution.")
                    
                else:
                    splitter = self.split_text_content

                # 使用字典暂存结果
                chunk_map[i] = splitter(doc)

            # 遍历字典，然后对片段注入上下文
            for i, splitted in chunk_map.items():
                if not splitted:
                    continue
                
                prev_context = "\n".join([doc.page_content for doc in chunk_map.get(i - 1, [])[-self.code_context_length:]])
                next_context = "\n".join([doc.page_content for doc in chunk_map.get(i + 1, [])[:self.code_context_length]])

                if prev_context:
                    splitted[0].metadata["prev_context"] = prev_context
                if next_context:
                    splitted[-1].metadata["next_context"] = next_context

                chunks.extend(splitted)
                
        print(chunks)

        return chunks

    def split_text(self, text):
        raise SystemError("split_text is not supported by this plugin, use split_documents instead.")
