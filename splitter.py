from typing import Iterable
from tqdm import tqdm
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter
from tree_sitter_javascript import language as language_javascript
from tree_sitter_typescript import language_tsx, language_typescript
from tree_sitter_python import language as language_python
from tree_sitter_html import language as language_html
from tree_sitter import Language, Parser, Node

def language_text():
    pass

def language_antlr():
    pass

# 将不同语言及其缩写映射为语言名称
# 若一个文件的扩展名也是下面中的一个，也会自动使用对应的语言解析器解析
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
    "": "text",
    "antlr": "antlr",
    "html": "html",
    "htm": "html",
}

# 不同语言名称会使用的解析器
parser_map: dict = {
    "javascript": language_javascript,
    "typescript": language_typescript,
    "tsx": language_tsx,
    "python": language_python,
    "text": language_text,
    "antlr": language_antlr,
    "html": language_html,
}

class CodeSplitter(TextSplitter):
    parser: dict[str, Parser] = {}
    """对应语言当前的解析器实例"""
    markdown_splitter: RecursiveCharacterTextSplitter
    """在无法使用语言解析器时，使用 md 分割器"""
    
    chunk_size: int = 500
    chunk_overlap: int = 100
    
    def __init__(self, md_splitter: RecursiveCharacterTextSplitter, chunk_size=500, chunk_overlap=100, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.markdown_splitter = md_splitter
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
    def check_code_parser(self, lang: str) -> Parser | None:
        """检查并获取对应语言的解析器"""
        lang_name = languages_map[lang]
        
        if lang_name in self.parser:
            return self.parser[lang_name]
        
        obj: object = parser_map[lang_name]()
        
        if not obj:
            return None
        
        parser = Parser(Language(obj))
        self.parser[lang_name] = parser
        return parser
    
    def collect_comments(self, node: Node, code: str, comments=None) -> list[object]:
        """递归遍历语法树并收集注释

        Args:
            node (Node): 根节点
            code (str): 全部代码
            comments (_type_, optional): 当前已经包含的注释. Defaults to None.

        Returns:
            list[object]: 注释信息
        """
        if comments is None:
            comments = []
        if node.type in {"comment", "line_comment", "block_comment"}:
            start_line, start_column = node.start_point
            end_line, end_column = node.end_point
            comment_text = code[node.start_byte:node.end_byte].decode("utf-8")
            comments.append({
                'start_line': start_line,
                'start_column': start_column,
                'end_line': end_line,
                'end_column': end_column,
                'start_byte': node.start_byte,
                'end_byte': node.end_byte,
                'content': comment_text
            })
        for child in node.children:
            self.collect_comments(child, code, comments)
        return comments
    
    def split_code(self, code: Document) -> list[Document]:
        text = code.page_content
        
        if len(text) <= self.chunk_size:
            return [code]
        
        parser: Parser = self.check_code_parser(code.metadata.get("code_language", "python"))
        
        if not parser:
            return self.markdown_splitter.split_documents([code])
        
        line_data = text.splitlines()
        unicode = text.encode("utf-8")
        
        tree = parser.parse(unicode)
        root_node = tree.root_node
        
        comment_data = self.collect_comments(root_node, unicode)
        split_line_index: set[int] = set()
        
        # 如果这一行以注释开头（包含前导缩进），那么这一行可以分段
        for comment in comment_data:
            pre = line_data[comment["start_line"]][:comment["start_column"]]
            if pre.strip() == "":
                split_line_index.add(comment["start_line"])
        
        # 如果是纯空行，也可以分段
        for i, line in enumerate(line_data):      
            if line.strip() == "":
                split_line_index.add(i)
        
        line_length = 0
        comment_index = 0
        line_splitted: list[str] = []
        prev_context: list[str] = []
        docs: list[Document] = []
        
        def append_document(line_index: int):
            """添加当前部分进入文档列表

            Args:
                line_index (int): 这段文档在哪一行结束，不包括这一行
            """
            nonlocal comment_index, line_length
            content = "\n".join(prev_context + line_splitted)
            
            prev_context.clear()
            now_length = 0
            for prev in reversed(line_splitted):
                now_length += len(prev)
                prev_context.append(prev)
                if now_length > self.chunk_overlap:
                    prev_context.reverse()
                    break
            
            comments: list[str] = []
            
            # 把这部分的注释提取出来
            for i, comment in enumerate(comment_data[comment_index:]):
                if comment["start_line"] < line_index:
                    comments.append(comment["content"])
                else:
                    comment_index += i
                    break
            
            metadata = { **code.metadata, "comments": "\n".join(comments) }
            docs.append(Document(page_content=content, metadata=metadata))
            
            # 记得把状态回归到初始状态
            line_splitted.clear()
            line_length = 0
        
        for i, line in enumerate(line_data):
            length = len(line)
            line_length += length
            
            if line_length > self.chunk_size * 2:
                # 超过 chunk_size 的二倍，强制分段
                append_document(i)
            
            if i in split_line_index:
                # 假如这一行可以分段，那么检查长度并决定分不分段
                if line_length > self.chunk_size // 2:
                    append_document(i)
            
            line_splitted.append(line)
        
        if line_splitted:
            append_document(len(line_data))
        
        return docs
    
    def split_documents(self, documents: Iterable[Document]):
        # 如果包含多个文档，那么合并成一个
        docs = list(documents)
        
        if not docs:
            return []
        
        # now_lang = docs[0].metadata.get("code_language", "python")
        # now_source = docs[0].metadata.get("source", "unknown")
        # now_docs: list[Document] = []
        # to_parse: list[Document] = []
        
        # def add_to_parse():
        #     """辅助函数：将当前批次文档合并并添加到 to_parse 列表中。"""
        #     if now_docs:
        #         to_parse.append(Document(
        #             page_content="\n".join(doc.page_content for doc in now_docs),
        #             metadata={"is_code": True, "code_language": now_lang, "source": now_source}
        #         ))

        # for doc in docs:
        #     doc_lang = doc.metadata.get("code_language", "python")
        #     doc_source = doc.metadata.get("source", "unknown")
            
        #     if doc_lang == now_lang and doc_source == now_source:
        #         now_docs.append(doc)
        #     else:
        #         add_to_parse()  # 处理上一批次
        #         now_docs = [doc]  # 重置为新的批次
        #         now_lang, now_source = doc_lang, doc_source
        
        res: list[Document] = []
        
        # 对代码执行分割
        for doc in docs:
            res.extend(self.split_code(doc))
        
        return res
    
    def split_text(self, text):
        raise SystemError("split_text is not supported by this plugin, use split_documents instead.")


class DocumentSplitter(TextSplitter):
    markdown_splitter: RecursiveCharacterTextSplitter
    """md 文档的分割器"""
    code_splitter: CodeSplitter
    
    code_context_length: int = 1
    """在解析代码块时，联系上下文的文本长度"""
    
    chunk_size: int = 500
    chunk_overlap: int = 100
    
    def __init__(self, code_context_length=1, chunk_size=500, chunk_overlap=100, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.code_context_length = code_context_length
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.markdown_splitter = RecursiveCharacterTextSplitter(
            separators=[
                "\n# ",      # 标题
                "\n\n",      # 段落
            ],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.code_splitter = CodeSplitter(md_splitter=self.markdown_splitter ,chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    def split_code(self, code: Document) -> list[Document]:
        """分割代码部分"""
        text = code.page_content
        if len(text) <= self.chunk_size // 2:
            return [code]
        
        splitted = self.code_splitter.split_documents([code])
        
        if not splitted:
            return self.split_text_content(code)
        
        return splitted
    
    def split_text_content(self, text: Document) -> list[Document]:
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
                        tqdm.write(f"Warn: Unknown code language: {lang}. Please refer to README to find solution.")
                    
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

        return chunks

    def split_text(self, text):
        raise SystemError("split_text is not supported by this plugin, use split_documents instead.")

