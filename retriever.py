from collections import deque
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStoreRetriever

class HybridRetriever(BaseRetriever):
    text_retriever: VectorStoreRetriever
    code_retriever: VectorStoreRetriever
    code_comment_retriever: VectorStoreRetriever
        
    def _get_relevant_documents(self, query: str):
        res = []

        # 初始化 deque
        text_docs = deque(self.text_retriever.invoke(query)) if self.text_retriever else deque()
        code_docs = deque(self.code_retriever.invoke(query)) if self.code_retriever else deque()
        code_comment_docs = deque(self.code_comment_retriever.invoke(query)) if self.code_comment_retriever else deque()

        # 如果所有检索器为空，直接返回空列表
        if not any([text_docs, code_docs, code_comment_docs]):
            return res

        # 获取相关文档直到满足条件
        while len(res) < 6:
            # 获取每个文档来源的元素
            text_doc = text_docs.popleft() if text_docs else None
            code_doc = code_docs.popleft() if code_docs else None
            code_comment_doc = code_comment_docs.popleft() if code_comment_docs else None

            # 添加到结果中
            if text_doc:
                res.append(text_doc)
            if code_doc:
                res.append(code_doc)
            if code_comment_doc:
                res.append(code_comment_doc)

            # 如果没有更多文档可供检索，结束循环
            if not any([text_doc, code_doc, code_comment_doc]):
                break

        return res
