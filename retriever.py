from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStoreRetriever

class HybridRetriever(BaseRetriever):
    text_retriever: VectorStoreRetriever
    code_retriever: VectorStoreRetriever
        
    def _get_relevant_documents(self, query: str):
        # 检测查询类型
        is_code_query = any(kw in query.lower() for kw in ["function", "class", "def", "如何实现"])
        
        # 调整权重比例
        code_weight = 0.7 if is_code_query else 0.3
        text_weight = 1 - code_weight
        
        # 分路检索
        text_docs = self.text_retriever.invoke(query)
        code_docs = self.code_retriever.invoke(query)
        
        res = []
        
        if text_docs:
            res.extend(text_docs[:3])
        if code_docs:
            res.extend(code_docs[:3])
            
        return res
        
        # 混合排序算法
        # combined = []
        # for doc in text_docs:
        #     doc.metadata["score"] *= text_weight
        #     combined.append(doc)
        # for doc in code_docs:
        #     doc.metadata["score"] *= code_weight
        #     combined.append(doc)
            
        # return sorted(combined, key=lambda x: x.metadata["score"], reverse=True)[:5]