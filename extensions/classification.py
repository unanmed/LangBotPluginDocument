import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class Classification:
    config: object = None
    
    def __init__(self, root: str, config: object):
        self.config = config
        if self.enabled():
            model_path = os.path.join(root, config.get("model_path"))
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
    def enabled(self):
        return self.config.get("enable", False)
    
    def classify_and_sort(self, query: str, code: list, comment: list, text: list):
        if not self.enabled():
            return []
        # 编码输入
        inputs = self.tokenizer(query, return_tensors="pt", truncation=True, padding=True)
        
        # 获取模型输出
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # 如果是分类任务，一般输出 logits
        logits = outputs.logits
        sigmoid_logits = torch.sigmoid(logits)
        
        value1 = sigmoid_logits[0, 0].item()
        value2 = sigmoid_logits[0, 1].item()
        
        # 文本至少有 0.2 的权重
        code_weight = value1 * 0.8
        need_doc = value2 > self.config.get("need_doc_threshold")
        
        if not need_doc:
            return []
        
        results: list[tuple] = []
        
        # 对 list1 中的每个元组乘以对应权重
        for item, score in code:
            weighted_score = (1 - score) * code_weight
            results.append((item, weighted_score))
        
        # 对 list2 中的每个元组乘以对应权重
        for item, score in comment:
            weighted_score = (1 - score) * code_weight
            results.append((item, weighted_score))
        
        # 对 list3 中的每个元组乘以对应权重
        for item, score in text:
            weighted_score = (1 - score) * (1 - code_weight)
            results.append((item, weighted_score))
            
        results_sorted = sorted(results, key=lambda x: x[1], reverse=True)
        
        return results_sorted[:6]
        