"""Unlearning 数据集

支持多种数据格式和加载方式
"""
import json
import random
from typing import Dict, List, Optional, Union
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class UnlearnDataset(Dataset):
    """Unlearning 数据集：同时包含 forget 和 retain 数据
    
    支持的数据格式：
    1. JSONL 格式：每行一个 JSON 对象
    2. JSON 格式：包含列表的 JSON 文件
    
    每个样本应包含：
    - question: 问题
    - answer: 答案（用于 forget）
    - alternate_answer (可选): 替代答案（用于 DPO 的 win_inputs）
    """
    
    def __init__(
        self,
        forget_path: str,
        retain_path: Optional[str] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        max_length: int = 512,
        question_key: str = "question",
        answer_key: str = "answer",
        alternate_answer_key: str = "alternate_answer",
        sample_retain: bool = True,
        seed: int = 42,
    ):
        """
        Args:
            forget_path: 遗忘集数据路径
            retain_path: 保留集数据路径（可选）
            tokenizer: Tokenizer
            max_length: 最大序列长度
            question_key: 问题字段名
            answer_key: 答案字段名
            alternate_answer_key: 替代答案字段名（用于 DPO）
            sample_retain: 是否对 retain 数据进行采样（每个 forget 样本配一个 retain）
            seed: 随机种子
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.question_key = question_key
        self.answer_key = answer_key
        self.alternate_answer_key = alternate_answer_key
        self.sample_retain = sample_retain
        
        random.seed(seed)
        
        # 加载遗忘集
        self.forget_data = self._load_data(forget_path)
        
        # 加载保留集
        self.retain_data = []
        if retain_path:
            self.retain_data = self._load_data(retain_path)
    
    def _load_data(self, path: str) -> List[Dict]:
        """加载数据文件"""
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            if path.endswith('.jsonl'):
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
            else:  # JSON 格式
                data = json.load(f)
                if isinstance(data, dict):
                    # 如果是字典，尝试找到数据列表
                    for key in ['data', 'examples', 'samples']:
                        if key in data:
                            data = data[key]
                            break
        return data
    
    def __len__(self) -> int:
        return len(self.forget_data)
    
    def __getitem__(self, idx: int) -> Dict:
        """获取一个样本"""
        forget_item = self.forget_data[idx]
        
        # 处理 forget 样本
        forget_text = self._format_text(
            forget_item,
            use_alternate=False
        )
        forget_enc = self.tokenizer(
            forget_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        result = {
            "forget": {
                "input_ids": forget_enc["input_ids"].squeeze(0),
                "attention_mask": forget_enc["attention_mask"].squeeze(0),
                "labels": forget_enc["input_ids"].squeeze(0),
            }
        }
        
        # 如果有 alternate_answer，添加用于 DPO
        if self.alternate_answer_key in forget_item:
            alternate_text = self._format_text(
                forget_item,
                use_alternate=True
            )
            alternate_enc = self.tokenizer(
                alternate_text,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )
            result["forget"]["original"] = result["forget"].copy()
            result["forget"]["alternate"] = {
                "input_ids": alternate_enc["input_ids"].squeeze(0),
                "attention_mask": alternate_enc["attention_mask"].squeeze(0),
                "labels": alternate_enc["input_ids"].squeeze(0),
            }
        
        # 如果有 retain 数据，采样一个
        if self.retain_data:
            if self.sample_retain:
                retain_item = random.choice(self.retain_data)
            else:
                # 循环使用
                retain_idx = idx % len(self.retain_data)
                retain_item = self.retain_data[retain_idx]
            
            retain_text = self._format_text(retain_item, use_alternate=False)
            retain_enc = self.tokenizer(
                retain_text,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )
            result["retain"] = {
                "input_ids": retain_enc["input_ids"].squeeze(0),
                "attention_mask": retain_enc["attention_mask"].squeeze(0),
                "labels": retain_enc["input_ids"].squeeze(0),
            }
        
        return result
    
    def _format_text(self, item: Dict, use_alternate: bool = False) -> str:
        """格式化文本
        
        Args:
            item: 数据项
            use_alternate: 是否使用替代答案（用于 DPO）
        """
        question = item.get(self.question_key, "")
        
        if use_alternate:
            answer = item.get(self.alternate_answer_key, item.get(self.answer_key, ""))
        else:
            answer = item.get(self.answer_key, "")
        
        # 简单的格式化，可以根据模型调整
        # 对于 chat 模型，可以使用 apply_chat_template
        if self.tokenizer and hasattr(self.tokenizer, 'apply_chat_template'):
            messages = [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer}
            ]
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
        else:
            return f"Question: {question}\nAnswer: {answer}"

