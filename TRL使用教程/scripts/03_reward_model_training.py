#!/usr/bin/env python3
"""
奖励模型训练示例
演示如何训练一个简单的奖励模型来评估文本质量
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments
from datasets import Dataset
import numpy as np
from sklearn.metrics import accuracy_score

class RewardModel(nn.Module):
    """简单的奖励模型"""
    
    def __init__(self, model_name="gpt2"):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.reward_head = nn.Linear(self.backbone.config.hidden_size, 1)
        
    def forward(self, input_ids, attention_mask=None):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # 使用最后一个token的hidden state
        last_hidden_state = outputs.last_hidden_state[:, -1, :]
        reward = self.reward_head(last_hidden_state)
        return reward

def create_preference_dataset():
    """创建偏好对比数据集"""
    
    # 示例数据：每个样本包含chosen(好的回复)和rejected(差的回复)
    data = [
        {
            "prompt": "今天天气很好，",
            "chosen": "我想出去散步，享受这美好的阳光。",
            "rejected": "呃呃呃呃呃呃呃呃呃呃呃。"
        },
        {
            "prompt": "推荐一本好书：",
            "chosen": "我推荐《三体》，这是一部优秀的科幻小说，情节引人入胜。",
            "rejected": "书书书书书书书书书书。"
        },
        {
            "prompt": "解释什么是AI：",
            "chosen": "AI是人工智能的缩写，是模拟人类智能的计算机系统。",
            "rejected": "AI就是AI啊AI就是AI。"
        },
        {
            "prompt": "给个健康建议：",
            "chosen": "建议每天保持适量运动，均衡饮食，充足睡眠。",
            "rejected": "吃吃吃睡睡睡玩玩玩。"
        },
        {
            "prompt": "描述春天：",
            "chosen": "春天万物复苏，花儿绽放，温暖的阳光洒在大地上。",
            "rejected": "春天就是春天春天春天春天。"
        }
    ] * 20  # 重复创建更多样本
    
    return data

def prepare_training_data(preference_data, tokenizer):
    """准备训练数据"""
    
    all_texts = []
    all_labels = []
    
    for item in preference_data:
        prompt = item["prompt"]
        chosen = prompt + item["chosen"]
        rejected = prompt + item["rejected"]
        
        all_texts.extend([chosen, rejected])
        all_labels.extend([1.0, 0.0])  # chosen=1, rejected=0
    
    # Tokenize
    encodings = tokenizer(
        all_texts,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    )
    
    return Dataset.from_dict({
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "labels": all_labels
    })

class RewardTrainer(Trainer):
    """自定义奖励模型训练器"""
    
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        rewards = outputs.squeeze(-1)
        
        # 使用MSE损失
        loss = nn.MSELoss()(rewards, torch.tensor(labels, device=rewards.device, dtype=torch.float))
        
        return (loss, outputs) if return_outputs else loss

def train_reward_model():
    """训练奖励模型"""
    
    print("🎯 开始训练奖励模型...")
    
    # 1. 准备数据
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    preference_data = create_preference_dataset()
    train_dataset = prepare_training_data(preference_data, tokenizer)
    
    print(f"训练数据大小: {len(train_dataset)}")
    
    # 2. 创建模型
    model = RewardModel("gpt2")
    
    # 3. 训练参数
    training_args = TrainingArguments(
        output_dir="./models/reward_model",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        warmup_steps=10,
        logging_steps=10,
        save_steps=100,
        evaluation_strategy="no",
        save_strategy="epoch",
        load_best_model_at_end=False,
        report_to=None,  # 不使用wandb
    )
    
    # 4. 创建训练器
    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )
    
    # 5. 开始训练
    print("🚀 开始训练...")
    trainer.train()
    
    # 6. 保存模型
    trainer.save_model()
    print("💾 奖励模型已保存")
    
    return model, tokenizer

def test_reward_model():
    """测试训练好的奖励模型"""
    
    try:
        print("🧪 测试奖励模型...")
        
        # 加载模型
        tokenizer = AutoTokenizer.from_pretrained("./models/reward_model")
        model = RewardModel()
        model.load_state_dict(torch.load("./models/reward_model/pytorch_model.bin"))
        model.eval()
        
        # 测试样本
        test_texts = [
            "今天天气很好，我想出去散步。",  # 好的回复
            "呃呃呃呃呃呃呃呃呃。",         # 差的回复
            "这是一个有意义的回答。",       # 中等回复
        ]
        
        print("\n奖励模型评分结果:")
        for text in test_texts:
            encoding = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128
            )
            
            with torch.no_grad():
                reward = model(**encoding)
                print(f"'{text}' -> 奖励: {reward.item():.3f}")
                
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        print("请先运行训练")

def main():
    print("🎯 奖励模型训练示例")
    
    choice = input("选择操作: (1)训练奖励模型 (2)测试现有模型: ")
    
    if choice == "1":
        train_reward_model()
        print("\n训练完成，可以运行测试了")
    elif choice == "2":
        test_reward_model()
    else:
        print("无效选择")

if __name__ == "__main__":
    main()