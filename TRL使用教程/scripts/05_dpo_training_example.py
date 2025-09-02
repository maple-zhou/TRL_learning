#!/usr/bin/env python3
"""
DPO训练示例
演示如何使用TRL进行DPO (Direct Preference Optimization) 训练
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import DPOTrainer, DPOConfig
from datasets import Dataset
import json

def create_dpo_dataset():
    """创建DPO训练数据集"""
    
    # DPO需要偏好对比数据：chosen vs rejected
    dpo_data = [
        {
            "prompt": "解释什么是深度学习",
            "chosen": "深度学习是机器学习的一个分支，使用多层神经网络来模拟人脑的学习过程，能够自动从数据中提取特征和模式。",
            "rejected": "深度学习就是很深的学习，学得很深很深的那种学习。"
        },
        {
            "prompt": "推荐一本编程书籍",
            "chosen": "我推荐《Python编程：从入门到实践》，这本书结构清晰，例子丰富，适合初学者系统学习Python。",
            "rejected": "书书书，推荐书书书书书书书。"
        },
        {
            "prompt": "如何保持健康的生活方式",
            "chosen": "保持健康需要：1)规律作息，早睡早起；2)均衡饮食，多吃蔬果；3)适量运动，每周至少150分钟；4)管理压力，保持良好心态。",
            "rejected": "健康就是健康健康健康健康健康。"
        },
        {
            "prompt": "描述你理想中的工作",
            "chosen": "理想的工作应该有挑战性和成长机会，团队氛围融洽，能够平衡工作与生活，同时对社会有积极贡献。",
            "rejected": "工作工作工作工作工作工作工作。"
        },
        {
            "prompt": "给初学者的学习建议",
            "chosen": "建议初学者：制定明确目标，保持耐心和坚持，多实践少理论，遇到困难要主动寻求帮助，建立良好的学习习惯。",
            "rejected": "学学学学学学学学学学学学。"
        }
    ] * 20  # 重复创建更多样本
    
    return dpo_data

def format_dpo_data(data, tokenizer):
    """格式化DPO数据"""
    
    formatted_data = []
    
    for item in data:
        # DPO训练器期望的数据格式
        formatted_item = {
            "prompt": item["prompt"],
            "chosen": item["chosen"],
            "rejected": item["rejected"]
        }
        formatted_data.append(formatted_item)
    
    return Dataset.from_list(formatted_data)

def train_dpo_model():
    """训练DPO模型"""
    
    print("🎯 开始DPO训练")
    
    # 1. 准备模型和tokenizer
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
    
    # 2. 准备数据
    raw_data = create_dpo_dataset()
    train_dataset = format_dpo_data(raw_data, tokenizer)
    
    print(f"训练数据大小: {len(train_dataset)}")
    print(f"示例数据: {train_dataset[0]}")
    
    # 3. DPO配置
    training_args = TrainingArguments(
        output_dir="./models/dpo_model",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        warmup_steps=10,
        logging_steps=5,
        save_steps=50,
        save_strategy="epoch",
        evaluation_strategy="no",
        report_to=None,
        remove_unused_columns=False,
        dataloader_drop_last=True,
    )
    
    # 4. 创建DPO训练器
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=None,  # 将使用模型副本作为参考
        args=training_args,
        beta=0.1,  # DPO温度参数
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        max_length=128,
        max_prompt_length=64,
    )
    
    print("🚀 开始DPO训练...")
    
    # 5. 开始训练
    dpo_trainer.train()
    
    # 6. 保存模型
    dpo_trainer.save_model()
    tokenizer.save_pretrained("./models/dpo_model")
    
    print("✅ DPO训练完成")
    print("💾 模型已保存到 ./models/dpo_model")

def test_dpo_model():
    """测试DPO训练后的模型"""
    
    try:
        print("🧪 测试DPO模型...")
        
        # 加载模型
        tokenizer = AutoTokenizer.from_pretrained("./models/dpo_model")
        model = AutoModelForCausalLM.from_pretrained("./models/dpo_model")
        
        test_prompts = [
            "解释什么是深度学习",
            "推荐一本编程书籍", 
            "如何保持健康的生活方式",
            "给初学者的学习建议"
        ]
        
        print("\n🎯 DPO模型生成结果:")
        print("=" * 50)
        
        for prompt in test_prompts:
            inputs = tokenizer.encode(prompt, return_tensors="pt")
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_length=100,
                    do_sample=True,
                    top_p=0.9,
                    temperature=0.7,
                    pad_token_id=tokenizer.pad_token_id
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_part = response[len(prompt):].strip()
            
            print(f"\n提示: {prompt}")
            print(f"回复: {generated_part}")
            print("-" * 30)
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        print("请先运行DPO训练")

def compare_with_original():
    """比较DPO训练前后的差异"""
    
    print("🔍 比较原始模型和DPO模型")
    
    # 加载原始模型
    original_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    original_model = AutoModelForCausalLM.from_pretrained("gpt2")
    
    if original_tokenizer.pad_token is None:
        original_tokenizer.pad_token = original_tokenizer.eos_token
    
    try:
        # 加载DPO模型
        dpo_tokenizer = AutoTokenizer.from_pretrained("./models/dpo_model")
        dpo_model = AutoModelForCausalLM.from_pretrained("./models/dpo_model")
        
        test_prompt = "给初学者的学习建议"
        
        print(f"\n测试提示: '{test_prompt}'")
        print("=" * 60)
        
        # 原始模型生成
        inputs = original_tokenizer.encode(test_prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = original_model.generate(
                inputs, max_length=80, do_sample=True, 
                top_p=0.9, temperature=0.7,
                pad_token_id=original_tokenizer.pad_token_id
            )
        original_response = original_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # DPO模型生成
        inputs = dpo_tokenizer.encode(test_prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = dpo_model.generate(
                inputs, max_length=80, do_sample=True,
                top_p=0.9, temperature=0.7,
                pad_token_id=dpo_tokenizer.pad_token_id
            )
        dpo_response = dpo_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print("📖 原始GPT2:")
        print(original_response[len(test_prompt):].strip())
        
        print("\n🎯 DPO训练后:")
        print(dpo_response[len(test_prompt):].strip())
        
    except Exception as e:
        print(f"❌ 比较失败: {e}")
        print("请先训练DPO模型")

def main():
    print("🎯 DPO (Direct Preference Optimization) 示例")
    
    while True:
        print("\n选择操作:")
        print("1. 训练DPO模型")
        print("2. 测试DPO模型")
        print("3. 比较训练前后效果")
        print("4. 退出")
        
        choice = input("请选择 (1-4): ")
        
        if choice == "1":
            train_dpo_model()
        elif choice == "2":
            test_dpo_model()
        elif choice == "3":
            compare_with_original()
        elif choice == "4":
            print("👋 退出")
            break
        else:
            print("❌ 无效选择")

if __name__ == "__main__":
    main()