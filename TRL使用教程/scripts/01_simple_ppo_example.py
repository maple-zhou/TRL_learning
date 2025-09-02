#!/usr/bin/env python3
"""
简单的PPO训练示例
这个脚本演示如何使用TRL进行基础的PPO训练
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from datasets import Dataset
import numpy as np

def create_sample_dataset():
    """创建一个简单的示例数据集"""
    prompts = [
        "写一首关于春天的诗：",
        "解释什么是人工智能：",
        "推荐一本好书：",
        "描述你的理想假期：",
        "给我一个健康饮食建议：",
    ] * 10  # 重复创建50个样本
    
    return Dataset.from_dict({"query": prompts})

def simple_reward_function(texts):
    """
    简单的奖励函数示例
    根据文本长度和积极词汇给出奖励
    """
    rewards = []
    positive_words = ["好", "棒", "优秀", "美丽", "快乐", "健康", "推荐", "excellent", "good", "great"]
    
    for text in texts:
        reward = 0.0
        
        # 长度奖励 (适中长度)
        length = len(text)
        if 20 <= length <= 100:
            reward += 1.0
        elif length > 100:
            reward += 0.5
        
        # 积极词汇奖励
        for word in positive_words:
            if word in text.lower():
                reward += 0.5
        
        rewards.append(reward)
    
    return rewards

def get_optimal_device():
    """获取最优计算设备"""
    if torch.backends.mps.is_available():
        print("🍎 使用苹果神经引擎(MPS)加速")
        return "mps"
    elif torch.cuda.is_available():
        print("🚀 使用CUDA GPU加速")
        return "cuda"
    else:
        print("💻 使用CPU模式")
        return "cpu"

def get_device_optimized_config():
    """根据设备获取优化配置"""
    device = get_optimal_device()
    
    # 检测内存大小
    import psutil
    memory_gb = psutil.virtual_memory().total / (1024**3)
    
    # 根据设备和内存调整配置
    if device == "mps":
        # 苹果芯片优化配置
        if memory_gb >= 16:
            return {
                "batch_size": 4,
                "mini_batch_size": 2,
                "gradient_accumulation_steps": 2,
                "optimize_cuda_cache": False,  # 关闭CUDA优化
            }
        else:
            return {
                "batch_size": 2,
                "mini_batch_size": 1,
                "gradient_accumulation_steps": 4,
                "optimize_cuda_cache": False,
            }
    elif device == "cuda":
        # GPU配置
        return {
            "batch_size": 8,
            "mini_batch_size": 4,
            "gradient_accumulation_steps": 1,
            "optimize_cuda_cache": True,
        }
    else:
        # CPU配置
        return {
            "batch_size": 2,
            "mini_batch_size": 1,
            "gradient_accumulation_steps": 4,
            "optimize_cuda_cache": False,
        }

def main():
    print("🎯 开始简单PPO训练示例")
    
    # 获取设备优化配置
    device_config = get_device_optimized_config()
    
    # 1. 配置
    config = PPOConfig(
        learning_rate=1.41e-5,
        early_stopping=False,
        target_kl=0.1,
        ppo_epochs=4,
        seed=42,
        steps=20,  # 少量步骤用于演示
        **device_config  # 应用设备优化配置
    )
    
    # 2. 加载模型和tokenizer
    print("📚 加载模型和tokenizer...")
    model = AutoModelForCausalLMWithValueHead.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # 设置pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
    
    # 移动模型到最优设备
    device = get_optimal_device()
    try:
        model = model.to(device)
        print(f"📱 模型已移动到 {device} 设备")
    except Exception as e:
        print(f"⚠️  设备移动失败，使用CPU: {e}")
        device = "cpu"
        model = model.to(device)
    
    # 3. 准备数据
    print("📊 准备训练数据...")
    dataset = create_sample_dataset()
    
    def tokenize_function(examples):
        return tokenizer(examples["query"], 
                        truncation=True, 
                        padding="max_length", 
                        max_length=64,
                        return_tensors="pt")
    
    # 4. 创建PPO训练器
    print("🏋️ 创建PPO训练器...")
    ppo_trainer = PPOTrainer(
        config=config,
        model=model,
        ref_model=None,  # 使用默认参考模型
        tokenizer=tokenizer,
        dataset=dataset,
        data_collator=None,
    )
    
    print("🎮 开始训练...")
    
    # 5. 训练循环
    for epoch, batch in enumerate(ppo_trainer.dataloader):
        if epoch >= config.steps:
            break
            
        print(f"\n--- Epoch {epoch + 1} ---")
        
        # 生成回复
        query_tensors = batch["input_ids"]
        response_tensors = ppo_trainer.generate(
            query_tensors,
            return_prompt=False,
            max_length=128,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
        )
        
        # 解码文本用于计算奖励
        batch_texts = []
        for i in range(len(query_tensors)):
            query_text = tokenizer.decode(query_tensors[i], skip_special_tokens=True)
            response_text = tokenizer.decode(response_tensors[i], skip_special_tokens=True)
            full_text = query_text + response_text
            batch_texts.append(response_text)
            print(f"Query: {query_text}")
            print(f"Response: {response_text}")
        
        # 计算奖励
        rewards = simple_reward_function(batch_texts)
        rewards = [torch.tensor(r) for r in rewards]
        
        print(f"Rewards: {[r.item() for r in rewards]}")
        
        # PPO更新
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        
        # 打印统计信息
        if stats:
            print(f"PPO Loss: {stats.get('ppo/loss/policy', 'N/A')}")
            print(f"Mean Reward: {stats.get('ppo/mean_scores', 'N/A')}")
    
    print("\n🎉 训练完成！")
    
    # 6. 保存模型
    model.save_pretrained("./models/ppo_gpt2_simple")
    tokenizer.save_pretrained("./models/ppo_gpt2_simple")
    print("💾 模型已保存到 ./models/ppo_gpt2_simple")

if __name__ == "__main__":
    main()