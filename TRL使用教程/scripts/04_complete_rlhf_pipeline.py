#!/usr/bin/env python3
"""
完整的RLHF训练流程示例
演示从SFT到奖励模型训练再到PPO的完整流程
"""

import torch
import json
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from datasets import Dataset
import os

class RLHFPipeline:
    """完整的RLHF训练流程"""
    
    def __init__(self, base_model="gpt2"):
        self.base_model = base_model
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def create_sft_dataset(self):
        """创建监督微调数据集"""
        
        sft_data = [
            {
                "instruction": "请写一首关于友谊的诗",
                "response": "友谊如春风，温暖人心田。\n真诚相待久，患难见真情。"
            },
            {
                "instruction": "解释什么是机器学习",
                "response": "机器学习是一种人工智能技术，让计算机能够从数据中自动学习和改进，无需明确编程。"
            },
            {
                "instruction": "给我一些学习编程的建议",
                "response": "建议从基础语法开始，多动手实践，参与开源项目，坚持每天编码练习。"
            },
            {
                "instruction": "描述理想的工作环境",
                "response": "理想的工作环境应该有良好的团队氛围、充足的学习机会、合理的工作强度和成长空间。"
            },
            {
                "instruction": "推荐一些健康的生活习惯",
                "response": "建议保持规律作息、均衡饮食、适量运动、充足睡眠和积极的心态。"
            }
        ] * 20  # 重复创建更多样本
        
        # 格式化为对话格式
        formatted_data = []
        for item in sft_data:
            text = f"指令: {item['instruction']}\n回复: {item['response']}{self.tokenizer.eos_token}"
            formatted_data.append({"text": text})
        
        return Dataset.from_list(formatted_data)
    
    def run_sft(self):
        """执行监督微调"""
        
        print("🎯 阶段一: 监督微调 (SFT)")
        
        # 准备数据
        dataset = self.create_sft_dataset()
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=128
            )
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        # 加载模型
        model = AutoModelForCausalLM.from_pretrained(self.base_model)
        model.resize_token_embeddings(len(self.tokenizer))
        
        # 训练参数
        training_args = TrainingArguments(
            output_dir="./models/sft_model",
            overwrite_output_dir=True,
            num_train_epochs=2,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=2,
            warmup_steps=10,
            logging_steps=5,
            save_steps=50,
            save_strategy="epoch",
            report_to=None,
            dataloader_drop_last=True,
        )
        
        # 数据整理器
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # 创建训练器
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
        )
        
        # 开始训练
        print("🚀 开始SFT训练...")
        trainer.train()
        
        # 保存模型
        trainer.save_model()
        self.tokenizer.save_pretrained("./models/sft_model")
        
        print("✅ SFT训练完成")
        return "./models/sft_model"
    
    def create_preference_dataset(self):
        """创建偏好数据集用于奖励模型训练"""
        
        preference_data = [
            {
                "prompt": "请写一首关于友谊的诗",
                "chosen": "友谊如春风，温暖人心田。\n真诚相待久，患难见真情。",
                "rejected": "诗诗诗诗诗诗诗诗诗。"
            },
            {
                "prompt": "解释什么是机器学习",
                "chosen": "机器学习是一种人工智能技术，让计算机能够从数据中自动学习。",
                "rejected": "机器学习就是机器学习机器学习。"
            },
            {
                "prompt": "给我一些学习建议",
                "chosen": "建议制定学习计划，保持持续练习，多思考总结。",
                "rejected": "学学学学学学学学学。"
            },
            {
                "prompt": "描述理想的假期",
                "chosen": "理想的假期是与家人朋友一起，放松身心，探索新地方。",
                "rejected": "假期假期假期假期假期。"
            },
            {
                "prompt": "健康生活的要素是什么",
                "chosen": "健康生活需要规律作息、均衡饮食、适量运动和积极心态。",
                "rejected": "健康健康健康健康健康。"
            }
        ] * 20
        
        return preference_data
    
    def train_reward_model(self):
        """训练奖励模型"""
        
        print("🎯 阶段二: 奖励模型训练")
        
        # 创建偏好数据
        preference_data = self.create_preference_dataset()
        
        # 准备训练数据
        texts = []
        labels = []
        
        for item in preference_data:
            prompt = item["prompt"]
            chosen_text = prompt + " " + item["chosen"]
            rejected_text = prompt + " " + item["rejected"]
            
            texts.extend([chosen_text, rejected_text])
            labels.extend([1.0, 0.0])  # chosen=1, rejected=0
        
        # Tokenize
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt"
        )
        
        dataset = Dataset.from_dict({
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": labels
        })
        
        # 创建奖励模型
        reward_model = RewardModel(self.base_model)
        
        # 训练参数
        training_args = TrainingArguments(
            output_dir="./models/reward_model",
            num_train_epochs=3,
            per_device_train_batch_size=4,
            logging_steps=10,
            save_strategy="epoch",
            report_to=None,
        )
        
        # 自定义训练器
        class RewardTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False):
                labels = inputs.pop("labels")
                outputs = model(**inputs)
                rewards = outputs.squeeze(-1)
                
                loss = nn.MSELoss()(
                    rewards, 
                    torch.tensor(labels, device=rewards.device, dtype=torch.float)
                )
                
                return (loss, outputs) if return_outputs else loss
        
        trainer = RewardTrainer(
            model=reward_model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=self.tokenizer,
        )
        
        print("🚀 开始奖励模型训练...")
        trainer.train()
        trainer.save_model()
        
        print("✅ 奖励模型训练完成")
        return reward_model
    
    def run_ppo_with_reward_model(self, reward_model):
        """使用奖励模型进行PPO训练"""
        
        print("🎯 阶段三: PPO强化学习")
        
        # 加载SFT模型
        try:
            model = AutoModelForCausalLMWithValueHead.from_pretrained("./models/sft_model")
        except:
            print("⚠️  未找到SFT模型，使用基础模型")
            model = AutoModelForCausalLMWithValueHead.from_pretrained(self.base_model)
        
        # PPO配置
        config = PPOConfig(
            model_name=self.base_model,
            learning_rate=1.41e-5,
            batch_size=4,
            mini_batch_size=2,
            steps=20,
            ppo_epochs=4,
            target_kl=0.1,
        )
        
        # 创建查询数据集
        queries = [
            "请写一首关于友谊的诗：",
            "解释什么是机器学习：",
            "给我一些学习建议：",
            "描述理想的假期：",
            "健康生活的要素是什么：",
        ] * 10
        
        query_dataset = Dataset.from_dict({"query": queries})
        
        # 创建PPO训练器
        ppo_trainer = PPOTrainer(
            config=config,
            model=model,
            ref_model=None,
            tokenizer=self.tokenizer,
            dataset=query_dataset,
        )
        
        def compute_rewards_with_model(texts):
            """使用奖励模型计算奖励"""
            rewards = []
            
            for text in texts:
                encoding = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=128
                )
                
                with torch.no_grad():
                    reward = reward_model(**encoding)
                    rewards.append(reward.squeeze().item())
            
            return rewards
        
        print("🚀 开始PPO训练...")
        
        # PPO训练循环
        for epoch, batch in enumerate(ppo_trainer.dataloader):
            if epoch >= config.steps:
                break
            
            print(f"\n--- PPO步骤 {epoch + 1} ---")
            
            # 生成回复
            query_tensors = batch["input_ids"]
            response_tensors = ppo_trainer.generate(
                query_tensors,
                return_prompt=False,
                max_length=100,
                do_sample=True,
                top_p=0.9,
                temperature=0.7,
            )
            
            # 解码文本
            batch_texts = []
            for i in range(len(response_tensors)):
                query_text = self.tokenizer.decode(query_tensors[i], skip_special_tokens=True)
                response_text = self.tokenizer.decode(response_tensors[i], skip_special_tokens=True)
                full_text = query_text + " " + response_text
                batch_texts.append(full_text)
                
                if i == 0:  # 显示第一个样本
                    print(f"查询: {query_text}")
                    print(f"回复: {response_text}")
            
            # 使用奖励模型计算奖励
            rewards = compute_rewards_with_model(batch_texts)
            rewards = [torch.tensor(r) for r in rewards]
            
            print(f"平均奖励: {np.mean([r.item() for r in rewards]):.3f}")
            
            # PPO更新
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
            
            if stats:
                print(f"策略损失: {stats.get('ppo/loss/policy', 'N/A')}")
        
        print("✅ PPO训练完成")
        
        # 保存最终模型
        model.save_pretrained("./models/rlhf_final_model")
        print("💾 最终模型已保存")

def main():
    pipeline = RLHFPipeline()
    
    print("🎯 完整RLHF训练流程")
    print("这个流程包含三个阶段:")
    print("1. 监督微调 (SFT)")
    print("2. 奖励模型训练 (RM)")
    print("3. PPO强化学习 (RL)")
    
    choice = input("\n选择执行模式: (1)完整流程 (2)单独阶段: ")
    
    if choice == "1":
        print("\n🚀 执行完整RLHF流程...")
        
        # 阶段1: SFT
        sft_model_path = pipeline.run_sft()
        
        # 阶段2: 奖励模型
        reward_model = pipeline.train_reward_model()
        
        # 阶段3: PPO
        pipeline.run_ppo_with_reward_model(reward_model)
        
        print("\n🎉 完整RLHF流程执行完成！")
        
    elif choice == "2":
        stage = input("选择阶段: (1)SFT (2)奖励模型 (3)PPO: ")
        
        if stage == "1":
            pipeline.run_sft()
        elif stage == "2":
            reward_model = pipeline.train_reward_model()
        elif stage == "3":
            # 需要先加载奖励模型
            try:
                from scripts.reward_model_training import RewardModel
                reward_model = RewardModel()
                reward_model.load_state_dict(torch.load("./models/reward_model/pytorch_model.bin"))
                pipeline.run_ppo_with_reward_model(reward_model)
            except Exception as e:
                print(f"❌ 加载奖励模型失败: {e}")
                print("请先训练奖励模型")
    else:
        print("无效选择")

if __name__ == "__main__":
    main()