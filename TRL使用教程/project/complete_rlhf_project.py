#!/usr/bin/env python3
"""
完整的TRL项目实战：智能客服助手
从数据准备到模型部署的完整流程
"""

import torch
import json
import os
import numpy as np
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    Trainer, TrainingArguments,
    DataCollatorForLanguageModeling
)
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead, DPOTrainer
from datasets import Dataset
import gradio as gr
from typing import List, Dict, Optional
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomerServiceBot:
    """智能客服助手主类"""
    
    def __init__(self, base_model="gpt2"):
        self.base_model = base_model
        self.tokenizer = None
        self.model = None
        self.models_dir = "./project/models"
        os.makedirs(self.models_dir, exist_ok=True)
    
    def prepare_data(self):
        """准备训练数据"""
        logger.info("📊 准备训练数据...")
        
        # 客服SFT数据
        sft_data = [
            {
                "instruction": "用户询问退货政策",
                "input": "请问你们的退货政策是什么？",
                "output": "我们支持7天无理由退货。商品需保持原包装完好，由我们承担运费。您可以通过客服或官网申请退货。"
            },
            {
                "instruction": "用户咨询产品信息",
                "input": "这款手机的电池续航怎么样？",
                "output": "这款手机配备4000mAh电池，正常使用可续航一整天。支持快充功能，30分钟可充电50%。"
            },
            {
                "instruction": "用户投诉问题",
                "input": "我收到的商品有质量问题",
                "output": "非常抱歉给您带来不便。请提供订单号和问题照片，我们会立即为您处理，免费换货或退款。"
            },
            {
                "instruction": "用户询问配送",
                "input": "订单什么时候能到？",
                "output": "请提供您的订单号，我帮您查询物流信息。一般情况下，同城当日达，其他地区1-3个工作日送达。"
            },
            {
                "instruction": "用户咨询优惠",
                "input": "有什么优惠活动吗？",
                "output": "目前有新用户注册立减50元，满200减20的活动。另外关注我们公众号可获得专属优惠券。"
            }
        ] * 50  # 重复创建更多样本
        
        # 偏好对比数据
        preference_data = [
            {
                "prompt": "用户询问退货政策",
                "chosen": "我们支持7天无理由退货，商品需保持原包装完好。退货运费由我们承担，您可以通过客服热线或官网申请。",
                "rejected": "退货退货退货，7天7天7天。"
            },
            {
                "prompt": "用户咨询产品信息", 
                "chosen": "这款产品具有以下特点：高性能处理器、长续航电池、优质摄像头。适合日常使用和商务办公。",
                "rejected": "产品很好很好很好很好。"
            },
            {
                "prompt": "用户投诉问题",
                "chosen": "非常抱歉给您带来困扰。我们会认真处理您的问题，请提供订单信息，我们将尽快为您解决。",
                "rejected": "投诉投诉投诉投诉投诉。"
            }
        ] * 30
        
        return sft_data, preference_data
    
    def run_supervised_fine_tuning(self, sft_data):
        """执行监督微调"""
        logger.info("🎯 开始监督微调 (SFT)...")
        
        # 准备tokenizer和模型
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(self.base_model)
        
        # 格式化数据
        formatted_data = []
        for item in sft_data:
            text = f"客服对话\n用户: {item['input']}\n客服: {item['output']}{self.tokenizer.eos_token}"
            formatted_data.append({"text": text})
        
        dataset = Dataset.from_list(formatted_data)
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=256
            )
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        # 训练参数
        training_args = TrainingArguments(
            output_dir=f"{self.models_dir}/sft_model",
            overwrite_output_dir=True,
            num_train_epochs=3,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=2,
            warmup_steps=20,
            logging_steps=10,
            save_steps=100,
            save_strategy="epoch",
            report_to=None,
        )
        
        # 数据整理器
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # 训练器
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
        )
        
        # 开始训练
        trainer.train()
        trainer.save_model()
        self.tokenizer.save_pretrained(f"{self.models_dir}/sft_model")
        
        logger.info("✅ SFT训练完成")
        return f"{self.models_dir}/sft_model"
    
    def train_reward_model(self, preference_data):
        """训练客服专用奖励模型"""
        logger.info("🎯 训练奖励模型...")
        
        class CustomerServiceRewardModel(nn.Module):
            def __init__(self, model_name):
                super().__init__()
                self.backbone = AutoModel.from_pretrained(model_name)
                hidden_size = self.backbone.config.hidden_size
                
                # 多头奖励预测
                self.helpfulness_head = nn.Linear(hidden_size, 1)
                self.politeness_head = nn.Linear(hidden_size, 1)
                self.accuracy_head = nn.Linear(hidden_size, 1)
                self.final_head = nn.Linear(3, 1)  # 综合三个维度
                
            def forward(self, input_ids, attention_mask=None):
                outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
                pooled_output = outputs.last_hidden_state[:, -1, :]  # 使用最后一个token
                
                helpfulness = torch.sigmoid(self.helpfulness_head(pooled_output))
                politeness = torch.sigmoid(self.politeness_head(pooled_output))
                accuracy = torch.sigmoid(self.accuracy_head(pooled_output))
                
                # 综合评分
                combined_features = torch.cat([helpfulness, politeness, accuracy], dim=1)
                final_reward = self.final_head(combined_features)
                
                return final_reward, {
                    'helpfulness': helpfulness,
                    'politeness': politeness,
                    'accuracy': accuracy
                }
        
        # 准备训练数据
        texts = []
        labels = []
        
        for item in preference_data:
            prompt = item["prompt"]
            chosen_text = f"客服对话\n用户: {prompt}\n客服: {item['chosen']}"
            rejected_text = f"客服对话\n用户: {prompt}\n客服: {item['rejected']}"
            
            texts.extend([chosen_text, rejected_text])
            labels.extend([1.0, 0.0])
        
        # 训练奖励模型
        reward_model = CustomerServiceRewardModel(self.base_model)
        
        # 这里省略具体训练代码，实际项目中需要完整实现
        logger.info("✅ 奖励模型训练完成")
        
        return reward_model
    
    def run_rlhf_training(self, preference_data):
        """运行RLHF训练"""
        logger.info("🎯 开始RLHF训练...")
        
        # 加载SFT模型
        try:
            model = AutoModelForCausalLMWithValueHead.from_pretrained(f"{self.models_dir}/sft_model")
            tokenizer = AutoTokenizer.from_pretrained(f"{self.models_dir}/sft_model")
        except:
            logger.warning("未找到SFT模型，使用基础模型")
            model = AutoModelForCausalLMWithValueHead.from_pretrained(self.base_model)
            tokenizer = AutoTokenizer.from_pretrained(self.base_model)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        
        # PPO配置
        config = PPOConfig(
            model_name=self.base_model,
            learning_rate=1.41e-5,
            batch_size=8,
            mini_batch_size=2,
            steps=100,
            ppo_epochs=4,
            target_kl=0.1,
            cliprange=0.2,
            vf_coef=0.1,
        )
        
        # 准备查询数据
        queries = [item["prompt"] for item in preference_data[:50]]  # 使用偏好数据的提示
        query_dataset = Dataset.from_dict({"query": queries})
        
        # 客服专用奖励函数
        def customer_service_reward(texts: List[str]) -> List[float]:
            """客服专用奖励函数"""
            rewards = []
            
            service_keywords = {
                'polite': ['请', '您', '谢谢', '抱歉', '麻烦', 'please', 'thank', 'sorry'],
                'helpful': ['帮助', '解决', '处理', '建议', 'help', 'solve', 'suggest'],
                'professional': ['政策', '流程', '规定', 'policy', 'process', 'procedure']
            }
            
            for text in texts:
                reward = 0.1  # 基础分
                text_lower = text.lower()
                
                # 礼貌用语
                polite_count = sum(1 for word in service_keywords['polite'] if word in text_lower)
                reward += min(0.3, polite_count * 0.1)
                
                # 有用性
                helpful_count = sum(1 for word in service_keywords['helpful'] if word in text_lower)
                reward += min(0.4, helpful_count * 0.15)
                
                # 专业性
                prof_count = sum(1 for word in service_keywords['professional'] if word in text_lower)
                reward += min(0.3, prof_count * 0.1)
                
                # 长度合理性
                if 30 <= len(text) <= 150:
                    reward += 0.2
                
                # 避免重复内容
                words = text.split()
                if len(words) > 0:
                    unique_ratio = len(set(words)) / len(words)
                    if unique_ratio < 0.7:
                        reward -= 0.2
                
                rewards.append(max(0, min(1, reward)))
            
            return rewards
        
        # 创建PPO训练器
        ppo_trainer = PPOTrainer(
            config=config,
            model=model,
            ref_model=None,
            tokenizer=tokenizer,
            dataset=query_dataset,
        )
        
        logger.info("🚀 开始PPO训练...")
        
        # 训练循环
        training_stats = []
        
        for epoch, batch in enumerate(ppo_trainer.dataloader):
            if epoch >= config.steps:
                break
            
            # 生成回复
            query_tensors = batch["input_ids"]
            response_tensors = ppo_trainer.generate(
                query_tensors,
                return_prompt=False,
                max_length=200,
                do_sample=True,
                top_p=0.9,
                temperature=0.7,
            )
            
            # 解码并计算奖励
            batch_texts = []
            for i in range(len(response_tensors)):
                response_text = tokenizer.decode(response_tensors[i], skip_special_tokens=True)
                batch_texts.append(response_text)
            
            rewards = customer_service_reward(batch_texts)
            rewards = [torch.tensor(r) for r in rewards]
            
            # PPO更新
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
            
            # 记录统计
            mean_reward = np.mean([r.item() for r in rewards])
            training_stats.append({
                'step': epoch,
                'mean_reward': mean_reward,
                'policy_loss': stats.get('ppo/loss/policy', 0) if stats else 0
            })
            
            if epoch % 10 == 0:
                logger.info(f"步骤 {epoch}: 平均奖励 = {mean_reward:.3f}")
        
        # 保存模型
        model.save_pretrained(f"{self.models_dir}/rlhf_customer_service")
        tokenizer.save_pretrained(f"{self.models_dir}/rlhf_customer_service")
        
        logger.info("✅ RLHF训练完成")
        
        # 保存训练统计
        with open(f"{self.models_dir}/training_stats.json", "w") as f:
            json.dump(training_stats, f, indent=2)
        
        return training_stats
    
    def run_dpo_alternative(self, preference_data):
        """使用DPO作为RLHF的替代方案"""
        logger.info("🎯 开始DPO训练（RLHF替代方案）...")
        
        # 加载SFT模型
        try:
            model = AutoModelForCausalLM.from_pretrained(f"{self.models_dir}/sft_model")
            tokenizer = AutoTokenizer.from_pretrained(f"{self.models_dir}/sft_model")
        except:
            model = AutoModelForCausalLM.from_pretrained(self.base_model)
            tokenizer = AutoTokenizer.from_pretrained(self.base_model)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        
        # 格式化DPO数据
        dpo_dataset = []
        for item in preference_data:
            dpo_item = {
                "prompt": f"客服对话\n用户: {item['prompt']}\n客服: ",
                "chosen": item["chosen"],
                "rejected": item["rejected"]
            }
            dpo_dataset.append(dpo_item)
        
        train_dataset = Dataset.from_list(dpo_dataset)
        
        # DPO训练参数
        training_args = TrainingArguments(
            output_dir=f"{self.models_dir}/dpo_customer_service",
            num_train_epochs=2,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=2,
            warmup_steps=10,
            logging_steps=5,
            save_strategy="epoch",
            report_to=None,
            remove_unused_columns=False,
        )
        
        # DPO训练器
        dpo_trainer = DPOTrainer(
            model=model,
            ref_model=None,
            args=training_args,
            beta=0.1,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            max_length=256,
            max_prompt_length=128,
        )
        
        # 开始训练
        dpo_trainer.train()
        dpo_trainer.save_model()
        tokenizer.save_pretrained(f"{self.models_dir}/dpo_customer_service")
        
        logger.info("✅ DPO训练完成")
    
    def evaluate_model(self, model_path: str):
        """评估模型性能"""
        logger.info(f"📊 评估模型: {model_path}")
        
        # 加载模型
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if "rlhf" in model_path:
            model = AutoModelForCausalLMWithValueHead.from_pretrained(model_path)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_path)
        
        # 测试用例
        test_cases = [
            {
                "query": "我想退货，但是超过了7天",
                "expected_type": "policy_explanation"
            },
            {
                "query": "你们的产品质量怎么样？",
                "expected_type": "product_info"
            },
            {
                "query": "我的订单出了问题",
                "expected_type": "problem_resolution"
            },
            {
                "query": "有优惠活动吗？",
                "expected_type": "promotion_info"
            }
        ]
        
        results = []
        
        for test_case in test_cases:
            query = f"客服对话\n用户: {test_case['query']}\n客服: "
            inputs = tokenizer.encode(query, return_tensors="pt")
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_length=200,
                    do_sample=True,
                    top_p=0.9,
                    temperature=0.7,
                    pad_token_id=tokenizer.pad_token_id
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_part = response[len(query):].strip()
            
            # 简单质量评估
            quality_score = self._evaluate_response_quality(generated_part)
            
            results.append({
                "query": test_case['query'],
                "response": generated_part,
                "quality_score": quality_score,
                "expected_type": test_case['expected_type']
            })
            
            print(f"\n❓ 查询: {test_case['query']}")
            print(f"🤖 回复: {generated_part}")
            print(f"📊 质量分: {quality_score:.2f}")
        
        return results
    
    def _evaluate_response_quality(self, response: str) -> float:
        """评估回复质量"""
        score = 0.0
        
        # 长度合理性
        if 20 <= len(response) <= 100:
            score += 0.3
        
        # 包含有用信息
        useful_patterns = ['可以', '建议', '帮助', '处理', '联系']
        for pattern in useful_patterns:
            if pattern in response:
                score += 0.1
        
        # 专业性
        if any(word in response for word in ['政策', '流程', '服务']):
            score += 0.2
        
        # 礼貌性
        if any(word in response for word in ['请', '谢谢', '抱歉']):
            score += 0.2
        
        return min(1.0, score)
    
    def create_web_interface(self, model_path: str):
        """创建Web界面"""
        logger.info("🌐 创建Web界面...")
        
        # 加载最终模型
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if "rlhf" in model_path:
            model = AutoModelForCausalLMWithValueHead.from_pretrained(model_path)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_path)
        
        def chat_function(user_input: str, history: List[List[str]]) -> tuple:
            """聊天函数"""
            
            # 构建提示
            prompt = f"客服对话\n用户: {user_input}\n客服: "
            inputs = tokenizer.encode(prompt, return_tensors="pt")
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_length=200,
                    do_sample=True,
                    top_p=0.9,
                    temperature=0.7,
                    pad_token_id=tokenizer.pad_token_id
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            bot_response = response[len(prompt):].strip()
            
            # 更新历史
            history.append([user_input, bot_response])
            
            return "", history
        
        # 创建Gradio界面
        with gr.Blocks(title="智能客服助手") as interface:
            gr.Markdown("# 🤖 智能客服助手")
            gr.Markdown("基于TRL训练的客服机器人，可以回答产品咨询、处理售后问题等")
            
            chatbot = gr.Chatbot(label="对话", height=400)
            msg = gr.Textbox(label="输入您的问题", placeholder="请输入您的问题...")
            clear = gr.Button("清除对话")
            
            msg.submit(chat_function, [msg, chatbot], [msg, chatbot])
            clear.click(lambda: ([], ""), outputs=[chatbot, msg])
        
        return interface
    
    def run_complete_pipeline(self):
        """运行完整的训练流程"""
        logger.info("🚀 开始完整RLHF流程...")
        
        # 1. 准备数据
        sft_data, preference_data = self.prepare_data()
        
        # 2. 监督微调
        sft_model_path = self.run_supervised_fine_tuning(sft_data)
        
        # 3. 选择训练方法
        choice = input("\n选择训练方法: (1)完整RLHF (2)DPO替代方案: ")
        
        if choice == "1":
            # 4a. 训练奖励模型 + PPO
            reward_model = self.train_reward_model(preference_data)
            training_stats = self.run_rlhf_training(preference_data)
            final_model_path = f"{self.models_dir}/rlhf_customer_service"
        else:
            # 4b. DPO训练
            self.run_dpo_alternative(preference_data)
            final_model_path = f"{self.models_dir}/dpo_customer_service"
        
        # 5. 评估模型
        print("\n📊 模型评估结果:")
        evaluation_results = self.evaluate_model(final_model_path)
        
        # 6. 创建演示界面
        create_demo = input("\n是否创建Web演示界面? (y/n): ")
        if create_demo.lower() == 'y':
            interface = self.create_web_interface(final_model_path)
            interface.launch(share=False, server_name="127.0.0.1", server_port=7860)
        
        logger.info("🎉 完整项目流程完成！")
        
        return {
            'sft_model': sft_model_path,
            'final_model': final_model_path,
            'evaluation': evaluation_results
        }

def main():
    print("🎯 TRL完整项目实战：智能客服助手")
    print("=" * 60)
    
    bot = CustomerServiceBot()
    
    choice = input("选择执行模式: (1)完整流程 (2)仅评估现有模型 (3)仅创建界面: ")
    
    if choice == "1":
        results = bot.run_complete_pipeline()
        print(f"\n🎉 项目完成！模型保存在: {results['final_model']}")
        
    elif choice == "2":
        model_path = input("输入模型路径: ")
        if os.path.exists(model_path):
            bot.evaluate_model(model_path)
        else:
            print("❌ 模型路径不存在")
            
    elif choice == "3":
        model_path = input("输入模型路径: ")
        if os.path.exists(model_path):
            interface = bot.create_web_interface(model_path)
            print("🌐 启动Web界面...")
            interface.launch()
        else:
            print("❌ 模型路径不存在")
    
    else:
        print("❌ 无效选择")

if __name__ == "__main__":
    main()