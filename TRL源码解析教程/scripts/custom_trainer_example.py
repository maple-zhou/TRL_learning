#!/usr/bin/env python3
"""
自定义训练器实战示例
基于TRL框架开发一个多目标优化的自定义训练器
"""

import torch
import torch.nn as nn
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from transformers import TrainingArguments, AutoTokenizer, AutoModelForCausalLM
from trl import PPOConfig, PPOTrainer
from datasets import Dataset
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

@dataclass
class CustomRLConfig(PPOConfig):
    """自定义强化学习配置类"""
    
    # 多目标优化参数
    fluency_weight: float = field(default=0.4, metadata={"help": "流畅性权重"})
    creativity_weight: float = field(default=0.3, metadata={"help": "创造性权重"})
    relevance_weight: float = field(default=0.3, metadata={"help": "相关性权重"})
    
    # 动态调整参数
    enable_dynamic_weights: bool = field(default=True, metadata={"help": "启用动态权重调整"})
    weight_adaptation_rate: float = field(default=0.01, metadata={"help": "权重适应速率"})
    
    # 高级监控参数
    enable_rich_logging: bool = field(default=True, metadata={"help": "启用丰富日志"})
    diversity_penalty: float = field(default=0.1, metadata={"help": "多样性惩罚系数"})
    
    # 自定义采样参数
    custom_temperature_schedule: bool = field(default=True, metadata={"help": "自定义温度调度"})
    initial_temperature: float = field(default=1.0, metadata={"help": "初始温度"})
    final_temperature: float = field(default=0.1, metadata={"help": "最终温度"})

class MultiObjectiveReward:
    """多目标奖励函数"""
    
    def __init__(self, config: CustomRLConfig):
        self.config = config
        self.weights = {
            'fluency': config.fluency_weight,
            'creativity': config.creativity_weight,
            'relevance': config.relevance_weight
        }
        self.history = defaultdict(list)
    
    def compute_fluency_score(self, texts: List[str]) -> List[float]:
        """计算流畅性得分"""
        scores = []
        for text in texts:
            score = 0.5  # 基础分
            
            # 检查语言流畅性指标
            sentences = text.split('。')
            if len(sentences) > 1:
                # 有完整句子结构
                score += 0.2
            
            # 检查连接词使用
            connectors = ['因为', '所以', '但是', '然而', '并且', '而且']
            connector_count = sum(1 for word in connectors if word in text)
            score += min(0.3, connector_count * 0.1)
            
            # 避免重复
            words = text.split()
            if len(words) > 0:
                unique_ratio = len(set(words)) / len(words)
                score += unique_ratio * 0.3
            
            scores.append(max(0, min(1, score)))
        
        return scores
    
    def compute_creativity_score(self, texts: List[str]) -> List[float]:
        """计算创造性得分"""
        scores = []
        creative_indicators = [
            '想象', '创新', '独特', '新颖', '有趣', '奇妙',
            'creative', 'innovative', 'unique', 'novel'
        ]
        
        for text in texts:
            score = 0.3  # 基础创造性
            text_lower = text.lower()
            
            # 创造性词汇
            for indicator in creative_indicators:
                if indicator in text_lower:
                    score += 0.2
            
            # 词汇多样性
            words = text.split()
            if len(words) > 5:
                vocab_diversity = len(set(words)) / len(words)
                score += vocab_diversity * 0.3
            
            # 长度适中奖励
            if 30 <= len(text) <= 150:
                score += 0.2
            
            scores.append(max(0, min(1, score)))
        
        return scores
    
    def compute_relevance_score(self, prompts: List[str], texts: List[str]) -> List[float]:
        """计算相关性得分"""
        scores = []
        
        for prompt, text in zip(prompts, texts):
            score = 0.5
            
            # 简单的关键词匹配
            prompt_words = set(prompt.lower().split())
            text_words = set(text.lower().split())
            
            # 计算交集比例
            if prompt_words:
                overlap = len(prompt_words & text_words) / len(prompt_words)
                score += overlap * 0.5
            
            scores.append(max(0, min(1, score)))
        
        return scores
    
    def compute_composite_reward(self, prompts: List[str], texts: List[str]) -> List[float]:
        """计算综合奖励"""
        fluency_scores = self.compute_fluency_score(texts)
        creativity_scores = self.compute_creativity_score(texts)
        relevance_scores = self.compute_relevance_score(prompts, texts)
        
        # 记录历史用于动态调整
        self.history['fluency'].extend(fluency_scores)
        self.history['creativity'].extend(creativity_scores)
        self.history['relevance'].extend(relevance_scores)
        
        # 加权平均
        composite_scores = []
        for f, c, r in zip(fluency_scores, creativity_scores, relevance_scores):
            score = (
                f * self.weights['fluency'] +
                c * self.weights['creativity'] +
                r * self.weights['relevance']
            )
            composite_scores.append(score)
        
        return composite_scores
    
    def adapt_weights(self):
        """动态调整权重"""
        if not self.config.enable_dynamic_weights:
            return
        
        if len(self.history['fluency']) < 50:  # 需要足够样本
            return
        
        # 计算各维度的最近表现
        recent_fluency = np.mean(self.history['fluency'][-20:])
        recent_creativity = np.mean(self.history['creativity'][-20:])
        recent_relevance = np.mean(self.history['relevance'][-20:])
        
        # 如果某个维度表现不好，增加其权重
        adaptation_rate = self.config.weight_adaptation_rate
        
        if recent_fluency < 0.6:
            self.weights['fluency'] += adaptation_rate
        if recent_creativity < 0.6:
            self.weights['creativity'] += adaptation_rate
        if recent_relevance < 0.6:
            self.weights['relevance'] += adaptation_rate
        
        # 重新归一化权重
        total_weight = sum(self.weights.values())
        for key in self.weights:
            self.weights[key] /= total_weight
        
        print(f"🔄 权重调整: {self.weights}")

class CustomRLTrainer(PPOTrainer):
    """自定义强化学习训练器"""
    
    def __init__(self, config: CustomRLConfig, **kwargs):
        # 使用自定义配置
        super().__init__(config, **kwargs)
        
        # 初始化自定义组件
        self.multi_objective_reward = MultiObjectiveReward(config)
        self.custom_config = config
        self.training_stats = defaultdict(list)
        self.step_count = 0
    
    def get_temperature_for_step(self, step: int) -> float:
        """自定义温度调度"""
        if not self.custom_config.custom_temperature_schedule:
            return 0.7  # 默认温度
        
        # 线性衰减温度调度
        total_steps = self.custom_config.num_train_epochs * len(self.dataloader)
        progress = min(1.0, step / total_steps)
        
        initial_temp = self.custom_config.initial_temperature
        final_temp = self.custom_config.final_temperature
        
        current_temp = initial_temp * (1 - progress) + final_temp * progress
        return current_temp
    
    def custom_generate(self, query_tensors, **generation_kwargs):
        """自定义生成函数，支持动态温度"""
        
        current_temp = self.get_temperature_for_step(self.step_count)
        generation_kwargs['temperature'] = current_temp
        
        # 调用父类生成方法
        response_tensors = super().generate(
            query_tensors,
            **generation_kwargs
        )
        
        return response_tensors
    
    def compute_custom_rewards(self, query_tensors, response_tensors):
        """计算自定义奖励"""
        
        # 解码文本
        prompts = [self.tokenizer.decode(q, skip_special_tokens=True) for q in query_tensors]
        responses = [self.tokenizer.decode(r, skip_special_tokens=True) for r in response_tensors]
        
        # 计算多目标奖励
        rewards = self.multi_objective_reward.compute_composite_reward(prompts, responses)
        
        # 动态调整权重
        if self.step_count % 10 == 0:  # 每10步调整一次
            self.multi_objective_reward.adapt_weights()
        
        return [torch.tensor(r) for r in rewards]
    
    def custom_step(self, queries, responses):
        """自定义训练步骤"""
        
        # 计算自定义奖励
        rewards = self.compute_custom_rewards(queries, responses)
        
        # 执行PPO更新
        stats = super().step(queries, responses, rewards)
        
        # 记录自定义统计信息
        if stats:
            self.training_stats['policy_loss'].append(stats.get('ppo/loss/policy', 0))
            self.training_stats['value_loss'].append(stats.get('ppo/loss/value', 0))
            self.training_stats['mean_reward'].append(np.mean([r.item() for r in rewards]))
            self.training_stats['current_weights'].append(self.multi_objective_reward.weights.copy())
        
        self.step_count += 1
        
        # 每20步打印统计信息
        if self.step_count % 20 == 0:
            self.print_training_stats()
        
        return stats
    
    def print_training_stats(self):
        """打印训练统计信息"""
        if not self.training_stats['mean_reward']:
            return
        
        recent_reward = np.mean(self.training_stats['mean_reward'][-10:])
        recent_policy_loss = np.mean(self.training_stats['policy_loss'][-10:])
        
        print(f"\n📊 训练统计 (步骤 {self.step_count}):")
        print(f"   平均奖励: {recent_reward:.3f}")
        print(f"   策略损失: {recent_policy_loss:.3f}")
        print(f"   当前权重: {self.multi_objective_reward.weights}")
    
    def visualize_training_progress(self):
        """可视化训练进度"""
        if not self.training_stats['mean_reward']:
            print("❌ 没有训练数据可视化")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # 奖励曲线
        axes[0, 0].plot(self.training_stats['mean_reward'])
        axes[0, 0].set_title('平均奖励变化')
        axes[0, 0].set_xlabel('训练步骤')
        axes[0, 0].set_ylabel('奖励')
        axes[0, 0].grid(True)
        
        # 损失曲线
        axes[0, 1].plot(self.training_stats['policy_loss'], label='策略损失')
        axes[0, 1].plot(self.training_stats['value_loss'], label='价值损失')
        axes[0, 1].set_title('损失函数变化')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 权重变化
        if self.training_stats['current_weights']:
            weights_history = self.training_stats['current_weights']
            fluency_weights = [w['fluency'] for w in weights_history]
            creativity_weights = [w['creativity'] for w in weights_history]
            relevance_weights = [w['relevance'] for w in weights_history]
            
            axes[1, 0].plot(fluency_weights, label='流畅性')
            axes[1, 0].plot(creativity_weights, label='创造性')
            axes[1, 0].plot(relevance_weights, label='相关性')
            axes[1, 0].set_title('权重动态调整')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # 奖励分布
        if len(self.training_stats['mean_reward']) > 20:
            axes[1, 1].hist(self.training_stats['mean_reward'][-50:], bins=20, alpha=0.7)
            axes[1, 1].set_title('最近奖励分布')
            axes[1, 1].set_xlabel('奖励值')
            axes[1, 1].set_ylabel('频次')
        
        plt.tight_layout()
        plt.savefig('custom_training_progress.png', dpi=150)
        plt.show()
        
        print("📈 训练可视化图表已保存到 custom_training_progress.png")

class AdaptiveKLController:
    """自适应KL散度控制器"""
    
    def __init__(self, initial_kl_coef=0.05, target_kl=0.1):
        self.kl_coef = initial_kl_coef
        self.target_kl = target_kl
        self.kl_history = []
    
    def update(self, current_kl: float):
        """根据当前KL散度调整系数"""
        self.kl_history.append(current_kl)
        
        # 保持最近100个KL值
        if len(self.kl_history) > 100:
            self.kl_history = self.kl_history[-100:]
        
        # 计算最近的平均KL
        recent_kl = np.mean(self.kl_history[-10:])
        
        # 自适应调整KL系数
        if recent_kl > self.target_kl * 1.5:
            # KL太大，增加惩罚
            self.kl_coef *= 1.1
        elif recent_kl < self.target_kl * 0.5:
            # KL太小，减少惩罚
            self.kl_coef *= 0.95
        
        # 限制KL系数范围
        self.kl_coef = max(0.001, min(1.0, self.kl_coef))
        
        return self.kl_coef

def create_demo_dataset():
    """创建演示数据集"""
    prompts = [
        "请写一个创意故事：",
        "解释一个复杂概念：",
        "给出实用建议：",
        "描述一个场景：",
        "分析一个问题：",
    ] * 20
    
    return Dataset.from_dict({"query": prompts})

def demo_custom_trainer():
    """演示自定义训练器"""
    
    print("🎯 自定义训练器演示")
    print("=" * 50)
    
    # 1. 创建配置
    config = CustomRLConfig(
        learning_rate=1e-5,
        batch_size=4,
        mini_batch_size=2,
        steps=30,
        fluency_weight=0.4,
        creativity_weight=0.3,
        relevance_weight=0.3,
        enable_dynamic_weights=True,
    )
    
    print(f"初始权重配置:")
    print(f"  流畅性: {config.fluency_weight}")
    print(f"  创造性: {config.creativity_weight}")
    print(f"  相关性: {config.relevance_weight}")
    
    # 2. 准备模型和数据
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    from trl import AutoModelForCausalLMWithValueHead
    model = AutoModelForCausalLMWithValueHead.from_pretrained("gpt2")
    
    dataset = create_demo_dataset()
    
    # 3. 创建自定义训练器
    trainer = CustomRLTrainer(
        config=config,
        model=model,
        ref_model=None,
        tokenizer=tokenizer,
        dataset=dataset
    )
    
    print(f"\n🚀 开始自定义训练 (共{config.steps}步)...")
    
    # 4. 模拟训练循环
    for epoch, batch in enumerate(trainer.dataloader):
        if epoch >= config.steps:
            break
        
        print(f"\n--- 步骤 {epoch + 1} ---")
        
        # 生成响应
        query_tensors = batch["input_ids"]
        current_temp = trainer.get_temperature_for_step(epoch)
        
        response_tensors = trainer.custom_generate(
            query_tensors,
            return_prompt=False,
            max_length=100,
            do_sample=True,
            temperature=current_temp,
            top_p=0.9
        )
        
        # 显示第一个样本
        if epoch < 3:  # 只显示前3个样本
            query_text = tokenizer.decode(query_tensors[0], skip_special_tokens=True)
            response_text = tokenizer.decode(response_tensors[0], skip_special_tokens=True)
            print(f"提示: {query_text}")
            print(f"回复: {response_text}")
            print(f"温度: {current_temp:.2f}")
        
        # 自定义训练步骤
        stats = trainer.custom_step(query_tensors, response_tensors)
    
    print("\n🎉 训练完成！")
    
    # 5. 可视化结果
    trainer.visualize_training_progress()
    
    # 6. 保存模型
    save_choice = input("\n是否保存训练后的模型? (y/n): ")
    if save_choice.lower() == 'y':
        model.save_pretrained("./models/custom_rl_model")
        tokenizer.save_pretrained("./models/custom_rl_model")
        print("💾 模型已保存到 ./models/custom_rl_model")

def analyze_custom_implementation():
    """分析自定义实现的设计模式"""
    
    print("🔍 自定义训练器设计模式分析")
    print("=" * 50)
    
    patterns = [
        {
            "name": "策略模式",
            "description": "MultiObjectiveReward支持不同的奖励计算策略",
            "example": "可以轻松替换为其他奖励函数"
        },
        {
            "name": "观察者模式", 
            "description": "AdaptiveKLController监控训练状态并自动调整",
            "example": "KL散度超标时自动调整惩罚系数"
        },
        {
            "name": "模板方法模式",
            "description": "CustomRLTrainer继承PPOTrainer，重写关键方法",
            "example": "custom_step()方法扩展了标准PPO流程"
        },
        {
            "name": "配置模式",
            "description": "CustomRLConfig扩展了标准配置",
            "example": "添加多目标优化的专用参数"
        }
    ]
    
    for pattern in patterns:
        print(f"\n🎨 {pattern['name']}:")
        print(f"   描述: {pattern['description']}")
        print(f"   示例: {pattern['example']}")

def main():
    print("🎯 TRL自定义训练器开发实战")
    
    while True:
        print("\n选择操作:")
        print("1. 运行自定义训练器演示")
        print("2. 分析设计模式")
        print("3. 查看训练可视化")
        print("4. 退出")
        
        choice = input("请选择 (1-4): ")
        
        if choice == "1":
            demo_custom_trainer()
        elif choice == "2":
            analyze_custom_implementation()
        elif choice == "3":
            # 如果有保存的图片，显示路径
            import os
            if os.path.exists("custom_training_progress.png"):
                print("📈 训练图表: custom_training_progress.png")
            else:
                print("❌ 请先运行训练演示")
        elif choice == "4":
            print("👋 退出")
            break
        else:
            print("❌ 无效选择")

if __name__ == "__main__":
    main()