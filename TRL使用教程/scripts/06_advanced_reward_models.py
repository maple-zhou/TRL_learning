#!/usr/bin/env python3
"""
高级奖励模型实现
演示多种自定义奖励模型的实现方法
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import numpy as np
from typing import List, Dict
import re

class MultiDimensionalReward:
    """多维度奖励模型"""
    
    def __init__(self):
        self.weights = {
            'helpfulness': 0.4,    # 有用性
            'harmlessness': 0.3,   # 无害性
            'honesty': 0.3,        # 诚实性
        }
    
    def compute_helpfulness(self, texts: List[str]) -> List[float]:
        """计算有用性得分"""
        scores = []
        helpful_indicators = [
            '具体', '详细', '建议', '方法', '步骤', '例子', 
            'specific', 'detailed', 'recommend', 'suggest'
        ]
        
        for text in texts:
            score = 0.0
            text_lower = text.lower()
            
            # 检查有用指标词
            for indicator in helpful_indicators:
                if indicator in text_lower:
                    score += 0.2
            
            # 长度合理性 (50-200字符比较好)
            length = len(text)
            if 50 <= length <= 200:
                score += 0.5
            elif length < 20:
                score -= 0.3
            
            # 结构化内容 (有数字编号、分点等)
            if re.search(r'\d+[.)]', text) or '：' in text:
                score += 0.3
            
            scores.append(max(0, min(1, score)))  # 限制在0-1范围
        
        return scores
    
    def compute_harmlessness(self, texts: List[str]) -> List[float]:
        """计算无害性得分"""
        scores = []
        harmful_words = ['暴力', '伤害', '危险', '违法', 'violence', 'harm', 'illegal']
        
        for text in texts:
            score = 1.0  # 默认无害
            text_lower = text.lower()
            
            # 检查有害词汇
            for word in harmful_words:
                if word in text_lower:
                    score -= 0.5
            
            # 检查重复内容 (可能是低质量)
            words = text.split()
            if len(words) > 0:
                unique_ratio = len(set(words)) / len(words)
                if unique_ratio < 0.5:  # 重复词太多
                    score -= 0.3
            
            scores.append(max(0, min(1, score)))
        
        return scores
    
    def compute_honesty(self, texts: List[str]) -> List[float]:
        """计算诚实性得分"""
        scores = []
        uncertainty_words = ['可能', '也许', '大概', '不确定', 'maybe', 'possibly', 'uncertain']
        confident_words = ['确实', '肯定', '一定', 'definitely', 'certainly']
        
        for text in texts:
            score = 0.5  # 中性起点
            text_lower = text.lower()
            
            # 适当的不确定性表达是诚实的
            uncertainty_count = sum(1 for word in uncertainty_words if word in text_lower)
            if uncertainty_count > 0:
                score += 0.2
            
            # 过于绝对的表达可能不诚实
            confident_count = sum(1 for word in confident_words if word in text_lower)
            if confident_count > 2:  # 太多绝对词汇
                score -= 0.2
            
            # 承认不知道是诚实的表现
            if any(phrase in text_lower for phrase in ['不知道', '不确定', "don't know", "not sure"]):
                score += 0.3
            
            scores.append(max(0, min(1, score)))
        
        return scores
    
    def __call__(self, texts: List[str]) -> List[float]:
        """计算综合奖励"""
        helpfulness = self.compute_helpfulness(texts)
        harmlessness = self.compute_harmlessness(texts)
        honesty = self.compute_honesty(texts)
        
        combined_rewards = []
        for h1, h2, h3 in zip(helpfulness, harmlessness, honesty):
            reward = (h1 * self.weights['helpfulness'] + 
                     h2 * self.weights['harmlessness'] + 
                     h3 * self.weights['honesty'])
            combined_rewards.append(reward)
        
        return combined_rewards

class SafetyRewardModel:
    """安全性导向的奖励模型"""
    
    def __init__(self):
        # 在实际应用中，这里会加载预训练的安全分类器
        self.toxicity_threshold = 0.1
        self.safety_keywords = {
            'safe': ['帮助', '建议', '学习', '健康', '积极', 'help', 'suggest', 'learn'],
            'unsafe': ['伤害', '攻击', '欺骗', 'harm', 'attack', 'deceive']
        }
    
    def detect_toxicity(self, texts: List[str]) -> List[float]:
        """检测文本毒性"""
        scores = []
        
        for text in texts:
            toxicity_score = 0.0
            text_lower = text.lower()
            
            # 检查不安全关键词
            for word in self.safety_keywords['unsafe']:
                if word in text_lower:
                    toxicity_score += 0.3
            
            # 检查安全关键词
            safe_count = sum(1 for word in self.safety_keywords['safe'] if word in text_lower)
            if safe_count > 0:
                toxicity_score -= 0.2
            
            scores.append(max(0, min(1, toxicity_score)))
        
        return scores
    
    def __call__(self, texts: List[str]) -> List[float]:
        """计算安全性奖励"""
        toxicity_scores = self.detect_toxicity(texts)
        safety_rewards = [1.0 - score for score in toxicity_scores]
        return safety_rewards

class TaskSpecificReward:
    """任务特定的奖励模型"""
    
    def __init__(self, task_type="qa"):
        self.task_type = task_type
        self.task_patterns = {
            'qa': {
                'good_patterns': [r'因为', r'所以', r'首先', r'其次', r'总结'],
                'bad_patterns': [r'不知道', r'不清楚', r'随便']
            },
            'creative': {
                'good_patterns': [r'生动', r'形象', r'创意', r'独特'],
                'bad_patterns': [r'平淡', r'无聊', r'重复']
            },
            'code': {
                'good_patterns': [r'def\s+\w+', r'class\s+\w+', r'#.*', r'import\s+\w+'],
                'bad_patterns': [r'错误', r'bug', r'不能运行']
            }
        }
    
    def __call__(self, texts: List[str]) -> List[float]:
        """根据任务类型计算奖励"""
        if self.task_type not in self.task_patterns:
            return [0.5] * len(texts)  # 默认中性奖励
        
        patterns = self.task_patterns[self.task_type]
        rewards = []
        
        for text in texts:
            reward = 0.5  # 基础奖励
            
            # 检查好的模式
            for pattern in patterns['good_patterns']:
                if re.search(pattern, text):
                    reward += 0.2
            
            # 检查坏的模式
            for pattern in patterns['bad_patterns']:
                if re.search(pattern, text):
                    reward -= 0.2
            
            rewards.append(max(0, min(1, reward)))
        
        return rewards

class AdaptiveRewardScheduler:
    """自适应奖励调度器"""
    
    def __init__(self):
        self.step_count = 0
        self.reward_history = []
        self.adaptive_weights = {
            'quality': 1.0,
            'diversity': 0.5,
            'safety': 1.5
        }
    
    def update_weights(self, current_rewards: List[float]):
        """根据训练进度动态调整权重"""
        self.step_count += 1
        self.reward_history.extend(current_rewards)
        
        # 保持最近1000个奖励的历史
        if len(self.reward_history) > 1000:
            self.reward_history = self.reward_history[-1000:]
        
        if len(self.reward_history) >= 100:
            recent_mean = np.mean(self.reward_history[-100:])
            overall_mean = np.mean(self.reward_history)
            
            # 如果最近表现比整体差，增加质量权重
            if recent_mean < overall_mean * 0.9:
                self.adaptive_weights['quality'] *= 1.1
                self.adaptive_weights['diversity'] *= 0.95
            else:
                # 表现好时，增加多样性权重
                self.adaptive_weights['diversity'] *= 1.05
                self.adaptive_weights['quality'] *= 0.98
    
    def get_current_weights(self) -> Dict[str, float]:
        """获取当前权重"""
        return self.adaptive_weights.copy()

def create_ensemble_reward_model():
    """创建集成奖励模型"""
    
    multi_dim_reward = MultiDimensionalReward()
    safety_reward = SafetyRewardModel()
    qa_reward = TaskSpecificReward("qa")
    
    def ensemble_reward(texts: List[str]) -> List[float]:
        """集成多个奖励模型的输出"""
        
        # 获取各个模型的评分
        multi_scores = multi_dim_reward(texts)
        safety_scores = safety_reward(texts)
        qa_scores = qa_reward(texts)
        
        # 权重集成
        ensemble_scores = []
        for m_score, s_score, q_score in zip(multi_scores, safety_scores, qa_scores):
            ensemble_score = (
                0.4 * m_score +      # 多维度评分
                0.4 * s_score +      # 安全性评分  
                0.2 * q_score        # 任务特定评分
            )
            ensemble_scores.append(ensemble_score)
        
        return ensemble_scores
    
    return ensemble_reward

def demonstrate_reward_models():
    """演示不同奖励模型的效果"""
    
    print("🎯 奖励模型效果演示")
    
    # 测试文本样本
    test_texts = [
        "深度学习是一种机器学习方法，使用多层神经网络来学习数据的复杂模式，广泛应用于图像识别、自然语言处理等领域。",  # 高质量
        "深度学习就是学习学习学习很深很深的学习。",  # 低质量
        "我不确定深度学习的具体定义，但我知道它与神经网络相关，可能需要查阅更专业的资料。",  # 诚实但不确定
        "深度学习绝对是最好的技术，一定能解决所有问题！",  # 过于绝对
        "这是一个有害的内容，包含攻击性言论。"  # 有害内容
    ]
    
    # 创建不同的奖励模型
    multi_dim_reward = MultiDimensionalReward()
    safety_reward = SafetyRewardModel()
    qa_reward = TaskSpecificReward("qa")
    ensemble_reward = create_ensemble_reward_model()
    
    print("\n📊 奖励模型评分对比:")
    print("=" * 80)
    
    for i, text in enumerate(test_texts):
        print(f"\n📝 文本 {i+1}: {text[:50]}...")
        
        multi_score = multi_dim_reward([text])[0]
        safety_score = safety_reward([text])[0]
        qa_score = qa_reward([text])[0]
        ensemble_score = ensemble_reward([text])[0]
        
        print(f"  🎯 多维度奖励: {multi_score:.3f}")
        print(f"  🛡️  安全性奖励: {safety_score:.3f}")
        print(f"  ❓ 问答奖励: {qa_score:.3f}")
        print(f"  🏆 集成奖励: {ensemble_score:.3f}")

class CustomPPOConfig:
    """自定义PPO配置"""
    
    def __init__(self):
        self.base_config = {
            'learning_rate': 1.41e-5,
            'batch_size': 32,
            'mini_batch_size': 4,
            'ppo_epochs': 4,
            'cliprange': 0.2,
            'vf_coef': 0.1,
            'target_kl': 0.1,
        }
        
        # 自适应参数
        self.adaptive_params = {
            'lr_scheduler': 'cosine',
            'early_stopping': True,
            'patience': 5,
            'min_improvement': 0.01
        }
    
    def get_adaptive_lr(self, step: int, total_steps: int) -> float:
        """自适应学习率"""
        if self.adaptive_params['lr_scheduler'] == 'cosine':
            return self.base_config['learning_rate'] * (
                0.5 * (1 + np.cos(np.pi * step / total_steps))
            )
        return self.base_config['learning_rate']
    
    def should_stop_early(self, recent_rewards: List[float]) -> bool:
        """早停判断"""
        if not self.adaptive_params['early_stopping']:
            return False
        
        if len(recent_rewards) < self.adaptive_params['patience']:
            return False
        
        # 检查最近几步是否有改进
        recent_mean = np.mean(recent_rewards[-3:])
        earlier_mean = np.mean(recent_rewards[-6:-3])
        
        improvement = recent_mean - earlier_mean
        return improvement < self.adaptive_params['min_improvement']

def advanced_training_strategies():
    """演示高级训练策略"""
    
    print("🔧 高级训练策略演示")
    
    # 1. curriculum learning示例
    def curriculum_learning_schedule(step: int) -> Dict[str, float]:
        """课程学习：逐步增加任务难度"""
        
        if step < 50:
            # 简单阶段：只关注基础质量
            return {'quality': 1.0, 'complexity': 0.0, 'creativity': 0.0}
        elif step < 100:
            # 中等阶段：增加复杂性要求
            return {'quality': 0.7, 'complexity': 0.3, 'creativity': 0.0}
        else:
            # 高级阶段：全面评估
            return {'quality': 0.5, 'complexity': 0.3, 'creativity': 0.2}
    
    # 2. 温度调度示例  
    def temperature_schedule(step: int) -> float:
        """温度调度：控制探索vs利用"""
        # 开始时高温度(更多探索)，逐渐降低(更多利用)
        initial_temp = 1.0
        final_temp = 0.1
        decay_steps = 200
        
        if step >= decay_steps:
            return final_temp
        
        decay_ratio = step / decay_steps
        return initial_temp * (1 - decay_ratio) + final_temp * decay_ratio
    
    # 3. 演示调度效果
    steps = list(range(0, 250, 10))
    
    print("\n📈 训练策略调度演示:")
    
    for step in [0, 50, 100, 150, 200]:
        weights = curriculum_learning_schedule(step)
        temp = temperature_schedule(step)
        
        print(f"步骤 {step:3d}: 权重 {weights}, 温度 {temp:.2f}")

def main():
    print("🎯 TRL高级定制功能演示")
    
    choice = input("\n选择演示: (1)奖励模型对比 (2)训练策略 (3)全部: ")
    
    if choice == "1" or choice == "3":
        demonstrate_reward_models()
    
    if choice == "2" or choice == "3":
        advanced_training_strategies()
    
    print("\n✨ 高级功能演示完成")
    print("💡 提示: 在实际项目中，根据具体需求选择和组合这些技术")

if __name__ == "__main__":
    main()