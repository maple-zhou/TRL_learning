#!/usr/bin/env python3
"""
自定义训练策略实现
演示如何实现各种高级训练技巧和策略
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from datasets import Dataset
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional
import math

class CurriculumLearning:
    """课程学习实现"""
    
    def __init__(self):
        self.current_stage = 0
        self.stage_thresholds = [0.3, 0.6, 0.8]  # 进入下一阶段的奖励阈值
        
        self.stages = [
            {
                'name': '基础阶段',
                'description': '学习基本的回复格式和礼貌用语',
                'reward_weights': {'politeness': 0.8, 'length': 0.2, 'coherence': 0.0},
                'data_complexity': 'simple'
            },
            {
                'name': '中级阶段', 
                'description': '提高回复的连贯性和信息含量',
                'reward_weights': {'politeness': 0.3, 'length': 0.2, 'coherence': 0.5},
                'data_complexity': 'medium'
            },
            {
                'name': '高级阶段',
                'description': '生成创造性和深度的回复',
                'reward_weights': {'politeness': 0.2, 'length': 0.1, 'coherence': 0.3, 'creativity': 0.4},
                'data_complexity': 'complex'
            }
        ]
    
    def should_advance_stage(self, recent_rewards: List[float]) -> bool:
        """判断是否应该进入下一阶段"""
        if self.current_stage >= len(self.stage_thresholds):
            return False
        
        if len(recent_rewards) < 10:
            return False
        
        avg_reward = np.mean(recent_rewards[-10:])
        threshold = self.stage_thresholds[self.current_stage]
        
        return avg_reward >= threshold
    
    def advance_stage(self):
        """进入下一阶段"""
        if self.current_stage < len(self.stages) - 1:
            self.current_stage += 1
            print(f"🎓 进入{self.stages[self.current_stage]['name']}")
            print(f"   {self.stages[self.current_stage]['description']}")
    
    def get_current_stage_info(self) -> Dict:
        """获取当前阶段信息"""
        return self.stages[self.current_stage]
    
    def compute_stage_reward(self, texts: List[str]) -> List[float]:
        """根据当前阶段计算奖励"""
        stage_info = self.get_current_stage_info()
        weights = stage_info['reward_weights']
        
        # 计算各个维度的得分
        politeness_scores = self._compute_politeness(texts)
        length_scores = self._compute_length_score(texts)
        coherence_scores = self._compute_coherence(texts)
        creativity_scores = self._compute_creativity(texts) if 'creativity' in weights else [0] * len(texts)
        
        # 加权平均
        final_rewards = []
        for p, l, c, cr in zip(politeness_scores, length_scores, coherence_scores, creativity_scores):
            reward = (
                weights.get('politeness', 0) * p +
                weights.get('length', 0) * l +
                weights.get('coherence', 0) * c +
                weights.get('creativity', 0) * cr
            )
            final_rewards.append(reward)
        
        return final_rewards
    
    def _compute_politeness(self, texts: List[str]) -> List[float]:
        """计算礼貌程度"""
        polite_words = ['请', '谢谢', '您好', '不好意思', 'please', 'thank', 'sorry']
        scores = []
        
        for text in texts:
            score = 0.5  # 基础分
            text_lower = text.lower()
            
            for word in polite_words:
                if word in text_lower:
                    score += 0.1
            
            scores.append(min(1.0, score))
        
        return scores
    
    def _compute_length_score(self, texts: List[str]) -> List[float]:
        """计算长度合理性"""
        scores = []
        
        for text in texts:
            length = len(text)
            
            if 20 <= length <= 100:
                score = 1.0
            elif 10 <= length < 20 or 100 < length <= 150:
                score = 0.7
            elif length < 10:
                score = 0.3
            else:
                score = 0.5
            
            scores.append(score)
        
        return scores
    
    def _compute_coherence(self, texts: List[str]) -> List[float]:
        """计算连贯性"""
        scores = []
        
        for text in texts:
            # 简单的连贯性检查
            sentences = text.split('。')
            
            if len(sentences) <= 1:
                score = 0.5
            else:
                # 检查句子之间的连接词
                connectors = ['因此', '所以', '然而', '但是', '并且', '而且']
                connector_count = sum(1 for conn in connectors if conn in text)
                score = min(1.0, 0.3 + connector_count * 0.2)
            
            scores.append(score)
        
        return scores
    
    def _compute_creativity(self, texts: List[str]) -> List[float]:
        """计算创造性"""
        creative_words = ['创新', '独特', '新颖', '想象', 'creative', 'innovative', 'unique']
        scores = []
        
        for text in texts:
            score = 0.3  # 基础创造性分数
            text_lower = text.lower()
            
            # 检查创造性词汇
            for word in creative_words:
                if word in text_lower:
                    score += 0.2
            
            # 检查词汇多样性
            words = text.split()
            if len(words) > 0:
                unique_ratio = len(set(words)) / len(words)
                score += unique_ratio * 0.3
            
            scores.append(min(1.0, score))
        
        return scores

class OnlineRewardLearning:
    """在线奖励学习"""
    
    def __init__(self):
        self.reward_buffer = []
        self.quality_buffer = []
        self.learning_rate = 0.01
        
        # 可学习的奖励权重
        self.learned_weights = {
            'length': 0.3,
            'sentiment': 0.3, 
            'coherence': 0.4
        }
    
    def update_reward_function(self, texts: List[str], human_scores: List[float]):
        """根据人类反馈更新奖励函数"""
        
        # 计算当前奖励函数的预测
        predicted_scores = self.compute_learned_rewards(texts)
        
        # 计算预测误差
        errors = [h - p for h, p in zip(human_scores, predicted_scores)]
        
        # 更新权重（简化的梯度下降）
        features = self._extract_features(texts)
        
        for i, error in enumerate(errors):
            for feature_name, feature_value in features[i].items():
                if feature_name in self.learned_weights:
                    self.learned_weights[feature_name] += self.learning_rate * error * feature_value
        
        # 归一化权重
        total_weight = sum(self.learned_weights.values())
        for key in self.learned_weights:
            self.learned_weights[key] /= total_weight
        
        print(f"🔄 奖励函数权重更新: {self.learned_weights}")
    
    def _extract_features(self, texts: List[str]) -> List[Dict[str, float]]:
        """提取文本特征"""
        features_list = []
        
        for text in texts:
            features = {
                'length': min(1.0, len(text) / 100),  # 归一化长度
                'sentiment': self._simple_sentiment(text),
                'coherence': self._simple_coherence(text)
            }
            features_list.append(features)
        
        return features_list
    
    def _simple_sentiment(self, text: str) -> float:
        """简单情感分析"""
        positive_words = ['好', '棒', '优秀', '喜欢', 'good', 'great', 'excellent']
        negative_words = ['坏', '差', '糟糕', 'bad', 'terrible', 'awful']
        
        pos_count = sum(1 for word in positive_words if word in text.lower())
        neg_count = sum(1 for word in negative_words if word in text.lower())
        
        return (pos_count - neg_count + 1) / 2  # 归一化到0-1
    
    def _simple_coherence(self, text: str) -> float:
        """简单连贯性评估"""
        sentences = text.split('。')
        if len(sentences) <= 1:
            return 0.5
        
        # 检查连接词
        connectors = ['因此', '所以', '然而', '但是']
        connector_count = sum(1 for conn in connectors if conn in text)
        return min(1.0, connector_count / len(sentences))
    
    def compute_learned_rewards(self, texts: List[str]) -> List[float]:
        """使用学习到的权重计算奖励"""
        features_list = self._extract_features(texts)
        rewards = []
        
        for features in features_list:
            reward = sum(
                self.learned_weights[name] * value 
                for name, value in features.items()
                if name in self.learned_weights
            )
            rewards.append(reward)
        
        return rewards

def demonstrate_curriculum_learning():
    """演示课程学习效果"""
    
    print("🎓 课程学习演示")
    
    curriculum = CurriculumLearning()
    
    # 模拟训练过程
    all_rewards = []
    stage_changes = []
    
    for step in range(100):
        # 模拟一个batch的奖励
        batch_rewards = np.random.normal(0.4 + step * 0.003, 0.1, 5)
        batch_rewards = [max(0, min(1, r)) for r in batch_rewards]
        all_rewards.extend(batch_rewards)
        
        # 检查是否需要进入下一阶段
        if curriculum.should_advance_stage(all_rewards):
            stage_changes.append(step)
            curriculum.advance_stage()
    
    # 绘制学习曲线
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    # 计算移动平均
    window_size = 10
    moving_avg = []
    for i in range(len(all_rewards)):
        start_idx = max(0, i - window_size + 1)
        moving_avg.append(np.mean(all_rewards[start_idx:i+1]))
    
    plt.plot(moving_avg)
    for change_step in stage_changes:
        plt.axvline(x=change_step*5, color='red', linestyle='--', alpha=0.7)
    plt.title('课程学习奖励曲线')
    plt.xlabel('训练步骤')
    plt.ylabel('平均奖励')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    stage_names = [stage['name'] for stage in curriculum.stages]
    stage_counts = [0] * len(stage_names)
    
    current_stage = 0
    for step in range(100):
        if step in [s//5 for s in stage_changes]:
            current_stage += 1
        if current_stage < len(stage_counts):
            stage_counts[current_stage] += 1
    
    plt.bar(stage_names, stage_counts)
    plt.title('各阶段训练步数分布')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    print("📊 课程学习可视化完成")

def main():
    print("🎯 高级训练策略演示")
    
    while True:
        print("\n选择演示:")
        print("1. 奖励模型对比")
        print("2. 课程学习策略")
        print("3. 在线奖励学习")
        print("4. 退出")
        
        choice = input("请选择 (1-4): ")
        
        if choice == "1":
            from scripts.advanced_reward_models import demonstrate_reward_models
            demonstrate_reward_models()
        elif choice == "2":
            demonstrate_curriculum_learning()
        elif choice == "3":
            online_learner = OnlineRewardLearning()
            
            # 演示在线学习
            sample_texts = ["这是一个很好的回答", "回答回答回答", "我认为这个问题很有趣"]
            human_feedback = [0.8, 0.2, 0.7]  # 模拟人类评分
            
            print("初始权重:", online_learner.learned_weights)
            online_learner.update_reward_function(sample_texts, human_feedback)
            print("更新后权重:", online_learner.learned_weights)
            
        elif choice == "4":
            break
        else:
            print("❌ 无效选择")

if __name__ == "__main__":
    main()