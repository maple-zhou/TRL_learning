# PPO算法基础与实践

## 1. PPO算法原理

### 什么是PPO？
PPO (Proximal Policy Optimization) 是一种策略梯度强化学习算法，专门设计来稳定地改进策略。

### 核心思想
- **代理目标函数**: 通过限制策略更新幅度来避免破坏性更新
- **重要性采样**: 使用旧策略收集的数据来更新新策略
- **剪切机制**: 防止新旧策略差异过大

### 数学公式
```
L^CLIP(θ) = E[min(r_t(θ)A_t, clip(r_t(θ), 1-ε, 1+ε)A_t)]
```
其中：
- r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t) (重要性比率)
- A_t 是优势函数
- ε 是剪切参数 (通常0.2)

## 2. TRL中的PPO工作流程

```python
# 1. 准备模型和数据
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
dataset = load_dataset("imdb")

# 2. 配置PPO参数
ppo_config = PPOConfig(
    model_name="gpt2",
    learning_rate=1.41e-5,
    batch_size=32,
    mini_batch_size=4
)

# 3. 创建PPO训练器
ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=model,
    ref_model=None,  # 参考模型
    tokenizer=tokenizer,
    dataset=dataset
)

# 4. 训练循环
for batch in ppo_trainer.dataloader:
    # 生成回复
    response_tensors = ppo_trainer.generate(
        batch["input_ids"],
        return_prompt=False
    )
    
    # 计算奖励
    rewards = compute_rewards(batch, response_tensors)
    
    # PPO更新
    stats = ppo_trainer.step(
        batch["input_ids"],
        response_tensors,
        rewards
    )
```

## 3. 关键概念解释

### 奖励函数 (Reward Function)
- 定义模型输出的好坏
- 可以是人工设计的规则
- 也可以是训练好的奖励模型

### 参考模型 (Reference Model)
- 通常是原始的预训练模型
- 用于计算KL散度惩罚
- 防止模型偏离太远

### 优势函数 (Advantage Function)
- 衡量某个动作比平均水平好多少
- 帮助模型学习哪些行为应该被强化