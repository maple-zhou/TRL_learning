#!/usr/bin/env python3
"""
交互式PPO演示
让你直接体验PPO训练前后的模型差异
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import AutoModelForCausalLMWithValueHead

def load_original_model():
    """加载原始GPT2模型"""
    print("📚 加载原始GPT2模型...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def load_trained_model():
    """加载PPO训练后的模型"""
    try:
        print("🎯 加载PPO训练后的模型...")
        tokenizer = AutoTokenizer.from_pretrained("./models/ppo_gpt2_simple")
        model = AutoModelForCausalLMWithValueHead.from_pretrained("./models/ppo_gpt2_simple")
        return model, tokenizer
    except Exception as e:
        print(f"❌ 无法加载训练后的模型: {e}")
        print("请先运行 01_simple_ppo_example.py 训练模型")
        return None, None

def generate_text(model, tokenizer, prompt, max_length=100):
    """生成文本"""
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=max_length,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id
        )
    
    # 如果是带value head的模型，只取logits部分
    if hasattr(model, 'pretrained_model'):
        # 对于带value head的模型，直接使用generate的输出
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    else:
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return generated_text

def compare_models():
    """比较训练前后的模型输出"""
    
    # 加载模型
    original_model, original_tokenizer = load_original_model()
    trained_model, trained_tokenizer = load_trained_model()
    
    if trained_model is None:
        return
    
    test_prompts = [
        "写一首关于春天的诗：",
        "给我一个健康饮食建议：",
        "推荐一本好书：",
        "描述你的理想假期：",
        "解释什么是人工智能："
    ]
    
    print("\n" + "="*60)
    print("🔍 模型对比测试")
    print("="*60)
    
    for i, prompt in enumerate(test_prompts):
        print(f"\n🎯 测试 {i+1}: {prompt}")
        print("-" * 50)
        
        # 原始模型输出
        print("📖 原始GPT2输出:")
        original_output = generate_text(original_model, original_tokenizer, prompt)
        print(original_output[len(prompt):])  # 只显示生成的部分
        
        print("\n🎯 PPO训练后输出:")
        trained_output = generate_text(trained_model, trained_tokenizer, prompt)
        print(trained_output[len(prompt):])  # 只显示生成的部分
        
        print("-" * 50)

def interactive_test():
    """交互式测试"""
    trained_model, trained_tokenizer = load_trained_model()
    
    if trained_model is None:
        return
    
    print("\n🎮 交互式测试模式")
    print("输入提示词，模型会生成回复 (输入 'quit' 退出)")
    
    while True:
        prompt = input("\n💭 你的提示: ")
        if prompt.lower() == 'quit':
            break
        
        output = generate_text(trained_model, trained_tokenizer, prompt)
        print(f"🤖 模型回复: {output[len(prompt):]}")

def main():
    print("🎯 PPO模型演示程序")
    
    while True:
        print("\n选择操作:")
        print("1. 比较训练前后模型")
        print("2. 交互式测试")
        print("3. 退出")
        
        choice = input("请选择 (1-3): ")
        
        if choice == "1":
            compare_models()
        elif choice == "2":
            interactive_test()
        elif choice == "3":
            print("👋 再见！")
            break
        else:
            print("❌ 无效选择，请重新输入")

if __name__ == "__main__":
    main()