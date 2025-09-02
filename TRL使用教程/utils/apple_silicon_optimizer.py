#!/usr/bin/env python3
"""
苹果芯片Mac优化的设备检测和配置工具
"""

import torch
import platform
import psutil
from typing import Dict, Any

def detect_apple_silicon():
    """检测是否为苹果芯片"""
    try:
        # 检查是否为macOS ARM64
        return platform.system() == "Darwin" and platform.machine() == "arm64"
    except:
        return False

def get_optimal_device():
    """获取最优计算设备"""
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

def get_device_info():
    """获取详细设备信息"""
    info = {
        "platform": platform.system(),
        "architecture": platform.machine(),
        "is_apple_silicon": detect_apple_silicon(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
    }
    
    # 设备支持
    info["mps_available"] = torch.backends.mps.is_available()
    info["mps_built"] = torch.backends.mps.is_built() if hasattr(torch.backends.mps, 'is_built') else False
    info["cuda_available"] = torch.cuda.is_available()
    info["optimal_device"] = get_optimal_device()
    
    # 内存信息
    memory = psutil.virtual_memory()
    info["total_memory_gb"] = round(memory.total / (1024**3), 1)
    info["available_memory_gb"] = round(memory.available / (1024**3), 1)
    
    return info

def get_apple_silicon_config():
    """获取苹果芯片优化的训练配置"""
    
    device_info = get_device_info()
    memory_gb = device_info["total_memory_gb"]
    
    # 根据内存大小调整配置
    if memory_gb >= 32:
        # 高配置 (32GB+)
        config = {
            "batch_size": 8,
            "mini_batch_size": 4,
            "gradient_accumulation_steps": 2,
            "max_length": 256,
        }
    elif memory_gb >= 16:
        # 中等配置 (16-32GB)
        config = {
            "batch_size": 4,
            "mini_batch_size": 2,
            "gradient_accumulation_steps": 4,
            "max_length": 128,
        }
    else:
        # 低配置 (8-16GB)
        config = {
            "batch_size": 2,
            "mini_batch_size": 1,
            "gradient_accumulation_steps": 8,
            "max_length": 64,
        }
    
    # 通用苹果芯片优化
    config.update({
        "learning_rate": 5e-6,          # 稍微降低学习率
        "dataloader_num_workers": 0,    # 苹果芯片推荐
        "optimize_cuda_cache": False,   # 关闭CUDA优化
        "fp16": False,                  # MPS可能不完全支持fp16
        "device": get_optimal_device(),
    })
    
    return config

def apply_apple_silicon_optimizations():
    """应用苹果芯片优化设置"""
    
    if detect_apple_silicon():
        print("🍎 检测到苹果芯片，应用优化设置...")
        
        # 设置环境变量
        import os
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # 启用MPS回退
        os.environ["TOKENIZERS_PARALLELISM"] = "false"   # 避免tokenizer并行问题
        
        # 显示建议
        print("💡 苹果芯片优化建议:")
        print("   - 使用MPS设备加速")
        print("   - 设置较小的batch_size")
        print("   - 关闭dataloader多进程")
        print("   - 监控内存使用情况")
        
        return True
    else:
        print("🖥️  非苹果芯片设备，使用标准配置")
        return False

def show_device_recommendations():
    """显示设备相关建议"""
    
    info = get_device_info()
    config = get_apple_silicon_config()
    
    print("🔍 设备信息和建议配置")
    print("=" * 50)
    
    print(f"平台: {info['platform']} {info['architecture']}")
    print(f"Python: {info['python_version']}")
    print(f"PyTorch: {info['pytorch_version']}")
    print(f"总内存: {info['total_memory_gb']}GB")
    print(f"可用内存: {info['available_memory_gb']}GB")
    
    print(f"\n🎯 推荐设备: {info['optimal_device']}")
    
    if info["is_apple_silicon"]:
        print("🍎 苹果芯片专用优化:")
        print(f"   - MPS可用: {info['mps_available']}")
        print(f"   - 建议batch_size: {config['batch_size']}")
        print(f"   - 建议max_length: {config['max_length']}")
    
    print(f"\n📊 推荐训练配置:")
    for key, value in config.items():
        if key != "device":
            print(f"   {key}: {value}")

def create_device_specific_config():
    """创建设备特定的配置文件"""
    
    config = get_apple_silicon_config()
    device_info = get_device_info()
    
    # 生成配置字典
    full_config = {
        "device_info": device_info,
        "training_config": config,
        "optimization_tips": [
            "使用较小的batch_size减少内存使用",
            "利用gradient_accumulation_steps保持有效批次大小", 
            "在苹果芯片上关闭CUDA相关优化",
            "监控内存使用，避免交换",
            "使用MPS设备获得最佳性能"
        ]
    }
    
    # 保存配置
    import json
    with open("apple_silicon_config.json", "w", encoding="utf-8") as f:
        json.dump(full_config, f, indent=2, ensure_ascii=False)
    
    print("💾 设备配置已保存到 apple_silicon_config.json")

def main():
    print("🍎 苹果芯片Mac优化工具")
    print("=" * 40)
    
    # 检测并优化
    apply_apple_silicon_optimizations()
    
    # 显示设备信息
    show_device_recommendations()
    
    # 创建配置文件
    create_device_specific_config()
    
    print("\n✅ 苹果芯片优化完成！")
    print("现在可以正常运行所有TRL教程了")

if __name__ == "__main__":
    main()