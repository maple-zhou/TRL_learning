#!/usr/bin/env python3
"""
项目运行器 - 一键运行完整的TRL学习项目
"""

import os
import sys
import subprocess
import argparse

def run_command(command, description):
    """运行命令并显示进度"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description}完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description}失败: {e}")
        print(f"错误输出: {e.stderr}")
        return False

def check_environment():
    """检查环境是否就绪"""
    print("🔍 检查环境...")
    
    # 检查uv
    try:
        import subprocess
        result = subprocess.run(['uv', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ uv: {result.stdout.strip()}")
        else:
            print("⚠️  uv未找到，建议使用uv管理依赖")
    except:
        print("⚠️  uv未找到，建议使用uv管理依赖")
    
    try:
        import trl
        import torch
        import transformers
        print("✅ 环境检查通过")
        return True
    except ImportError as e:
        print(f"❌ 环境检查失败: {e}")
        print("请先运行: ./install.sh 或 uv pip install -r requirements.txt")
        return False

def run_learning_sequence():
    """按顺序运行学习示例"""
    
    scripts = [
        ("python scripts/01_simple_ppo_example.py", "PPO基础示例"),
        ("python scripts/03_reward_model_training.py", "奖励模型训练"),  
        ("python scripts/05_dpo_training_example.py", "DPO训练示例"),
        ("python scripts/06_advanced_reward_models.py", "高级奖励模型"),
    ]
    
    print("🎯 开始按序运行TRL学习示例...")
    
    for command, description in scripts:
        success = run_command(command, description)
        if not success:
            print(f"⏸️  在 {description} 步骤停止")
            return False
        
        print("-" * 50)
    
    print("🎉 所有学习示例运行完成！")
    return True

def run_complete_project():
    """运行完整项目"""
    print("🏆 启动完整项目实战...")
    
    success = run_command(
        "python project/complete_rlhf_project.py", 
        "智能客服助手项目"
    )
    
    if success:
        print("🎉 完整项目运行成功！")
    else:
        print("❌ 项目运行失败")

def launch_jupyter():
    """启动Jupyter notebooks"""
    print("📓 启动Jupyter notebooks...")
    
    try:
        # 启动Jupyter并打开notebooks目录
        os.system("jupyter notebook notebooks/")
    except KeyboardInterrupt:
        print("📓 Jupyter已关闭")

def show_project_status():
    """显示项目状态"""
    print("📊 TRL学习项目状态")
    print("=" * 40)
    
    # 检查文件结构
    expected_files = [
        "01_TRL基础概念.md",
        "scripts/01_simple_ppo_example.py",
        "notebooks/PPO入门实践.ipynb", 
        "project/complete_rlhf_project.py"
    ]
    
    for file_path in expected_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} (缺失)")
    
    # 检查模型目录
    models_dirs = ["models", "project/models"]
    for models_dir in models_dirs:
        if os.path.exists(models_dir):
            models = os.listdir(models_dir)
            print(f"\n📁 {models_dir}: {len(models)} 个模型")
            for model in models:
                print(f"   - {model}")
        else:
            print(f"\n📁 {models_dir}: 目录不存在")

def main():
    parser = argparse.ArgumentParser(description="TRL学习项目运行器")
    parser.add_argument("--mode", choices=["learn", "project", "jupyter", "status"], 
                       default="learn", help="运行模式")
    parser.add_argument("--check-env", action="store_true", help="仅检查环境")
    
    args = parser.parse_args()
    
    print("🎯 TRL学习项目运行器")
    print("=" * 40)
    
    # 检查环境
    if not check_environment():
        sys.exit(1)
    
    if args.check_env:
        return
    
    # 根据模式执行
    if args.mode == "learn":
        run_learning_sequence()
    elif args.mode == "project":
        run_complete_project()
    elif args.mode == "jupyter":
        launch_jupyter()
    elif args.mode == "status":
        show_project_status()

if __name__ == "__main__":
    main()