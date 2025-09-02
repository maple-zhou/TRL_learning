#!/usr/bin/env python3
"""
TRL环境验证和测试脚本 (uv环境)
运行此脚本来验证你的TRL环境是否正确安装
"""

import subprocess
import sys
import os

def check_uv_installation():
    """检查uv是否已安装"""
    print("🔍 检查uv包管理器...")
    
    try:
        result = subprocess.run(['uv', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ uv已安装: {result.stdout.strip()}")
            return True
        else:
            print("❌ uv未安装")
            return False
    except FileNotFoundError:
        print("❌ uv未安装")
        return False

def check_virtual_env():
    """检查是否在虚拟环境中"""
    print("\n🐍 检查虚拟环境...")
    
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ 当前在虚拟环境中")
        print(f"   环境路径: {sys.prefix}")
        return True
    else:
        print("⚠️  未检测到虚拟环境")
        print("   建议运行: source trl_env/bin/activate")
        return False

def check_imports():
    """检查所有必要的包是否能正确导入"""
    print("\n🔍 检查包导入...")
    
    required_packages = [
        ('torch', 'PyTorch'),
        ('transformers', 'Transformers'),
        ('trl', 'TRL'),
        ('datasets', 'Datasets'),
        ('accelerate', 'Accelerate'),
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('matplotlib', 'Matplotlib'),
    ]
    
    all_imported = True
    
    for package_name, display_name in required_packages:
        try:
            module = __import__(package_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"✅ {display_name}: {version}")
            
            # PyTorch特殊检查
            if package_name == 'torch':
                print(f"   CUDA可用: {module.cuda.is_available()}")
                if module.cuda.is_available():
                    print(f"   CUDA设备数: {module.cuda.device_count()}")
                    
        except ImportError:
            print(f"❌ {display_name}未安装")
            all_imported = False
    
    return all_imported

def test_basic_functionality():
    """测试基础功能"""
    print("\n🧪 测试基础功能...")
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from trl import PPOConfig
        
        # 测试配置创建 (新版本TRL不需要model_name参数)
        try:
            config = PPOConfig()  # 新版本方式
            print("✅ PPO配置创建成功 (新版本TRL)")
        except Exception:
            # 尝试旧版本方式
            config = PPOConfig(model_name="gpt2")
            print("✅ PPO配置创建成功 (旧版本TRL)")
        
        # 测试tokenizer加载
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("✅ Tokenizer加载成功")
        
        # 测试设备支持
        import torch
        if torch.backends.mps.is_available():
            print("✅ 苹果芯片MPS加速可用")
        elif torch.cuda.is_available():
            print("✅ CUDA GPU加速可用")
        else:
            print("✅ CPU模式可用")
        
        print("✅ 基础功能测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 基础功能测试失败: {e}")
        print("💡 提示: 这可能是TRL版本兼容性问题")
        return False

def check_project_structure():
    """检查项目目录结构"""
    print("\n📁 检查项目结构...")
    
    required_dirs = ["notebooks", "scripts", "models", "data", "configs", "utils"]
    required_files = ["pyproject.toml", "requirements.txt", "install.sh"]
    
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"✅ 目录存在: {dir_name}/")
        else:
            print(f"⚠️  目录缺失: {dir_name}/")
    
    for file_name in required_files:
        if os.path.exists(file_name):
            print(f"✅ 文件存在: {file_name}")
        else:
            print(f"⚠️  文件缺失: {file_name}")

def create_project_structure():
    """创建项目目录结构"""
    print("\n📁 创建项目目录结构...")
    
    import os
    
    dirs = [
        "notebooks",
        "scripts", 
        "models",
        "data",
        "configs",
        "utils"
    ]
    
    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)
        print(f"✅ 创建目录: {dir_name}/")

def show_next_steps():
    """显示下一步操作建议"""
    print("\n📚 接下来你可以:")
    print("1. 查看学习指南: cat 学习指南.md")
    print("2. 阅读理论基础: cat 01_TRL基础概念.md")
    print("3. 运行PPO示例: python scripts/01_simple_ppo_example.py")
    print("4. 启动Jupyter: jupyter notebook notebooks/")
    print("5. 运行完整项目: python project/run_project.py")

def show_uv_tips():
    """显示uv使用技巧"""
    print("\n💡 uv使用技巧:")
    print("- 查看已安装包: uv pip list")
    print("- 安装新包: uv pip install package_name")
    print("- 更新包: uv pip install --upgrade package_name")
    print("- 生成锁定文件: uv pip freeze > requirements.txt")
    print("- 同步环境: uv pip sync requirements.txt")

def main():
    print("🎯 TRL环境设置向导 (uv版本)")
    print("=" * 50)
    
    # 检查uv
    if not check_uv_installation():
        print("\n❌ uv未安装。请运行以下命令安装:")
        print("curl -LsSf https://astral.sh/uv/install.sh | sh")
        return
    
    # 检查虚拟环境
    in_venv = check_virtual_env()
    
    # 检查导入
    if not check_imports():
        print("\n❌ 环境检查失败。请运行安装脚本:")
        if in_venv:
            print("uv pip install -r requirements.txt")
        else:
            print("./install.sh")
        return
    
    # 测试功能
    if not test_basic_functionality():
        print("\n❌ 功能测试失败。请检查安装。")
        return
    
    # 检查项目结构
    check_project_structure()
    
    print("\n🎉 环境搭建完成！")
    
    # 显示后续步骤
    show_next_steps()
    show_uv_tips()

if __name__ == "__main__":
    main()