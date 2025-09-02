#!/bin/bash
# TRL环境安装脚本 (使用uv包管理器)

echo "🚀 开始安装TRL学习环境..."

# 检查uv是否已安装
if ! command -v uv &> /dev/null; then
    echo "📦 安装uv包管理器..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # 重新加载shell配置
    source ~/.bashrc 2>/dev/null || source ~/.zshrc 2>/dev/null || true
fi

# 检查Python版本
python_version=$(python3 --version 2>&1)
echo "Python版本: $python_version"

# 创建虚拟环境
echo "🐍 创建Python虚拟环境..."
uv venv trl_env

# 激活虚拟环境
echo "⚡ 激活虚拟环境..."
source trl_env/bin/activate

# 使用uv安装依赖
echo "📦 使用uv安装依赖包..."
uv pip install -r requirements.txt

echo "✅ 安装完成！"
echo ""
echo "🎯 使用方法："
echo "1. 激活环境: source trl_env/bin/activate"
echo "2. 验证安装: python setup_environment.py"
echo "3. 开始学习: python project/run_project.py"