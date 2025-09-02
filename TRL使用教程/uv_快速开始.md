# 快速开始指南 (uv版本)

## 🚀 30秒快速开始

```bash
# 1. 安装uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 创建并激活环境
./install.sh
source trl_env/bin/activate

# 3. 验证安装
python setup_environment.py

# 4. 开始学习
python scripts/01_simple_ppo_example.py
```

## 🎯 不同学习路径

### 路径1：理论优先 (推荐初学者)
```bash
# 1. 阅读基础概念
cat 01_TRL基础概念.md

# 2. 理解PPO算法
cat 03_PPO基础教程.md

# 3. 动手实践
jupyter notebook notebooks/PPO入门实践.ipynb
```

### 路径2：实践优先 (有经验者)
```bash
# 1. 直接运行示例
python scripts/01_simple_ppo_example.py

# 2. 尝试DPO训练
python scripts/05_dpo_training_example.py

# 3. 完整项目实战
python project/complete_rlhf_project.py
```

### 路径3：项目驱动
```bash
# 直接运行完整学习流程
python project/run_project.py --mode learn
```

## 🛠️ uv命令速查

```bash
# 环境管理
uv venv trl_env                    # 创建环境
source trl_env/bin/activate        # 激活环境
uv pip list                        # 查看已安装包

# 包管理
uv pip install package_name        # 安装包
uv pip install -e .               # 安装项目依赖
uv pip install -r requirements.txt # 从文件安装
uv pip install --upgrade package   # 更新包

# 依赖管理
uv pip freeze > requirements.txt   # 生成依赖文件
uv pip sync requirements.txt       # 同步依赖
uv pip compile pyproject.toml      # 编译依赖

# 项目管理
uv pip install -e ".[dev]"        # 安装开发依赖
uv pip install -e ".[gpu]"        # 安装GPU依赖
```

## ⚡ uv的核心优势

- **🚀 极速**: 比pip快10-100倍
- **🔒 锁定**: 自动生成lock文件确保一致性
- **🎯 精确**: 更好的依赖冲突解决
- **💾 智能**: 全局缓存和增量安装
- **🔧 现代**: 支持PEP标准和现代Python工作流

## 📚 学习资源链接

- [uv官方文档](https://docs.astral.sh/uv/)
- [TRL官方文档](https://huggingface.co/docs/trl)
- [Transformers文档](https://huggingface.co/docs/transformers)
- [PyTorch教程](https://pytorch.org/tutorials/)