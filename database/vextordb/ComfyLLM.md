
# 1. ComfyLLM介绍
**ComfyLLM**是基于ComfyUI的一款插件，旨在**用可视化方式辅助用户理解人工智能大模型以及构建大模型相关应用**，相关应用可以辅助组织或者个人建立数据搜集与分析系统，为建立“信息化主体”开辟道路。
什么是“信息化主体”？“信息化主体”是指用规模化流程化方法对获取信息进行搜集、加工、处理与应用的组织或者个人。以个人为例，传统层面个人靠强大记忆力已经能基本完成各项社会任务，但随信息大爆炸时代的到来，个人记忆力在信息处理层面上的可靠性逐渐降低。信息遗忘与信息混淆问题为“外脑”软件如Notion,Obsidian等提供了广阔市场，但是这些软件固有缺陷在于它们是为信息存储与检索设计的，并不能为旧的信息赋予新的价值。世界是广泛联系着的，因此**以信息处理视角构建新的“外脑”** 必然优于以数据存储视角构建“外脑”，ComfyLLM的使命就是用可视化方式构建信息的处理流程，完成信息处理闭环。
为什么要构建“信息化主体”？前三次工业革命为人类社会带来了天翻地覆的变化，轻重工业基础以及高精尖工业基础在物质增密的演进过程中逐渐被确立。信息增密与处理是四次工业革命的重要组成部分，随着越来越多的人认识到终身学习的重要性，如何**从软件层面构建新的引擎来使得个人完成信息化与智能化转型**是我们这代人要回答的问题。


# 可视化计算图


# 2. ComfyLLM文件组成 
├── __init__.py
├── lib：类库文件，用于构建复杂逻辑
├── models：大模型以及权重文件以及缓存
├── nodes：节点文件
├── test：测试文件
└── utils：辅助文件

# 3. ComfyLLM应用拓展

## 3.1大模型组件开发与构建

## 3.2 大模型部署
## 3.3 大模型信息整合系统
### 3.3.1基于RAG的个人信息分析系统
用本地模型与RAG建立信息检索分析系统
### 3.3.2 基于API的数据增广系统
用开放大语言模型对本地数据进行增广
### 3.3.3 基于LoRA的本地数据资产
用LoRA进行训练