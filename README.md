# -PhySO- 物理符号回归建模方法 —— 中文注释项目

本项目对 PhySO 开展完整的中文注释，从部署到精通，再到如何实际应用到自己所以学科，即使你是一个没有机器学习基础的科研人员，也基本可以看懂。让我们一起，推动 AI for Science 的进步。

## 一些官方资料

出自以下项目，随着时间推移作者已完整更新该算法，中文注释项目也迎来完整更新：https://github.com/WassimTenachi/PhySO

参考论文： https://arxiv.org/abs/2303.03192

参考官方手册： [Welcome to PhySO’s documentation! — PhySO  documentation](https://physo.readthedocs.io/en/latest/)  

## 快速安装和使用

注意： 由于 windows 系统的局限性，项目运行速度大幅度慢于 linux 系统，但如果计算量较小，可以忽略这个差异。

Latex ，不安装也可以，推荐安装一下。

仅介绍 windows （ Linux 环境下按照官方教程正常安装即可），推荐 python 3.8 ，不支持新版 torch ，`requirements.txt`文件`jupyterlab`，删除 `pytorch>=1.11.0` ，然后安装，命令行输入以下代码，注意路径部分修改：

```bash
conda activate physo
S:\condaenv\physo\python.exe -m pip install --upgrade pip
conda install pytorch
cd PhySO
conda install ipykernel
conda install matplotlib
conda install pandas tqdm scikit-learn
pip3 install -r requirements.txt
python -m pip install -e .
conda install jupyterlab
```

测试（ 这里也推荐 VS Code ），也可以直接输入：

```bash
jupyter lab
```

[JupyterLab](http://localhost:8888/lab) 打开`class_sr_quick_start.ipynb`  (`physo` 执行符号回归 （SR）)，实测 windows 下需要一个小时左右。

符号回归（SR）是指推断一个自由形式的符号解析函数 $f: \mathbb{R}^n \longrightarrow \mathbb{R}$，该函数能够拟合给定的 $(x_0,..., x_n, y)$ 数据，即 $y = f(x_0,..., x_n)$。$\Phi$-SO 可以利用量纲分析（DA）使符号回归（SR）更加高效。无论单位的顺序如何，都可以进行维度分析，只要在整个 X、y 和常数之间保持一致，用户可以使用任何约定（例如 [长度, 质量, 时间] 或 [质量, 时间, 长度] 等）。

或者

```bash
cd PhySO\demos
python sr_quick_start.py
```

## 本项目文件构成

PhySO 文件夹，2024年12月版本该项目内容。

202303_version 文件夹，上一版项目内容加部分注释。

## 亮点

Φ -SO 的符号回归模块使用深度强化学习来推断适合数据点的分析物理定律，在函数形式的空间中搜索。

`physo` 能够用于：

1. 物理单位约束，通过维度分析减少搜索空间([[Tenachi et al 2023]](https://arxiv.org/abs/2303.03192))

2. 类约束，搜索准确拟合多个数据集的单个分析函数形式 - 每个数据集都由自己（可能）独特的拟合参数集控制 ([[Tenachi et al 2024]](https://arxiv.org/abs/2312.01816)) 

在 [SRBench](https://github.com/cavalab/srbench/tree/master) 的标准 Feynman 基准测试上的性能），包括 Feynman Lectures on Physics 中的 120 个表达式，与流行的 SR 包相对应。

Φ -SO 在存在噪声（超过 0.1%）的情况下实现了最先进的性能，即使在存在大量 （10%） 噪声的情况下也表现出稳健的性能:

![bbb051a2-2737-40ca-bfbf-ed185c48aa71](https://github.com/WassimTenachi/PhySO/assets/63928316/bbb051a2-2737-40ca-bfbf-ed185c48aa71) 






