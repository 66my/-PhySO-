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

### 已完成

以下为翻译顺序，也是本项目作者非常推荐的源码学习路线图：

`-PhySO-\PhySO\demos\sr_quick_start.ipynb`，介绍了符号回归多自变量对单因变量的求解原理、概念与过程，调用 Pytorch ，建议在使用前对各变量数据进行归一化，数学原理上能够对各种物理量实现兼容，使用量纲分析（DA）提升符号回归的效果。

    传统符号回归传统符号回归（Symbolic Regression, SR）是一种机器学习方法，旨在发现能够拟合给定数据集的数学表达式。它通常使用遗传算法或其他搜索策略来探索可能的数学表达式空间，以找到最佳的解析形式。

`-PhySO-\PhySO\demos\class_sr_quick_start.ipynb`，符号回归类：

    符号回归类（Class SR）是一种自动寻找单一解析函数形式的方法，该函数能够准确拟合多个数据集。每个数据集可能遵循其独特的拟合参数集。这种层次框架利用了同一类物理现象的所有成员都遵循相同的基本定律这一共同约束。

`-PhySO-\PhySO\demos\sr\_deprecated-interface\demos_natural_const_discovery\demo_classical_gravity`完成。

`-PhySO-\PhySO\benchmarking\readme_reproducibility.md`基准测试手册；

demo文件夹其他文件注释完成。



在`-PhySO-\PhySO\demos\sr\_deprecated-interface\demo_damped_harmonic_oscillator`具有更为详细的帕累托前沿研究，直接调用 physo 库绘制帕累托前沿图。

## 可直接借鉴

`-PhySO-\PhySO\demos\sr\_deprecated-interface\demos_natural_const_discovery\demo_ideal_gas_law`演示理想气体状态方程求解过程。

`-PhySO-\PhySO\demos\sr\_deprecated-interface\demo_damped_harmonic_oscillator\demo_damped_harmonic_oscillator.ipynb` 用于演示阻尼谐振子的求解过程。生成了一个阻尼谐振子的响应数据，并绘制了相应的图形。

阻尼谐振子是一种常见的物理系统，其运动方程通常形式为：

- \( $\ddot{x}$ \) 是位移的二阶导数（加速度）。
- \( $\dot{x}$ \) 是位移的一阶导数（速度）。
- \( $\alpha$ \) 是阻尼系数。
- \( $\omega_0$ \) 是固有频率。
  通过生成的时间点和响应数据，可以进一步分析和可视化阻尼谐振子的行为。

`-PhySO-\PhySO\demos\sr\_deprecated-interface\demo_mechanical_energy\demo_mechanical_energy.ipynb` 预测分析机械能（势能和动能）变化。

`-PhySO-\PhySO\demos\sr\_deprecated-interface\demos_natural_const_discovery\demo_classical_gravity\demo_classical_gravity.ipynb`演示经典引力公式求解过程，模拟两个质点引力，验证牛顿万有引力公式。

`demos\sr\_deprecated-interface\demos_natural_const_discovery\demo_planck_law_nphotons\demo_planck_law_nphotons.ipynb`演示普朗克定律在计算光子数的应用，模拟黑体辐射的光子数分布。

`demos\sr\_deprecated-interface\demos_natural_const_discovery\demo_terminal_velocity\demo_terminal_velocity.py` 使用符号回归自动发现物体在流体中自由下落，重力与阻力平衡的速度。

`demos\sr\_deprecated-interface\demos_natural_const_discovery\demo_wave_interferences\demo_wave_interferences.py` 发现波干涉物理公式。



## 亮点

Φ -SO 的符号回归模块使用深度强化学习来推断适合数据点的分析物理定律，在函数形式的空间中搜索。

`physo` 能够用于：

1. 物理单位约束，通过维度分析减少搜索空间([[Tenachi et al 2023]](https://arxiv.org/abs/2303.03192))

2. 类约束，搜索准确拟合多个数据集的单个分析函数形式 - 每个数据集都由自己（可能）独特的拟合参数集控制 ([[Tenachi et al 2024]](https://arxiv.org/abs/2312.01816)) 

在 [SRBench](https://github.com/cavalab/srbench/tree/master) 的标准 Feynman 基准测试上的性能），包括 Feynman Lectures on Physics 中的 120 个表达式，与流行的 SR 包相对应。

Φ -SO 在存在噪声（超过 0.1%）的情况下实现了最先进的性能，即使在存在大量 （10%） 噪声的情况下也表现出稳健的性能:

![bbb051a2-2737-40ca-bfbf-ed185c48aa71](https://github.com/WassimTenachi/PhySO/assets/63928316/bbb051a2-2737-40ca-bfbf-ed185c48aa71) 
