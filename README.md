# -PhySO- 物理符号回归建模方法 —— 中文注释项目



本项目对 PhySO 开展完整的中文注释，从部署到精通，再到如何实际应用到自己所以学科，即使你是一个没有机器学习基础的科研人员，也基本可以看懂。让我们一起，推动 AI for Science 的进步。

## 一些官方资料

出自以下项目，随着时间推移作者已完整更新该算法，中文注释项目也迎来完整更新：https://github.com/WassimTenachi/PhySO

参考论文： https://arxiv.org/abs/2303.03192

参考官方手册： [Welcome to PhySO’s documentation! — PhySO  documentation](https://physo.readthedocs.io/en/latest/)  



## 快速安装和使用

注意： 由于 windows 系统的局限性，项目运行速度大幅度慢于 linux 系统，但如果计算量较小，可以忽略这个差异。

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

[JupyterLab](http://localhost:8888/lab) 打开`class_sr_quick_start.ipynb`  ，实测 windows 下需要一个小时左右。

或者

```bash
cd PhySO\demos
python sr_quick_start.py
```



## 本项目文件构成

PhySO 文件夹，2024年12月版本该项目内容。

202303_version 文件夹，上一版项目内容加部分注释。




