# -PhySO- 物理符号回归建模方法 —— 中文注释项目

在该项目更新前，第一版已收获 60 star ，相信非常多开发者需要这份注释 ，因此进行大型更新

在demo_damped_harmonic_oscillator.ipynb文件中，有关于本模型使用原理的逐行注释，即使你是一个没有机器学习基础的科研人员，也基本可以看懂。

demo出自以下项目，再次对这位作者的贡献致意：https://github.com/WassimTenachi/PhySO

同样的，如果想本地部署PhySO库，也请参照上述网址

需要借助于Jupiter Lab或者Jupiter NoteBook打开，基于Python语言。（当然也可以在github直接打开）

## 亮点

使用Windows环境（Python3.10）完成了测试，而WassimTenachi仅完成了Linux与OSX（ARM和英特尔）的测试。

## physo基于以下论文

https://arxiv.org/abs/2303.03192

未来会读一下相关文献，梳理一下原理

## demo中以下片段调用了PhySo相关功能，具有不可替代性

#库的调用

monitoring；benchmark；physo；

#奖励函数
  #奖励函数是一个名为 SquashedNRMSE 的函数，它来自 physo.physym.reward 模块

                 "reward_function"     : physo.physym.reward.SquashedNRMSE,

  #奖励相关的配置 Reward related

    'risk_factor'      : 0.05,
    'rewards_computer' : physo.physym.reward.make_RewardsComputer (**reward_config),

#运行过程，拟合调用physo
  #调用 physo.fit 方法进行拟合，得到奖励和候选答案

  rewards, candidates = physo.fit (X, y, run_config,
                                stop_reward = 0.9999, 
                                stop_after_n_epochs = 5)
