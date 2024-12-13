from . import physym
from . import learn
from . import task
from . import config
from . import benchmark

# 将重要的接口函数暴露在根级别，方便用户调用
fit = task.fit.fit  # 拟合模型的函数
SR = task.sr.SR  # 符号回归的主要类或函数
ClassSR = task.class_sr.ClassSR  # 分类符号回归的主要类或函数

# 提供给用户的日志加载工具
read_pareto_csv = benchmark.utils.read_logs.read_pareto_csv  # 从 CSV 文件读取帕累托前沿的日志
read_pareto_pkl = learn.monitoring.read_pareto_pkl  # 从 Pickle 文件读取帕累托前沿的日志