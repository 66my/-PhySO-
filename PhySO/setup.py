import setuptools
import os

# 定义软件包的版本号
VERSION = '1.1.0a'

# 定义软件包的描述信息
DESCRIPTION = 'Physical Symbolic Optimization'

# 读取依赖项列表（这里被注释掉了）
# REQUIRED = open("requirements_pip.txt").read().splitlines()

# 定义可选依赖项（这里被注释掉了）
# EXTRAS = {
#     "display": [
#         "pygraphviz",
#     ],
# }
# EXTRAS['all'] = list(set([item for group in EXTRAS.values() for item in group]))

# 定义包含 Feynman 数据集 CSV 文件的路径模式
PATH_FEYNMAN_CSVs = os.path.join("benchmark", "FeynmanDataset", "*.csv")

# 定义包含 Class 数据集 CSV 文件的路径模式
PATH_CLASS_CSVs   = os.path.join("benchmark", "ClassDataset"  , "*.csv")

# 定义软件包中包含的数据文件
package_data = [PATH_FEYNMAN_CSVs, PATH_CLASS_CSVs]

# 使用 setuptools.setup 函数配置并安装软件包
setuptools.setup(
    name             = 'physo',  # 软件包的名称
    version          = VERSION,  # 软件包的版本号
    description      = DESCRIPTION,  # 软件包的描述信息
    author           = 'Wassim Tenachi',  # 软件包的作者
    author_email     = 'w.tenachi@gmail.com',  # 作者的电子邮件地址
    license          = 'MIT',  # 软件包的许可证类型
    packages         = setuptools.find_packages(),  # 自动查找并包含所有 Python 包
    package_data     = {"physo": package_data},  # 指定包含的数据文件
    # install_requires = REQUIRED,  # 安装软件包所需的依赖项（这里被注释掉了）
    # extras_require   = EXTRAS,  # 可选的额外依赖项（这里被注释掉了）
)