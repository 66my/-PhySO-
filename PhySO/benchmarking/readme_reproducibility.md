在这里，我们提供了重现论文中提出的基准测试实验结果的说明。
请参阅文档中的[Benchmarks](https://physo.readthedocs.io/en/latest/r_benchmarks.html)部分，了解基准测试中的挑战（例如，如果您想对您自己的方法进行基准测试）。

## Feynman 基准测试

Feynman 基准测试的目的是评估符号回归系统，特别是为科学发现构建的方法。
这些方法能够从潜在的噪声数据中生成紧凑、预测性强且可解释的表达式。

参见 [[Udrescu 2019]](https://arxiv.org/abs/1905.11481)，该基准测试的介绍，[[La Cava 2021]](https://arxiv.org/abs/2107.14351)，对该基准测试的正式化，以及 [[Tenachi 2023]](https://arxiv.org/abs/2303.03192)，关于 `physo` 在此基准测试上的评估。

### 运行单个 Feynman 问题

运行 `physo` 对 Feynman 基准测试中的第 `i` 个挑战（`i` ∈ {0, 1, ..., 119}），使用试验种子 `t` ∈ ℕ，并采用噪声水平 `n` ∈ [0,1]。

```bash
python feynman_run.py --equation i --trial t --noise n
```

例如，运行问题 5，试验种子为 1，噪声水平为 0.0，启用并行模式并使用 4 个 CPU：

```bash
python feynman_run.py --equation 5 --trial 1 --noise 0.0 --parallel_mode 1 --ncpus 4
```

### 创建 HPC 作业文件

创建一个作业文件以在噪声水平为 0.0 的情况下运行所有 Feynman 问题：

```bash
python feynman_make_run_file.py --noise 0.0
```

### 分析结果

分析结果文件夹：

```bash
python feynman_results_analysis.py --path [结果文件夹]
```

### 结果

![logo](https://raw.githubusercontent.com/WassimTenachi/PhySO/main/benchmarking/FeynmanBenchmark/results/feynman_results.png)

## 类别基准测试

类别基准测试的目的是评估类别符号回归系统，即：自动找到一个单一的解析函数形式，能够准确拟合多个数据集的方法——每个数据集都受其自己（可能是独特的）一组拟合参数的控制。

参见 [[Tenachi 2024]](https://arxiv.org/abs/2312.01816)，我们在其中介绍了第一个类别 SR 方法的基准测试，并评估了 `physo` 在此基准测试上的表现。

### 运行单个类别问题

运行 `physo` 对类别基准测试中的第 `i` 个挑战（`i` ∈ {0, 1, ..., 7}），使用试验种子 `t` ∈ ℕ，采用噪声水平 `n` ∈ [0,1]，并利用 `Nr` ∈ ℕ 次实现。

```bash
python classbench_run.py --equation i --trial t --noise n --n_reals Nr
```

例如，运行问题 7，试验种子为 3，噪声水平为 0.001，并进行 10 次实现。

```bash
python classbench_run.py --equation 7 --trial 3 --noise 0.001 --n_reals 10
```

### 创建 HPC 作业文件

创建一个作业文件以运行所有类别问题。

```bash
python classbench_make_run_file.py
```

### 分析结果

分析结果文件夹：

```bash
python classbench_results_analysis.py --path [结果文件夹]
```

### 结果

![logo](https://raw.githubusercontent.com/WassimTenachi/PhySO/main/benchmarking/ClassBenchmark/results/class_results.png)

## 类别基准测试（MW 奖励）

这个奖励的目的是评估类别符号回归系统在现实世界问题上的表现，即从观测到的恒星流的位置和速度中发现银河系暗物质分布。

![logo](https://raw.githubusercontent.com/WassimTenachi/PhySO/main/docs/assets/demo_streams_plot.png)

参见 [[Tenachi 2024]](https://arxiv.org/abs/2312.01816)，我们在其中介绍了这个奖励，并评估了 `physo` 在此基准测试上的表现。

### 运行单个配置

运行 `physo` 对银河系恒星流问题，使用试验种子 `t` ∈ ℕ，采用噪声水平 `n` ∈ [0,1]，并利用 `fr` ∈ [0,1] 比例的实现。

```bash
python MW_streams_run.py --noise n --trial t --frac_real fr
```

### 创建 HPC 作业文件

创建一个作业文件以运行所有类别问题。

```bash
python MW_streams_make_run_file.py
```

### 分析结果

分析结果文件夹：

```bash
python MW_streams_results_analysis.py --path [结果文件夹]
```

### 结果

![logo](https://raw.githubusercontent.com/WassimTenachi/PhySO/main/demos/class_sr/demo_milky_way_streams/results/MW_benchmark.png)

---