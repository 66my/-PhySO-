import torch
import numpy as np
import time

# 内部导入
from . import loss

def learner(
    model,  # 模型，一个可微的RNN单元
    optimizer,  # 优化器
    n_epochs,  # 训练轮数
    batch_reseter,  # 重置批次的函数
    risk_factor,  # 精英程序的比例
    gamma_decay,  # 指数衰减因子
    entropy_weight,  # 熵损失权重
    verbose=True,  # 是否打印日志
    stop_reward=1.,  # 提前停止的奖励阈值
    stop_after_n_epochs=50,  # 达到提前停止条件后继续训练的轮数
    max_n_evaluations=None,  # 最大评估次数
    run_logger=None,  # 自定义运行日志记录器
    run_visualiser=None,  # 自定义运行可视化器
):
    """
    通过强化最佳候选者来训练模型生成满足奖励的符号程序。

    参数:
    - model: torch.nn.Module, 可微的RNN单元。
    - optimizer: torch.optim, 优化器。
    - n_epochs: int, 训练轮数。
    - batch_reseter: callable, 返回一个新的空批次的函数。
    - risk_factor: float, 精英程序的比例（0到1之间）。
    - gamma_decay: float, 沿程序长度的幂律权重。
    - entropy_weight: float, 熵损失部分的权重。
    - verbose: int, 是否打印日志。0或False表示不打印，1或True表示打印学习时间，大于1表示打印每轮进度。
    - stop_reward: float, 提前停止的奖励阈值，默认为1。
    - stop_after_n_epochs: int, 达到提前停止条件后继续训练的轮数。
    - max_n_evaluations: int 或 None, 允许的最大唯一表达式评估次数（用于基准测试）。
    - run_logger: object 或 None, 自定义运行日志记录器，必须有 `log` 方法。
    - run_visualiser: object 或 None, 自定义运行可视化器，必须有 `visualise` 方法。

    返回:
    - hall_of_fame_R: 历史最佳程序的奖励值列表。
    - hall_of_fame: 历史最佳程序列表。
    """

    t000 = time.perf_counter()

    # 基本日志
    overall_max_R_history = []  # 整体最大奖励的历史记录
    hall_of_fame = []  # 历史最佳程序
    n_evaluated = 0  # 已评估的表达式数量

    for epoch in range(n_epochs):
        if verbose > 1:
            print(f"Epoch {epoch}/{n_epochs}")

        # 初始化
        batch = batch_reseter()  # 重置新的批次
        batch_size = batch.batch_size  # 批次大小
        max_time_step = batch.max_time_step  # 最大时间步长

        # 初始RNN单元输入
        states = model.get_zeros_initial_state(batch_size)  # (n_layers, 2, batch_size, hidden_size)

        # 优化器重置
        optimizer.zero_grad()

        # 候选者
        logits = []
        actions = []

        # 要保留的精英候选者的数量
        n_keep = int(risk_factor * batch_size)

        # RNN运行
        for i in range(max_time_step):
            # 观察
            observations = torch.tensor(batch.get_obs().astype(np.float32), requires_grad=False)  # (batch_size, obs_size)

            # 模型
            output, states = model(input_tensor=observations, states=states)  # (batch_size, output_size), (n_layers, 2, batch_size, hidden_size)

            # 获取第i个动作的原始概率分布
            outlogit = output  # (batch_size, output_size)

            # 先验
            prior_array = batch.prior().astype(np.float32)  # (batch_size, output_size)

            # 保护0，确保总有东西可以采样
            epsilon = 0  # 1e-14  # 1e0 * np.finfo(np.float32).eps
            prior_array[prior_array == 0] = epsilon

            # 转换为对数
            prior = torch.tensor(prior_array, requires_grad=False)  # (batch_size, output_size)
            logprior = torch.log(prior)  # (batch_size, output_size)

            # 采样
            logit = outlogit + logprior  # (batch_size, output_size)
            action = torch.multinomial(torch.exp(logit), num_samples=1)[:, 0]  # (batch_size,)

            # 动作
            logits.append(logit)
            actions.append(action)

            # 通知嵌入层新的动作
            batch.programs.append(action.detach().cpu().numpy())

        # 候选者
        logits = torch.stack(logits, dim=0)  # (max_time_step, batch_size, n_choices)
        actions = torch.stack(actions, dim=0)  # (max_time_step, batch_size)

        # 程序作为NumPy数组用于黑盒奖励计算
        actions_array = actions.detach().cpu().numpy()  # (max_time_step, batch_size)

        # 奖励
        R = batch.get_rewards()  # (batch_size,)

        # 最佳候选者
        keep = R.argsort()[::-1][0:n_keep].copy()  # (n_keep,)
        notkept = R.argsort()[::-1][n_keep:].copy()  # (batch_size - n_keep,)

        # 训练批次：黑盒部分（NumPy）
        actions_array_train = actions_array[:, keep]  # (max_time_step, n_keep)
        ideal_probs_array_train = np.eye(batch.n_choices)[actions_array_train]  # (max_time_step, n_keep, n_choices)

        R_train = torch.tensor(R[keep], requires_grad=False)  # (n_keep,)
        R_lim = R_train.min()

        # 训练批次：可微部分（Torch）
        ideal_probs_train = torch.tensor(
            ideal_probs_array_train.astype(np.float32),
            requires_grad=False,
        )  # (max_time_step, n_keep, n_choices)

        logits_train = logits[:, keep]  # (max_time_step, n_keep, n_choices)

        # 损失
        lengths = batch.programs.n_lengths[keep]  # (n_keep,)
        baseline = R_lim

        loss_val = loss.loss_func(
            logits_train=logits_train,
            ideal_probs_train=ideal_probs_train,
            R_train=R_train,
            baseline=baseline,
            lengths=lengths,
            gamma_decay=gamma_decay,
            entropy_weight=entropy_weight,
        )

        # 反向传播
        if not model.is_lobotomized:
            loss_val.backward()
            optimizer.step()

        # 日志记录
        if epoch == 0:
            overall_max_R_history = [R.max()]
            hall_of_fame = [batch.programs.get_prog(R.argmax())]
        elif epoch > 0:
            if R.max() > np.max(overall_max_R_history):
                overall_max_R_history.append(R.max())
                hall_of_fame.append(batch.programs.get_prog(R.argmax()))
            else:
                overall_max_R_history.append(overall_max_R_history[-1])

        if run_logger is not None:
            run_logger.log(
                epoch=epoch,
                batch=batch,
                model=model,
                rewards=R,
                keep=keep,
                notkept=notkept,
                loss_val=loss_val
            )

        # 可视化
        if run_visualiser is not None:
            run_visualiser.visualise(run_logger=run_logger, batch=batch)

        # 提前停止
        early_stop_reward_eps = 2 * np.finfo(np.float32).eps

        if (stop_reward - overall_max_R_history[-1]) <= early_stop_reward_eps:
            if stop_after_n_epochs == 0:
                try:
                    run_visualiser.save_visualisation()
                    run_visualiser.save_data()
                    run_visualiser.save_pareto_data()
                    run_visualiser.save_pareto_fig()
                except:
                    print("无法在提前停止前保存最后一个图表和数据。")
                break
            stop_after_n_epochs -= 1

        # 最大评估次数停止
        n_evaluated += (R > 0.).sum()

        if (max_n_evaluations is not None) and (n_evaluated + batch_size > max_n_evaluations):
            try:
                run_visualiser.save_visualisation()
                run_visualiser.save_data()
                run_visualiser.save_pareto_data()
                run_visualiser.save_pareto_fig()
            except:
                print("无法在达到最大评估次数限制前保存最后一个图表和数据。")
            break

    t111 = time.perf_counter()
    if verbose:
        print(f"  -> Time = {t111 - t000} s")

    hall_of_fame_R = np.array(overall_max_R_history)
    return hall_of_fame_R, hall_of_fame