import torch
import numpy as np

def safe_cross_entropy(p, logq, dim=-1):
    """
    安全交叉熵计算，避免对数中的零值导致的数值不稳定。

    参数:
    - p: torch.tensor, 概率分布。
    - logq: torch.tensor, 对数概率分布。
    - dim: int, 沿哪个维度进行计算，默认为-1。

    返回:
    - cross_entropy: torch.tensor, 交叉熵值。
    """
    safe_logq = torch.where(p == 0, torch.ones_like(logq), logq)
    return -torch.sum(p * safe_logq, dim=dim)


def loss_func(logits_train, ideal_probs_train, R_train, baseline, lengths, gamma_decay, entropy_weight):
    """
    用于强化符号程序的损失函数。

    参数:
    - logits_train: torch.tensor, 形状为 (max_time_step, n_train, n_choices), RNN生成的概率。
    - ideal_probs_train: torch.tensor, 形状为 (max_time_step, n_train, n_choices), 理想的概率分布。
    - R_train: torch.tensor, 形状为 (n_train,), 程序的奖励。
    - baseline: float, 基线值，从奖励中减去。
    - lengths: torch.tensor, 形状为 (n_train,), 程序的有效长度（不包括占位符填充）。
    - gamma_decay: float, 沿程序长度的指数衰减因子。
    - entropy_weight: float, 熵损失部分的权重。

    返回:
    - loss: float, 损失值。
    """

    # 获取形状
    (max_time_step, n_train, n_choices,) = ideal_probs_train.shape

    # ----- 长度掩码 -----
    # 长度掩码（避免超出符号函数范围的学习）

    mask_length_np = np.tile(np.arange(0, max_time_step), (n_train, 1))  # (n_train, max_time_step,)
    mask_length_np = mask_length_np.astype(int) < np.tile(lengths, (max_time_step, 1)).transpose()
    mask_length_np = mask_length_np.transpose().astype(float)  # (max_time_step, n_train,)
    mask_length = torch.tensor(mask_length_np, requires_grad=False)  # (max_time_step, n_train,)

    # ----- 熵掩码 -----
    # 熵掩码（沿序列维度加权不同）

    entropy_gamma_decay = np.array([gamma_decay ** t for t in range(max_time_step)])  # (max_time_step,)
    entropy_decay_mask_np = np.tile(entropy_gamma_decay, (n_train, 1)).transpose() * mask_length_np  # (max_time_step, n_train,)
    entropy_decay_mask = torch.tensor(entropy_decay_mask_np, requires_grad=False)  # (max_time_step, n_train,)

    # ----- 损失：梯度策略 -----

    # 在动作维度上归一化概率和对数概率
    probs = torch.nn.functional.softmax(logits_train, dim=2)  # (max_time_step, n_train, n_choices,)
    logprobs = torch.nn.functional.log_softmax(logits_train, dim=2)  # (max_time_step, n_train, n_choices,)

    # 在动作维度上求和
    neglogp_per_step = safe_cross_entropy(ideal_probs_train, logprobs, dim=2)  # (max_time_step, n_train,)
    # 在序列维度上求和
    neglogp = torch.sum(neglogp_per_step * mask_length, dim=0)  # (n_train,)

    # 在批次的训练样本上求平均
    loss_gp = torch.mean((R_train - baseline) * neglogp)

    # ----- 损失：熵 -----

    # 在动作维度上求和
    entropy_per_step = safe_cross_entropy(probs, logprobs, dim=2)  # (max_time_step, n_train,)
    # 在序列维度上求和
    entropy = torch.sum(entropy_per_step * entropy_decay_mask, dim=0)  # (n_train,)

    loss_entropy = -entropy_weight * torch.mean(entropy)

    # ----- 总损失 -----
    loss = loss_gp + loss_entropy

    return loss