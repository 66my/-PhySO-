import torch
import numpy as np

class Cell(torch.nn.Module):
    """
    自定义RNN单元，用于符号表达式生成。

    属性:
    ----------
    input_size  : int
        观察向量的大小。
    n_layers    : int
        堆叠的RNN层数。
    hidden_size : int
        RNN单元状态中的隐藏特征数量。
    output_size : int
        输出特征的数量，即令牌库中的选择数量。

    input_dense       : torch.nn
        输入全连接层。
    stacked_cells     : torch.nn.ModuleList of torch.nn
        堆叠的RNN单元。
    output_dense      : torch.nn
        输出全连接层。
    output_activation : function
        输出激活函数。

    is_lobotomized : bool
        是否将神经网络的输出概率替换为随机数。

    logTemperature : torch.tensor
        退火参数。

    方法:
    -------
    forward (input_tensor, states)
        返回分类对数的RNN单元调用。
    get_zeros_initial_state (batch_size)
        返回包含零的单元状态。
    count_parameters()
        返回可训练参数的数量。

    示例:
    -------
    # RNN初始化 ---------
    input_size  = 3*7 + 33
    output_size = 7
    hidden_size = 32
    n_layers    = 1
    batch_size  = 1024
    time_steps  = 30

    initial_states = torch.zeros(n_layers, 2, batch_size, hidden_size)
    initial_obs    = torch.zeros(batch_size, input_size)

    RNN_CELL = Cell(input_size  = input_size,
                    output_size = output_size,
                    hidden_size = hidden_size,
                    n_layers    = n_layers)

    print(RNN_CELL)
    print("n_params = %i"%(RNN_CELL.count_parameters()))
    # RNN运行 --------------
    observations = initial_obs
    states       = initial_states
    outputs      = []
    for _ in range (time_steps):
        output, states = RNN_CELL(input_tensor = observations,
                                  states = states)
        observations   = observations
        outputs.append(output)
    outputs = torch.stack(outputs)
    print("outputs shape = ", outputs.shape)
    """

    def __init__(self,
                 input_size,
                 output_size,
                 hidden_size,
                 n_layers=1,
                 input_dense=None,
                 stacked_cells=None,
                 output_dense=None,
                 is_lobotomized=False):
        super().__init__()
        # --------- 输入全连接层 ---------
        self.input_size = input_size
        self.hidden_size = hidden_size
        if input_dense is None:
            input_dense = torch.nn.Linear(self.input_size, self.hidden_size)
        self.input_dense = input_dense
        # --------- 堆叠的RNN单元 ---------
        self.n_layers = n_layers
        if stacked_cells is None:
            stacked_cells = torch.nn.ModuleList([torch.nn.LSTMCell(input_size=self.hidden_size,
                                                                   hidden_size=self.hidden_size)
                                                 for _ in range(self.n_layers)])
        self.stacked_cells = stacked_cells
        # --------- 输出全连接层 ---------
        self.output_size = output_size
        if output_dense is None:
            output_dense = torch.nn.Linear(self.hidden_size, self.output_size)
        self.output_dense = output_dense
        self.output_activation = lambda x: -torch.nn.functional.relu(x)  # 将输出映射到log(p)
        # lambda x: torch.nn.functional.softmax(x, dim=1)
        # torch.sigmoid
        # --------- 退火参数 ---------
        self.logTemperature = torch.nn.Parameter(1.54 * torch.ones(1), requires_grad=True)
        # --------- 失能化 ---------
        self.is_lobotomized = is_lobotomized

    def get_zeros_initial_state(self, batch_size):
        """
        返回包含零的初始状态。

        参数:
        - batch_size: int, 批量大小。

        返回:
        - zeros_initial_state: torch.tensor, 形状为 (n_layers, 2, batch_size, hidden_size) 的初始状态。
        """
        zeros_initial_state = torch.zeros(self.n_layers, 2, batch_size, self.hidden_size, requires_grad=False)
        return zeros_initial_state

    def forward(self,
                input_tensor,  # (batch_size, input_size)
                states):       # (n_layers, 2, batch_size, hidden_size)
        """
        RNN单元的前向传播过程。

        参数:
        - input_tensor: torch.tensor, 形状为 (batch_size, input_size) 的输入张量。
        - states: torch.tensor, 形状为 (n_layers, 2, batch_size, hidden_size) 的初始状态。

        返回:
        - res: torch.tensor, 形状为 (batch_size, output_size) 的输出对数。
        - out_states: torch.tensor, 形状为 (n_layers, 2, batch_size, hidden_size) 的新状态。
        """
        # --------- 输入全连接层 ---------
        hx = self.input_dense(input_tensor)  # (batch_size, hidden_size)
        # 层归一化 + 激活函数
        # --------- 堆叠的RNN单元 ---------
        new_states = []  # 新状态的堆叠RNN
        for i in range(self.n_layers):
            hx, cx = self.stacked_cells[i](hx,  # (batch_size, hidden_size)
                                           (states[i, 0, :, :],  # (batch_size, hidden_size)
                                            states[i, 1, :, :]))  # (batch_size, hidden_size)
            new_states.append(torch.stack([hx, cx]))
        # --------- 输出全连接层 ---------
        # 来自神经网络的概率
        res = self.output_dense(hx) + self.logTemperature  # (batch_size, output_size)
        # 应用激活函数
        res = self.output_activation(res)  # (batch_size, output_size)
        # 来自随机数生成器的概率
        if self.is_lobotomized:
            res = torch.log(torch.rand(res.shape))
        out_states = torch.stack(new_states)  # (n_layers, 2, batch_size, hidden_size)
        # --------------- 返回 ---------------
        return res, out_states  # (batch_size, output_size), (n_layers, 2, batch_size, hidden_size)

    def count_parameters(self):
        """
        计算并返回模型的可训练参数数量。

        返回:
        - n_params: int, 可训练参数的数量。
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        n_params = sum([np.prod(p.size()) for p in model_parameters])
        return n_params