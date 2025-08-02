import torch
import torch.nn as nn
from mambapy.mamba import Mamba, MambaConfig
from mambapy.pscan import pscan  # 并行扫描算法

#  混合多模态空间
class HMSS(nn.Module):
    def __init__(self, d_model,n_layers,d_state, d_conv, expand_factor):
        """

        :param d_model: 模型隐藏层维度
        :param n_layers: mamba堆叠数量
        :param d_state: SSM状态扩展因子
        :param d_conv:  局部卷积核宽度
        :param expand_factor: 块扩展系数
        """
        super(HMSS, self).__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.n_layers = n_layers
        self.expand_factor = expand_factor
        self.mamba_config = MambaConfig(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand_factor=expand_factor,
            n_layers=n_layers,
        )
        self.mamba_fwd = Mamba(config=self.mamba_config)
        self.mamba_bwd = Mamba(config=self.mamba_config)
        self.layer_norm = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_model)
        self.linear2 = nn.Linear(d_model, d_model)
        self.linear3 = nn.Linear(d_model, d_model)
        self.down_linear = nn.Linear(d_model * 2, d_model)
        self.conv = nn.Conv1d(d_model, d_model, 3, 1, 1)

    def forward(self, x):
        x_layer = self.layer_norm(x)
        x_linear1 = self.linear1(x_layer)
        x_linear2 = self.linear2(x_layer)
        # [B,L,D]->[B,D,L]
        x_linear1 = x_linear1.permute(0, 2, 1)
        x_conv = self.conv(x_linear1)
        # [B,D,L]->[B,L,D]
        x_conv = x_conv.permute(0, 2, 1)
        # 顺序进入Mamba
        x_fwd = self.mamba_fwd(x_conv)
        # 逆序进入Mamba
        x_reversed = torch.flip(x_conv, dims=[1])
        x_bwd = self.mamba_bwd(x_reversed)
        # 融合对齐下采样
        x_bwd = torch.flip(x_bwd, dims=[1])
        x_fwd_bwd = torch.cat([x_fwd, x_bwd], dim=-1)
        x_fwd_bwd = self.down_linear(x_fwd_bwd)
        x_fwd_bwd = self.layer_norm(x_fwd_bwd)

        x_fwd_bwd = x_fwd_bwd * x_linear2
        out = self.linear3(x_fwd_bwd)

        return out + x

# 选择局部性增强
class SLE(nn.Module):
    def __init__(self, dim):
        super(SLE, self).__init__()
        self.layer_norm = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.down_linear = nn.Linear(dim * 2, dim)
        self.conv = nn.Conv1d(dim, dim, 3, 1, 1)

    def forward(self, x):
        x_layer = self.layer_norm(x)
        x_linear1 = self.linear1(x_layer)
        # [B,L,D]->[B,D,L]
        x_conv = x_linear1.permute(0, 2, 1)
        x_conv = self.conv(x_conv)
        x_conv = x_conv.permute(0, 2, 1)

        x_linear2 = self.linear2(x_layer)

        x_linear_conv = torch.cat([x_linear1, x_conv], dim=-1)
        x_down = self.down_linear(x_linear_conv)

        return x_down + x_linear2 + x

# 时空演变融合模块
class TEMF(nn.Module):
    def __init__(self, dim, n_layers, d_state, d_conv, expand_factor):
        super(TEMF, self).__init__()
        self.hmss = HMSS(
            d_model=dim,
            n_layers=n_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand_factor=expand_factor
        )
        self.sle = SLE(dim=dim)

    def forward(self, x):
        x = self.hmss(x)
        x = self.sle(x)
        return x





# config = MambaConfig(
#     d_model=256,  # 模型隐藏层输入维度
#     n_layers=4,  # Mamba层堆叠数量
#     d_state=16,  # SSM状态扩展因子
#     d_conv=4,  # 局部卷积核宽度
#     expand_factor=2,  # 块扩展系数
# )
# model = HMSS(
#     d_model=256,
#     n_layers=4,
#     d_state=16,
#     d_conv=4,
# )
# batch, length, dim = 8, 512, 256
# x = torch.randn(batch, length, dim)
# y = model(x)
# print(y)
x = torch.randn(8, 512, 256)
sle = SLE(256)
print(sle(x))
