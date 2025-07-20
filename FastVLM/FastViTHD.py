import torch
import torch.nn as nn
import torch.nn.functional as F


# class FastViTHD(nn.Module):
#     """
#
#
#     """
#     def __init__(self,
#                  depths=[2, 12, 24, 4, 2],  # 各阶段深度
#                  dims=[96, 192, 384, 768, 1536],  # 嵌入维度
#                  pool_layers=[2, 5, 8, 11],  # 池化层位置（Stage 2-3）
#                  downscale_ratio=4):  # 令牌压缩率
#         super().__init__()
#         # Stage 1: 大核卷积下采样
#         self.stem = nn.Sequential(
#             nn.Conv2d(3, dims[0], kernel_size=7, stride=2, padding=3),
#             nn.BatchNorm2d(dims[0]),
#             nn.GELU()
#         )
#
#         # Stages 2-5: 混合块序列
#         self.stages = nn.ModuleList()
#         for i in range(4):  # 遍历 Stage 2-5
#             stage = []
#             # 添加 RepMixer 或 Transformer 块
#             for j in range(depths[i + 1]):
#                 if i < 2:  # Stage 2-3 用 RepMixer（卷积主导）
#                     block = RepMixerBlock(dims[i + 1])
#                 else:  # Stage 4-5 用 Transformer（注意力主导）
#                     block = TransformerBlock(dims[i + 1], num_heads=8)
#                 stage.append(block)
#
#                 # 在指定位置插入池化层（压缩令牌）
#                 if (i == 1 or i == 2) and j in pool_layers:
#                     stage.append(MultiScalePooling(downscale_ratio))
#
#             self.stages.append(nn.Sequential(*stage))
#
#         # Stage 5 输出投影至LLM输入维度
#         self.proj = nn.Linear(dims[4], dims[4] * 2)  # 1536 -> 3072
#
#
# class RepMixerBlock(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         # 训练时结构（含跳连）
#         self.conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
#         self.bn = nn.BatchNorm2d(dim)
#         self.act = nn.GELU()
#         self.skip = nn.Identity()  # 训练时保留跳连
#
#     def forward(self, x):
#         return self.act(self.bn(self.conv(x))) + self.skip(x)
#
#     def reparametrize(self):  # 推理时合并卷积+BN+跳连
#         fused_conv = fuse_conv_bn(self.conv, self.bn)
#         # 跳连转换为1x1卷积并合并
#         if not isinstance(self.skip, nn.Identity):
#             skip_conv = nn.Conv2d(fused_conv.in_channels, fused_conv.out_channels, 1)
#             fused_conv.weight.data += skip_conv.weight.data
#         return fused_conv
#
#
# class MultiScalePooling(nn.Module):
#     def __init__(self, ratio=4):
#         super().__init__()
#         self.ratio = ratio
#         self.pool = nn.AdaptiveAvgPool2d((None, None))  # 自适应目标尺寸
#
#     def forward(self, x):
#         _, _, H, W = x.shape
#         target_h, target_w = H // self.ratio, W // self.ratio
#         return self.pool(x, output_size=(target_h, target_w))  # 动态压缩
#
#
# class TransformerBlock(nn.Module):
#     def __init__(self, dim, num_heads):
#         super().__init__()
#         self.norm1 = nn.LayerNorm(dim)
#         self.attn = nn.MultiheadAttention(dim, num_heads)
#         self.norm2 = nn.LayerNorm(dim)
#         self.mlp = nn.Sequential(
#             nn.Linear(dim, dim * 4),
#             nn.GELU(),
#             nn.Linear(dim * 4, dim)
#         )
#
#     def forward(self, x):
#         # 转换维度: [B, C, H, W] -> [B, H*W, C]
#         x_flat = x.flatten(2).permute(0, 2, 1)
#
#         # 自注意力 + MLP
#         attn_out, _ = self.attn(self.norm1(x_flat), self.norm1(x_flat), self.norm1(x_flat))
#         x_flat = x_flat + attn_out
#         x_flat = x_flat + self.mlp(self.norm2(x_flat))
#
#         # 恢复维度: [B, H*W, C] -> [B, C, H, W]
#         return x_flat.permute(0, 2, 1).view(x.shape)

class MultiScalePooling(nn.Module):
    def __init__(self, ratio=4):
        super().__init__()
        self.ratio = ratio
        self.pool = nn.AdaptiveAvgPool2d((None, None))  # 自适应目标尺寸

    def forward(self, x):
        _, _, H, W = x.shape
        target_h, target_w = H // self.ratio, W // self.ratio
        return self.pool(x, output_size=(target_h, target_w))  # 动态压缩


class RepMixerBlock(nn.Module):
    def __init__(self, dim, kernel_size=7):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim, bias=False)
        self.bn = nn.BatchNorm2d(dim)

    def forward(self, x):
        identity = x
        x = self.bn(x)
        x = self.dwconv(x)
        return identity + x

    def reparameterize(self):
        # 推理时重参数化
        fused_conv = nn.utils.fuse_conv_bn_eval(self.dwconv, self.bn)
        return fused_conv


class Attention(nn.Module):
    def __init__(self, dim, num_heads, kv_heads, max_seq_len=4096):
        super(Attention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.kv_heads = kv_heads
        self.head_dim = self.dim // self.num_heads

        assert dim % self.num_heads == 0

        self.wq = nn.Linear(self.dim, self.num_heads * self.head_dim, bias=True)
        self.wk = nn.Linear(self.dim, self.head_dim * self.kv_heads, bias=True)
        self.wv = nn.Linear(self.dim, self.head_dim * self.kv_heads, bias=True)
        self.wo = nn.Linear(self.num_heads * self.head_dim, self.dim, bias=False)

        self.register_buffer(
            'freqs',
            self._precompute_freqs(max_seq_len, self.head_dim)
        )

    def _precompute_freqs(self, seq_len, head_dim):
        theta = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        freqs = torch.outer(torch.arange(seq_len), theta)
        return torch.polar(torch.ones_like(freqs), freqs)  # 复数形式e^(iθ)

    def _apply_rotary(self, x, freqs):
        # 拆分实部和虚部 [[13]]
        x_complex = torch.view_as_complex(
            x.float().reshape(*x.shape[:-1], -1, 2)
        ).transpose(1, 2)
        # 旋转操作 (q * e^(iθ)) [[13]]
        x_rotated = (x_complex * freqs).transpose(1, 2)
        # 恢复为实数形式
        return torch.view_as_real(x_rotated).flatten(-2)

    def forward(self, x):
        batch_size, seq, dim = x.shape

        # [B,S,heads*head_dim]->[B,S,heads,head_dim]
        q = self.wq(x).view(batch_size, seq, self.num_heads, self.head_dim)
        # [B,S,kv_heads*head_dim]->[B,S,kv_heads,head_dim]
        k = self.wk(x).view(batch_size, seq, self.kv_heads, self.head_dim)
        v = self.wv(x).view(batch_size, seq, self.kv_heads, self.head_dim)

        q = self._apply_rotary(q, self.freqs[:seq])
        k = self._apply_rotary(k, self.freqs[:seq])

        # 广播
        repeat = self.num_heads // self.kv_heads
        k = k.repeat_interleave(repeat, dim=2)
        v = v.repeat_interleave(repeat, dim=2)

        # 计算分数
        att_score = torch.einsum("bqhd,bkhd -> bqhk", q, k) / (self.head_dim ** 0.5)
        att_weight = F.softmax(att_score, dim=-1)
        output = torch.einsum("bqhk,bkhd -> bqhd", att_weight, v)

        # 合并头
        output = output.reshape(batch_size, seq, -1)
        return self.wo(output)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, kv_heads):
        super(TransformerBlock, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.kv_heads = kv_heads
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, kv_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x):
        b, c, h, w = x.shape
        # [B,C,H,W]->[B,H*W,C]
        x = x.flatten(2).permute(0, 2, 1)
        print(x.shape)
        att = self.attn(self.norm1(x))
        x = x + att
        x = x + self.mlp(self.norm2(x))
        print(x.shape)
        return x.permute(0, 2, 1).view(b, c, h, w)


class FastViTHD(nn.Module):
    def __init__(self,
                 depths=[2, 12, 24, 4, 2],  # 各阶段深度
                 dims=[96, 192, 384, 768, 1536],  # 嵌入维度
                 pool_layers=[2, 5, 8, 11],  # 池化层位置（Stage 2-3）
                 downscale_ratio=4):  # 令牌压缩率
        super().__init__()
        # 核卷积下采样
        self.stem = nn.Sequential(
            nn.Conv2d(3, dims[0], kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(dims[0]),
            nn.GELU()
        )

        self.stages = nn.ModuleList()
        for i in range(4):
            stage = []
            for j in range(depths[i + 1]):
                if i < 2:  # Stage 2-3 用 RepMixer（卷积主导）
                    block = RepMixerBlock(dims[i + 1])
                else:  # Stage 4-5 用 Transformer（注意力主导）
                    block = TransformerBlock(dims[i + 1], num_heads=8, kv_heads=4)
                stage.append(block)

                # 在指定位置插入池化层
                if (i == 1 or i == 2) and j in pool_layers:
                    stage.append(MultiScalePooling(downscale_ratio))

            self.stages.append(nn.Sequential(*stage))

        # 输出投影至
        self.proj = nn.Linear(dims[4], dims[4] * 2)

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.proj(x)
        return x



