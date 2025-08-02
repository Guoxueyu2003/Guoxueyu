import torch
import torch.nn as nn


class RepresentationSpace(nn.Module):
    def __init__(self, visual_dim, text_dim, num_tokens=8, rep_dim=512):
        super().__init__()
        # 从高斯分布初始化可学习表示空间
        self.R = nn.Parameter(torch.randn(num_tokens, rep_dim))
        # 视觉投影层
        self.proj_v = nn.Linear(rep_dim, visual_dim)
        # 文本投影层
        self.proj_t = nn.Linear(rep_dim, text_dim)

    def forward(self):
        R_v = self.proj_v(self.R)
        R_t = self.proj_t(self.R)
        return R_v, R_t


class MMRL(nn.Module):
    def __init__(self, clip_model, start_layer, visual_dim, text_dim, rep_dim, clip_layers):
        super().__init__()
        # 冻结的CLIP模型
        self.clip = clip_model
        self.rep_space = RepresentationSpace(
            visual_dim=visual_dim,
            text_dim=text_dim,
            rep_dim=rep_dim
        )
        # 注入起始层
        self.start_layer = start_layer
        self.clip_layers = clip_layers

    def forward(self, image, text):
        # 原始CLIP特征提取
        with torch.no_grad():
            v_feats = self.clip.visual_encoder(image)
            t_feats = self.clip.text_encoder(text)
        # 获取原始模型的表示令牌
        R_v, R_t = self.rep_space()
        # 高层注入视觉令牌
        for i in range(self.start_layer, len(self.clip_layers)):
            if i == self.start_layer:
                v_input = torch.cat([v_feats[:, :1], R_v.unsqueeze(0), v_feats[:, 1:]], dim=1)
            v_feats = self.clip.layers[i](v_input)

        # 高层注入文本令牌
        for i in range(self.start_layer, len(self.clip_layers)):
            if i == self.start_layer:
                t_input = torch.cat([t_feats[:, :1], R_t.unsqueeze(0), t_feats[:, 1:]], dim=1)
            t_feats = self.clip.layers[i](t_input)

        return v_feats, t_feats
