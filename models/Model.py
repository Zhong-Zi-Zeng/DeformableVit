from .deformable_transformer import build_deformable_transformer
from .backbone import build_backbone
from util.misc import NestedTensor
import torch.nn as nn
import torch


class DeformableVit(nn.Module):
    def __init__(self, backbone, transformer, cls_num, num_feature_levels):
        super().__init__()

        self.backbone = backbone
        self.transformer = transformer
        self.cls_num = cls_num
        self.num_feature_levels = num_feature_levels

        hidden_dim = self.transformer.d_model

        # 建立input_proj 負責將每個feature project跟transformer d_m 一樣的維度
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])

        # 最後的分類層
        self.class_embed = nn.Linear(hidden_dim, self.cls_num)

    def forward(self, inputs):
        # features [(B, C1, H1, W1), (B, C2, H2, W2), ...]
        # pos [(1, d_m, H1, W1), (1, d_m, H2, W2), ...]
        features, pos = self.backbone(inputs)

        # 把features映射到跟transformer d_m 一樣的維度
        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None

        # 送入到transformer中
        # memory (B,  H1 * W1 + H2 * W2 + .., d_m)
        memory = self.transformer(srcs, masks, pos)

        # 將第一個序列送入分類層
        output = self.class_embed(memory[:, 0])

        return output

def build(args):
    transformer = build_deformable_transformer(args)
    backbone = build_backbone(args)
    model = DeformableVit(backbone=backbone,
                          transformer=transformer,
                          cls_num=args.cls_num,
                          num_feature_levels=args.num_feature_levels)
    # model.to('cuda')
    # inputs = torch.rand((1, 3, 200, 200), device='cuda')
    # mask = torch.zeros((1, 200, 200), device='cuda')
    # sample = NestedTensor(inputs, mask)
    # model(sample)

    return model
