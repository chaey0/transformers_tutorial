import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
from torchinfo import summary
from torchvision.models import swin_b

class PatchEmbedding(nn.Module):
    def __init__(self, patch_size=4, C=128):
        super().__init__()
        self.linear_embedding = nn.Conv2d(3, C, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.linear_embedding(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        return x


class PatchMerging(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.linear = nn.Linear(4 * C, 2 * C)
        self.layer_norm = nn.LayerNorm(2 * C)

    def forward(self, x):
        height = width = int(math.sqrt(x.shape[1]) / 2)
        x = rearrange(x, 'b (h s1 w s2) c -> b (h w) (s2 s1 c)', s1=2, s2=2, h=height, w=width)
        return self.layer_norm(self.linear(x))


class RelativePositionBias(nn.Module):
    def __init__(self, window_size):
        super().__init__()
        self.window_size = window_size
        self.relative_position_bias_table = nn.Parameter(torch.randn((2 * window_size - 1) ** 2, 1))

        # Create relative position index
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, window_size, window_size
        coords_flatten = torch.flatten(coords, 1)  # 2, window_size^2
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, window_size^2, window_size^2
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # window_size^2, window_size^2, 2
        relative_coords[:, :, 0] += self.window_size - 1  # to start from 0
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)  # window_size^2, window_size^2
        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self, num_heads):
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size ** 2, self.window_size ** 2, -1)  # window_size^2, window_size^2, num_heads
        return relative_position_bias.permute(2, 0, 1).contiguous()  # num_heads, window_size^2, window_size^2


class ShiftedWindowMSA(nn.Module):
    def __init__(self, embed_dim, num_heads, window_size=8, mask=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.mask = mask
        self.proj1 = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj2 = nn.Linear(embed_dim, embed_dim)
        self.relative_position_bias = RelativePositionBias(window_size)

    def forward(self, x):
        h_dim = self.embed_dim // self.num_heads
        height = width = int(math.sqrt(x.shape[1]))
        x = self.proj1(x)
        x = rearrange(x, 'b (h w) (c K) -> b h w c K', K=3, h=height, w=width)

        if self.mask:
            x = torch.roll(x, (-self.window_size // 2, -self.window_size // 2), dims=(1, 2))

        x = rearrange(x, 'b (h m1) (w m2) (H E) K -> b H h w (m1 m2) E K', H=self.num_heads, m1=self.window_size, m2=self.window_size)
        Q, K, V = x.chunk(3, dim=6)
        Q, K, V = Q.squeeze(-1), K.squeeze(-1), V.squeeze(-1)
        att_scores = (Q @ K.transpose(4, 5)) / math.sqrt(h_dim)

        # Apply relative position bias
        relative_position_bias = self.relative_position_bias(self.num_heads)
        att_scores = att_scores + relative_position_bias.unsqueeze(0)

        if self.mask:
            device = x.device
            row_mask = torch.zeros((self.window_size ** 2, self.window_size ** 2)).to(device)
            row_mask[-self.window_size * (self.window_size // 2):, 0:-self.window_size * (self.window_size // 2)] = float('-inf')
            row_mask[0:-self.window_size * (self.window_size // 2), -self.window_size * (self.window_size // 2):] = float('-inf')
            column_mask = rearrange(row_mask, '(r w1) (c w2) -> (w1 r) (w2 c)', w1=self.window_size, w2=self.window_size).to(device)
            att_scores[:, :, -1, :] += row_mask
            att_scores[:, :, :, -1] += column_mask

        att = F.softmax(att_scores, dim=-1) @ V
        x = rearrange(att, 'b H h w (m1 m2) E -> b (h m1) (w m2) (H E)', m1=self.window_size, m2=self.window_size)

        if self.mask:
            x = torch.roll(x, (self.window_size // 2, self.window_size // 2), (1, 2))

        x = rearrange(x, 'b h w c -> b (h w) c')
        return self.proj2(x)

class SwinEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, window_size, mask):
        super().__init__()
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.MSA = ShiftedWindowMSA(embed_dim=embed_dim, num_heads=num_heads, window_size=window_size, mask=mask)
        self.MLP1 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )

    def forward(self, x):
        res1 = self.MSA(self.layer_norm(x)) + x
        x = self.layer_norm(res1)
        x = self.MLP1(x)
        return x + res1

class AlternatingEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, window_size=8):
        super().__init__()
        self.WSA = SwinEncoderBlock(embed_dim=embed_dim, num_heads=num_heads, window_size=window_size, mask=False)
        self.SWSA = SwinEncoderBlock(embed_dim=embed_dim, num_heads=num_heads, window_size=window_size, mask=True)

    def forward(self, x):
        return self.SWSA(self.WSA(x))


class ClassificationHead(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.norm(x)
        x = x.mean(dim=1)  # Global Average Pooling
        return self.head(x)


class SwinTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.Embedding = PatchEmbedding()

        # Swin-B has larger embedding dimensions and more layers
        self.PatchMerge1 = PatchMerging(128)
        self.PatchMerge2 = PatchMerging(256)
        self.PatchMerge3 = PatchMerging(512)

        # Stage configurations: [2, 2, 18, 2]
        self.Stage1 = nn.Sequential(*[AlternatingEncoderBlock(128, 4) for _ in range(1)])
        self.Stage2 = nn.Sequential(*[AlternatingEncoderBlock(256, 8) for _ in range(1)])
        self.Stage3 = nn.Sequential(*[AlternatingEncoderBlock(512, 16) for _ in range(9)])
        self.Stage4 = nn.Sequential(*[AlternatingEncoderBlock(1024, 32) for _ in range(1)])

        self.ClassificationHead = ClassificationHead(1024, num_classes=4)

    def forward(self, x):
        x = self.Embedding(x)
        x = self.PatchMerge1(self.Stage1(x))
        x = self.PatchMerge2(self.Stage2(x))
        x = self.PatchMerge3(self.Stage3(x))
        x = self.Stage4(x)

        return self.ClassificationHead(x)


if __name__ == "__main__":
    device = torch.device('cuda:5')
    #model = SwinTransformer().to(device)

    model = swin_b()

    model.head = nn.Linear(model.head.in_features, 4)
    model.to(device)

    summary(model, (1, 3, 512,512))
