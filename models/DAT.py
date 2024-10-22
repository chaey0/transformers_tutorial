import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange

from torchinfo import summary

def count_parameters_in_block(model):
    return sum(p.numel() for p in block.parameters() if p.requires_grad)

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # If input is in (B, N, C) format, reshape it to (B, C, H, W)
        if len(x.shape) == 3:
            B, N, C = x.shape
            H = W = int(math.sqrt(N))
            x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)

        x = self.proj(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        return x

class RelativeEmbeddings(nn.Module):
    def __init__(self, window_size=8):
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

class ClassificationHead(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super().__init__()
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.linear = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.layer_norm(x)
        x = x.mean(dim=1)
        x = self.linear(x)
        return x

class Scale(nn.Module):
    def __init__(self, scale=2):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return x * self.scale


class LocalAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, window_size=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.proj1 = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj2 = nn.Linear(embed_dim, embed_dim)
        self.embeddings = RelativeEmbeddings()

    def forward(self, x):
        h_dim = self.embed_dim / self.num_heads
        height = width = int(math.sqrt(x.shape[1]))
        x = self.proj1(x)
        x = rearrange(x, 'b (h w) (c K) -> b h w c K', K=3, h=height, w=width)

        x = rearrange(x, 'b (h m1) (w m2) (H E) K -> b H h w (m1 m2) E K', H=self.num_heads, m1=self.window_size,
                      m2=self.window_size)

        Q, K, V = x.chunk(3, dim=6)
        Q, K, V = Q.squeeze(-1), K.squeeze(-1), V.squeeze(-1)
        att_scores = (Q @ K.transpose(4, 5)) / math.sqrt(h_dim)
        att_scores = self.embeddings(att_scores)

        att = F.softmax(att_scores, dim=-1) @ V
        x = rearrange(att, 'b H h w (m1 m2) E -> b (h m1) (w m2) (H E)', m1=self.window_size, m2=self.window_size)

        x = rearrange(x, 'b h w c -> b (h w) c')

        x = self.proj2(x)
        return x


class ShiftedWindowMSA(nn.Module):
    def __init__(self, embed_dim, num_heads, window_size=8, mask=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.mask = mask
        self.proj1 = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj2 = nn.Linear(embed_dim, embed_dim)

        self.embeddings = RelativeEmbeddings()

    def forward(self, x):
        h_dim = self.embed_dim / self.num_heads
        height = width = int(math.sqrt(x.shape[1]))
        x = self.proj1(x)
        x = rearrange(x, 'b (h w) (c K) -> b h w c K', K=3, h=height, w=width)

        if self.mask:
            x = torch.roll(x, (-self.window_size // 2, -self.window_size // 2), dims=(1, 2))

        x = rearrange(x, 'b (h m1) (w m2) (H E) K -> b H h w (m1 m2) E K', H=self.num_heads, m1=self.window_size,
                      m2=self.window_size)
        Q, K, V = x.chunk(3, dim=6)
        Q, K, V = Q.squeeze(-1), K.squeeze(-1), V.squeeze(-1)
        att_scores = (Q @ K.transpose(4, 5)) / math.sqrt(h_dim)
        att_scores = self.embeddings(att_scores)

        if self.mask:
            # row_mask and column_mask should have the same shape as att_scores
            row_mask = torch.zeros((self.window_size ** 2, self.window_size ** 2), device=x.device)
            row_mask[-self.window_size * (self.window_size // 2):, 0:-self.window_size * (self.window_size // 2)] = float('-inf')
            row_mask[0:-self.window_size * (self.window_size // 2), -self.window_size * (self.window_size // 2):] = float('-inf')

            # Expand row_mask to match att_scores shape: [B, num_heads, window_size^2, window_size^2]
            row_mask = row_mask.unsqueeze(0).unsqueeze(0)

            att_scores = att_scores + row_mask  # Apply the mask

        att = F.softmax(att_scores, dim=-1) @ V
        x = rearrange(att, 'b H h w (m1 m2) E -> b (h m1) (w m2) (H E)', m1=self.window_size, m2=self.window_size)

        if self.mask:
            x = torch.roll(x, (self.window_size // 2, self.window_size // 2), (1, 2))

        x = rearrange(x, 'b h w c -> b (h w) c')
        return self.proj2(x)


class SwinEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, window_size=8, mask=True):
        super().__init__()
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.local_attention = LocalAttention(embed_dim=embed_dim, num_heads=num_heads, window_size=window_size)
        self.SW_MSA = ShiftedWindowMSA(embed_dim=embed_dim, num_heads=num_heads, window_size=window_size, mask=mask)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim)
        )

        self.mlp2 = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim)
        )

    def forward(self, x):
        x = x + self.local_attention(self.layer_norm(x))
        x = x + self.mlp(x)

        x = x + self.SW_MSA(self.layer_norm(x))
        x = x + self.mlp2(x)
        return x

class DeformableAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, window_size=8, offset_scale=2, offset_kernel_size=5, groups=1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = embed_dim // num_heads
        self.groups = groups
        self.n_group_channels = self.embed_dim // self.groups
        self.n_group_heads = self.num_heads // self.groups
        self.offset_scale = offset_scale

        self.offset_network = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=offset_kernel_size, stride=1, padding=offset_kernel_size // 2,
                      groups=embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, 2 * num_heads, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Tanh(),
            Scale(offset_scale)
        )

        self.q_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1, stride=1, padding=0)
        self.k_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1, stride=1, padding=0)
        self.v_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1, stride=1, padding=0)
        self.out_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1, stride=1, padding=0)

        self.embeddings = RelativeEmbeddings(window_size=window_size)

    def forward(self, x):
        B, N, C = x.shape
        H = W = int(math.sqrt(N))

        # Reshape input for Conv2d
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)

        Q = self.q_proj(x)  # shape (B, C, H, W)
        K = self.k_proj(x)  # shape (B, C, H, W)
        V = self.v_proj(x)  # shape (B, C, H, W)

        # Compute offsets
        offsets = self.offset_network(x)  # [B, 2*num_heads, H, W]
        offsets = rearrange(offsets, 'b (n c) h w -> b n h w c', n=self.num_heads, c=2)

        grid_y, grid_x = torch.meshgrid(torch.linspace(-1, 1, H, device=x.device),
                                        torch.linspace(-1, 1, W, device=x.device),
                                        indexing='ij')
        grid = torch.stack((grid_x, grid_y), dim=-1)
        grid = grid.unsqueeze(0).unsqueeze(1).repeat(B, self.num_heads, 1, 1, 1)

        sampling_grid = grid + offsets
        sampling_grid = torch.clamp(sampling_grid, -1, 1)

        # Flatten batch and number of heads for grid_sample
        sampling_grid = rearrange(sampling_grid, 'b n h w c -> (b n) h w c')

        # Reshape K and V to match expected grid_sample format
        K = rearrange(K, 'b (n d) h w -> (b n) d h w', n=self.num_heads, d=self.head_dim)
        V = rearrange(V, 'b (n d) h w -> (b n) d h w', n=self.num_heads, d=self.head_dim)

        # Sample K and V using grid_sample
        K_sampled = F.grid_sample(K, sampling_grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        V_sampled = F.grid_sample(V, sampling_grid, mode='bilinear', padding_mode='zeros', align_corners=False)

        # Reshape K_sampled and V_sampled back to original dimensions
        K_sampled = rearrange(K_sampled, '(b n) d h w -> b n (h w) d', b=B, n=self.num_heads)
        V_sampled = rearrange(V_sampled, '(b n) d h w -> b n (h w) d', b=B, n=self.num_heads)

        # Reshape Q to match the required format
        Q = rearrange(Q, 'b (n d) h w -> b n (h w) d', n=self.num_heads, d=self.head_dim)

        # Compute attention scores
        attn_scores = torch.einsum('b h i d, b h j d -> b h i j', Q, K_sampled)
        attn_scores = attn_scores / math.sqrt(self.head_dim)

        # Add positional bias
        relative_position_bias = self.embeddings(self.num_heads)
        relative_position_bias = relative_position_bias.unsqueeze(0).expand(B, -1, -1, -1)
        relative_position_bias = F.interpolate(relative_position_bias, size=(H*W, H*W), mode='bicubic')
        attn_scores = attn_scores + relative_position_bias

        attn_probs = F.softmax(attn_scores, dim=-1)

        # Apply attention to V
        out = torch.einsum('b h i j, b h j d -> b h i d', attn_probs, V_sampled)
        out = rearrange(out, 'b h (p q) d -> b (h d) p q', p=H, q=W)

        # Output projection using Conv2d
        out = self.out_proj(out)
        out = rearrange(out, 'b c h w -> b (h w) c')

        return out

class DeformableEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, window_size=8, groups=1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.local_attention = LocalAttention(embed_dim=embed_dim, num_heads=num_heads, window_size=window_size)
        self.deformable_attention = DeformableAttention(embed_dim=embed_dim, num_heads=num_heads,
                                                        window_size=window_size, groups=groups)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim)
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim)
        )

    def forward(self, x):
        x = x + self.local_attention(self.layer_norm(x))
        x = x + self.mlp(x)

        x = x + self.deformable_attention(self.layer_norm(x))
        x = x + self.mlp2(x)
        return x

class DAT(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        # Stage 1
        self.patch_embedding1 = PatchEmbedding(in_channels=3, out_channels=128, kernel_size=4, stride=4)
        self.stage1 = nn.Sequential(*[SwinEncoderBlock(embed_dim=128, num_heads=4) for _ in range(1)])

        # Stage 2
        self.patch_embedding2 = PatchEmbedding(in_channels=128, out_channels=256, kernel_size=2, stride=2)
        self.stage2 = nn.Sequential(*[SwinEncoderBlock(embed_dim=256, num_heads=8)for _ in range(1)])

        # Stage 3
        self.patch_embedding3 = PatchEmbedding(256, 512, kernel_size=2, stride=2)
        self.stage3 = nn.Sequential(*[DeformableEncoderBlock(embed_dim=512, num_heads=16, window_size=8, groups=4) for _ in range(9)])


        # Stage 4
        self.patch_embedding4 = PatchEmbedding(in_channels=512, out_channels=1024, kernel_size=2, stride=2)
        self.stage4 = nn.Sequential(*[DeformableEncoderBlock(embed_dim=1024, num_heads=32, groups=8)for _ in range(1)])

        self.classification_head = ClassificationHead(embed_dim=1024, num_classes=num_classes)

    def forward(self, x):
        x = self.patch_embedding1(x)
        x = self.stage1(x)

        x = self.patch_embedding2(x)
        x = self.stage2(x)

        x = self.patch_embedding3(x)
        x = self.stage3(x)

        x = self.patch_embedding4(x)
        x = self.stage4(x)

        return self.classification_head(x)

if __name__ == "__main__":
    device = torch.device('cuda:4')
    model = DAT().to(device)
    '''
    for name, block in model.named_modules():
        print(f"{name}: {count_parameters_in_block(block)} parameters")

    for name, block in model.named_children():
        print(f"{name}: {count_parameters_in_block(block)} parameters")
    '''
    summary(model, (1, 3, 512, 512))

