import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_

class NeighborhoodAttention2D(nn.Module):
    def __init__(self, dim, kernel_size, num_heads, attn_drop=0., proj_drop=0., dilation=None):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.kernel_size = kernel_size
        self.dilation = dilation or 1
        self.window_size = self.kernel_size * self.dilation

        self.qkv = nn.Linear(dim, dim * 3)
        # 수정된 부분: rpb의 차원을 (num_heads, 1, 1, kernel_size, kernel_size)로 변경
        self.rpb = nn.Parameter(torch.zeros(num_heads, 1, 1, kernel_size, kernel_size))
        trunc_normal_(self.rpb, std=.02, mean=0., a=-2., b=2.)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, C, H, W = x.shape
        pad_l = pad_t = pad_r = pad_b = 0
        if H < self.window_size or W < self.window_size:
            pad_l = pad_t = 0
            pad_r = max(0, self.window_size - W)
            pad_b = max(0, self.window_size - H)
            x = F.pad(x, (pad_l, pad_r, pad_t, pad_b))
            _, _, H, W = x.shape

        x = x.permute(0, 2, 3, 1)  # B, H, W, C
        qkv = self.qkv(x).reshape(B, H, W, 3, self.num_heads, self.head_dim).permute(3, 0, 4, 1, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale

        # Implement local self-attention
        attn = torch.zeros(B, self.num_heads, H, W, self.kernel_size**2, device=x.device, dtype=x.dtype)
        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                q_shifted = torch.roll(q, shifts=(-i*self.dilation, -j*self.dilation), dims=(2, 3))
                k_shifted = torch.roll(k, shifts=(-i*self.dilation, -j*self.dilation), dims=(2, 3))
                attn_ij = (q_shifted * k_shifted).sum(dim=-1)
                attn[:, :, :, :, i*self.kernel_size+j] = attn_ij + self.rpb[:, :, :, i, j]

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Apply attention to values
        out = torch.zeros_like(v)
        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                v_shifted = torch.roll(v, shifts=(-i*self.dilation, -j*self.dilation), dims=(2, 3))
                out += attn[:, :, :, :, i*self.kernel_size+j].unsqueeze(-1) * v_shifted

        out = out.permute(0, 2, 3, 1, 4).reshape(B, H, W, C)
        if pad_r or pad_b:
            out = out[:, :H-pad_b, :W-pad_r, :]

        out = self.proj_drop(self.proj(out))
        return out.permute(0, 3, 1, 2)

# Usage example:
if __name__ == "__main__":
    batch_size, height, width = 1, 32, 32
    dim, num_heads, kernel_size = 64, 8, 3
    x = torch.randn(batch_size, dim, height, width)
    na2d = NeighborhoodAttention2D(dim, kernel_size, num_heads)
    output = na2d(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")