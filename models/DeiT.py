import torch, math
from torch import nn
from torchvision.models import regnet_y_16gf

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchinfo import summary

from transformers import DeiTForImageClassification, DeiTConfig

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, embed_size=768, img_size=512):
        super().__init__()
        num_patches = (img_size // patch_size) ** 2
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, embed_size, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e')
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_size))
        self.distill_token = nn.Parameter(torch.randn(1, 1, embed_size))
        self.positions = nn.Parameter(torch.randn(num_patches + 2, embed_size))  # Add +2 for distill and cls tokens

    def forward(self, x):
        b, _, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        distill_tokens = repeat(self.distill_token, '() n e -> b n e', b=b)

        # Concatenate cls_token and distill_token with the input
        x = torch.cat([cls_tokens, distill_tokens, x], dim=1)

        x += self.positions
        return x

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value, mask=None):
        d_k = query.size(-1)  # embed_size / num_heads
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, torch.finfo(scores.dtype).min)

        attn_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, value)
        return output, attn_weights

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads=12):
        super().__init__()
        assert embed_size % num_heads == 0, "embed_size must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads

        self.query_linear = nn.Linear(embed_size, embed_size)
        self.key_linear = nn.Linear(embed_size, embed_size)
        self.value_linear = nn.Linear(embed_size, embed_size)

        # 멀티헤드 결과 결합
        self.out_linear = nn.Linear(embed_size, embed_size)

        self.attention = ScaledDotProductAttention()

    def forward(self, query):
        batch_size = query.size(0)

        query = self.query_linear(query)
        key = self.key_linear(query)
        value = self.value_linear(query)

        query = query.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn_output, attn_weights = self.attention(query, key, value)

        # Concatenate attention outputs from all heads
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(batch_size, -1, self.num_heads * self.head_dim)
        output = self.out_linear(attn_output)

        return output

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        res = x
        x = self.fn(x)
        x += res
        return x

class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion = 4):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Linear(expansion * emb_size, emb_size),
        )

class TransformerEncoderBlock(nn.Sequential):
    def __init__(self, emb_size= 768, expansion = 4):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size),
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=expansion),
            )
            ))

class TransformerEncoder(nn.Sequential):
    def __init__(self, depth = 12, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])

class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size = 768, n_classes = 4):
        super().__init__(
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes)
        )

class DeiT(nn.Module):
    def __init__(self, in_channels = 3, patch_size = 16, emb_size = 768, img_size = 512, depth = 12, n_classes = 4):
        super().__init__()
        self.patch_embedding = PatchEmbedding(in_channels, patch_size, emb_size, img_size)
        self.transformer_encoder = TransformerEncoder(depth, emb_size=emb_size)
        self.classification_head = ClassificationHead(emb_size, n_classes)
        self.distillation_head = ClassificationHead(emb_size, n_classes)

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.transformer_encoder(x)

        cls_token_final = x[:, 0]  # 첫 번째 토큰 (classification)
        distill_token_final = x[:, 1]  # 두 번째 토큰 (distillation)

        cls_output = self.classification_head(cls_token_final)
        distill_output = self.distillation_head(distill_token_final)

        return cls_output, distill_output

if __name__ == "__main__":
    model = DeiT()
    '''
    config = DeiTConfig.from_pretrained('facebook/deit-base-distilled-patch16-224')
    # Modify the input channels in the configuration
    config.num_channels = 1
    config.num_labels=10

    # Load the DeiT model with the updated configuration and ignore size mismatch
    model = DeiTForImageClassification.from_pretrained(
        'facebook/deit-base-distilled-patch16-224',
        config=config,
        ignore_mismatched_sizes=True  # Ignore size mismatch to handle the modified input channels
    )

    # Modify the model to accept grayscale input (1 channel)
    model.deit.embeddings.patch_embeddings.projection = nn.Conv2d(
        in_channels=1,  # For grayscale input
        out_channels=768,  # Should match the original number of out channels
        kernel_size=16,
        stride=16
    )

    # Modify the classifier head to have 10 output classes
    model.classifier = nn.Linear(model.config.hidden_size, 10)
    '''
    summary(model.cuda(), (1, 3, 512, 512))