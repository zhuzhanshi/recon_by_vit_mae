import math
from typing import Tuple

import torch
from torch import nn


def _get_1d_sincos_pos_embed(embed_dim: int, length: int, device=None) -> torch.Tensor:
    position = torch.arange(length, dtype=torch.float32, device=device).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, embed_dim, 2, dtype=torch.float32, device=device)
        * (-math.log(10000.0) / embed_dim)
    )
    pe = torch.zeros((length, embed_dim), dtype=torch.float32, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


def get_2d_sincos_pos_embed(embed_dim: int, grid_h: int, grid_w: int, device=None) -> torch.Tensor:
    assert embed_dim % 2 == 0
    pe_h = _get_1d_sincos_pos_embed(embed_dim // 2, grid_h, device=device)
    pe_w = _get_1d_sincos_pos_embed(embed_dim // 2, grid_w, device=device)
    pe = torch.cat(
        [
            pe_h[:, None, :].expand(grid_h, grid_w, -1),
            pe_w[None, :, :].expand(grid_h, grid_w, -1),
        ],
        dim=-1,
    )
    return pe.view(grid_h * grid_w, embed_dim)


def patchify_gray(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    # x: (B, 1, H, W)
    b, _, h, w = x.shape
    p = patch_size
    assert h % p == 0 and w % p == 0
    x = x.reshape(b, 1, h // p, p, w // p, p)
    x = x.permute(0, 2, 4, 3, 5, 1)
    return x.reshape(b, (h // p) * (w // p), p * p)


def unpatchify_gray(x: torch.Tensor, patch_size: int, h: int, w: int) -> torch.Tensor:
    # x: (B, N, p*p)
    b, n, pp = x.shape
    p = patch_size
    gh = h // p
    gw = w // p
    x = x.reshape(b, gh, gw, p, p)
    x = x.permute(0, 1, 3, 2, 4)
    return x.reshape(b, 1, h, w)


class PatchEmbedConv(nn.Module):
    def __init__(self, img_size: int, patch_size: int, in_chans: int, embed_dim: int):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.grid_h = img_size // patch_size
        self.grid_w = img_size // patch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class PatchEmbedLinear(nn.Module):
    def __init__(self, patch_size: int, in_chans: int, embed_dim: int):
        super().__init__()
        self.proj = nn.Linear(patch_size * patch_size * in_chans, embed_dim)

    def forward(self, x: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
        b, c, h, w = x.shape
        p = int((h // grid_h))
        patches = x.reshape(b, c, grid_h, p, grid_w, p)
        patches = patches.permute(0, 2, 4, 3, 5, 1).reshape(b, grid_h * grid_w, p * p * c)
        return self.proj(patches)


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj(x)


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MAE(nn.Module):
    def __init__(
        self,
        img_size: int = 512,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 256,
        depth: int = 4,
        num_heads: int = 8,
        decoder_embed_dim: int = 128,
        decoder_depth: int = 2,
        decoder_num_heads: int = 4,
        mask_ratio: float = 0.75,
        include_cls_token: bool = False,
        use_conv_patch_embed: bool = False,
        mlp_ratio: float = 4.0,
        use_vit_blocks: bool = False,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.grid_h = img_size // patch_size
        self.grid_w = img_size // patch_size
        self.num_patches = self.grid_h * self.grid_w
        self.include_cls_token = include_cls_token
        self.use_vit_blocks = use_vit_blocks
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads

        if use_conv_patch_embed:
            self.patch_embed = PatchEmbedConv(img_size, patch_size, in_chans, embed_dim)
        else:
            self.patch_embed = PatchEmbedLinear(patch_size, in_chans, embed_dim)

        if include_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            pe = get_2d_sincos_pos_embed(embed_dim, self.grid_h, self.grid_w)
            pe = torch.cat([torch.zeros(1, embed_dim), pe], dim=0).unsqueeze(0)
        else:
            pe = get_2d_sincos_pos_embed(embed_dim, self.grid_h, self.grid_w).unsqueeze(0)
        self.pos_embed = nn.Parameter(pe, requires_grad=False)

        if use_vit_blocks:
            self.blocks = nn.ModuleList(
                [Block(embed_dim, num_heads, mlp_ratio) for _ in range(depth)]
            )
            self.norm = nn.LayerNorm(embed_dim)
        else:
            enc_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4, batch_first=True
            )
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(
            get_2d_sincos_pos_embed(decoder_embed_dim, self.grid_h, self.grid_w).unsqueeze(0),
            requires_grad=False,
        )

        dec_layer = nn.TransformerEncoderLayer(
            d_model=decoder_embed_dim,
            nhead=decoder_num_heads,
            dim_feedforward=decoder_embed_dim * 4,
            batch_first=True,
        )
        self.decoder = nn.TransformerEncoder(dec_layer, num_layers=decoder_depth)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size * patch_size)

        nn.init.normal_(self.mask_token, std=0.02)

    @classmethod
    def from_preset(cls, preset: str, **kwargs) -> "MAE":
        if preset == "vitb":
            return cls(
                embed_dim=768,
                depth=12,
                num_heads=12,
                decoder_embed_dim=512,
                decoder_depth=8,
                decoder_num_heads=16,
                include_cls_token=True,
                use_conv_patch_embed=True,
                mlp_ratio=4.0,
                use_vit_blocks=True,
                **kwargs,
            )
        # tiny (default)
        return cls(**kwargs)

    def random_masking(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.include_cls_token:
            cls = x[:, :1]
            x = x[:, 1:]
        b, n, _ = x.shape
        len_keep = int(n * (1 - self.mask_ratio))

        noise = torch.rand(b, n, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_keep = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, x.shape[2]))

        mask = torch.ones((b, n), device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        if self.include_cls_token:
            x_keep = torch.cat([cls, x_keep], dim=1)
        return x_keep, mask, ids_restore

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, C, H, W)
        b, c, h, w = x.shape
        p = self.patch_size
        assert h == self.img_size and w == self.img_size

        if isinstance(self.patch_embed, PatchEmbedConv):
            tokens = self.patch_embed(x)
        else:
            tokens = self.patch_embed(x, self.grid_h, self.grid_w)

        if self.include_cls_token:
            cls = self.cls_token.expand(b, -1, -1)
            cls = cls + self.pos_embed[:, :1, :].to(tokens.device)
            tokens = tokens + self.pos_embed[:, 1:, :].to(tokens.device)
            tokens = torch.cat([cls, tokens], dim=1)
        else:
            tokens = tokens + self.pos_embed.to(tokens.device)

        x_keep, mask, ids_restore = self.random_masking(tokens)

        if self.use_vit_blocks:
            enc = x_keep
            for blk in self.blocks:
                enc = blk(enc)
            enc = self.norm(enc)
        else:
            enc = self.encoder(x_keep)

        if self.include_cls_token:
            enc = enc[:, 1:, :]

        dec_tokens = self.decoder_embed(enc)
        mask_tokens = self.mask_token.repeat(b, self.num_patches - dec_tokens.shape[1], 1)
        dec_ = torch.cat([dec_tokens, mask_tokens], dim=1)
        dec_ = torch.gather(
            dec_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, dec_.shape[2])
        )
        dec_ = dec_ + self.decoder_pos_embed.to(dec_.device)
        dec_out = self.decoder(dec_)
        pred = self.decoder_pred(dec_out)
        return pred, mask

    def forward_loss(self, x: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        target = patchify_gray(x[:, 0:1], self.patch_size)
        loss = (pred - target).abs().mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum().clamp(min=1.0)
        return loss
