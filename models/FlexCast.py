import torch
from torch import nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayerWithRoPE


class RevIN(nn.Module):
    def __init__(self, dim=-1, std_min=1e-5, use_sinh=False):
        super().__init__()
        self.dim = dim
        self.std_min = std_min
        self.use_sinh = use_sinh

    def fit_transform(self, x, mask=None):
        device_type = "cuda" if x.is_cuda else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            self._get_statistics(x, mask)
            return self.transform(x)

    def transform(self, x):
        device_type = "cuda" if x.is_cuda else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            x = (x - self.mean) / self.std
            if self.use_sinh:
                x = torch.asinh(x)
            return x

    def inverse_transform(self, x):
        device_type = "cuda" if x.is_cuda else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            if self.use_sinh:
                x = torch.sinh(x)
            if x.ndim != self.mean.ndim:
                x = x * self.std.unsqueeze(1) + self.mean.unsqueeze(1)
            else:
                x = x * self.std + self.mean
            return x

    def _get_statistics(self, x, mask=None):
        if mask is None:
            self.mean = x.mean(dim=self.dim, keepdim=True)
            std = x.std(dim=self.dim, keepdim=True)
            self.std = torch.where(std > self.std_min, std, torch.ones_like(std))
        else:
            mask = mask.bool()
            unmask = (~mask).float()
            count = unmask.sum(dim=self.dim, keepdim=True).clamp(min=1)
            x_mean = (x * unmask).sum(dim=self.dim, keepdim=True) / count
            x_std = (((x - x_mean) * unmask) ** 2).sum(dim=self.dim, keepdim=True) / count
            x_std = x_std.sqrt()
            x_std = torch.where(x_std > self.std_min, x_std, torch.ones_like(x_std))
            self.mean = x_mean
            self.std = x_std


class ResidualBlock(nn.Module):
    def __init__(self, d_in: int, d_out: int, d_hidden: int):
        super().__init__()
        self.layer1 = nn.Linear(d_in, d_hidden)
        self.layer2 = nn.Linear(d_hidden, d_out)
        self.residual = nn.Linear(d_in, d_out)
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer2(self.activation(self.layer1(x))) + self.residual(x)


class Model(nn.Module):
    def __init__(self, configs, patch_len=None, stride=None):
        super().__init__()
        self.patch_len = patch_len if patch_len is not None else getattr(configs, 'patch_len', 16)
        self.stride    = stride if stride is not None else getattr(configs, 'stride', self.patch_len)

        self.num_quantile = getattr(configs, 'num_quantile', 9)
        d_model = configs.d_model
        d_ff    = getattr(configs, 'd_ff', d_model * 4)

        self.in_layer  = ResidualBlock(self.patch_len * 2, d_model, d_model)
        self.out_layer = ResidualBlock(d_model, self.patch_len * (self.num_quantile + 1), d_model)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayerWithRoPE(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False), d_model, configs.n_heads),
                    d_model, d_ff,
                    dropout=configs.dropout,
                    activation=getattr(configs, 'activation', 'gelu'),
                    norm_first=True,
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(d_model),
        )

        self.norm_fn = RevIN(dim=-1, std_min=1e-5, use_sinh=True)

    def _build_attention_mask(self, pad_patch):
        batch_size, n_patch, _ = pad_patch.shape
        pad_ratio = pad_patch.float().mean(dim=-1)
        fully_padded = pad_ratio > 0.9
        attn_mask = torch.zeros(batch_size, 1, 1, n_patch,
                                device=pad_patch.device, dtype=torch.float32)
        attn_mask.masked_fill_(fully_padded[:, None, None, :], float('-inf'))
        return attn_mask

    def _decode_quantiles(self, raw, T):
        flat_batch, P, _ = raw.shape
        Q = self.num_quantile
        d = self.patch_len

        raw = raw.reshape(flat_batch, P, Q + 1, d)
        base    = raw[:, :, 0:1, :]
        offsets = raw[:, :, 1:, :]

        q = base + torch.cumsum(F.softplus(offsets) / Q, dim=2)
        q = q.permute(0, 2, 1, 3)

        unpatched = q.reshape(flat_batch * Q, P * d)

        return unpatched.reshape(flat_batch, Q, T)

    def imputation(self, x_enc, mask, miss_mask=None, pad_mask=None, return_raw=True):
        B, T, C = x_enc.shape
        d  = self.patch_len

        x_flat = x_enc.permute(0, 2, 1).reshape(B * C, T)
        mask_flat = mask.permute(0, 2, 1).reshape(B * C, T)

        pred_mask = (mask_flat == 0)
        if miss_mask is None and pad_mask is None:
            miss_flat = torch.zeros_like(pred_mask)
            pad_flat = torch.isnan(x_flat)
        else:
            miss_flat = (
                miss_mask.permute(0, 2, 1).reshape(B * C, T).bool()
                if miss_mask is not None else torch.zeros_like(pred_mask)
            )
            pad_flat = (
                pad_mask.permute(0, 2, 1).reshape(B * C, T).bool()
                if pad_mask is not None else torch.zeros_like(pred_mask)
            )
        ts_mask = pred_mask | miss_flat | pad_flat

        safe_x = torch.nan_to_num(x_flat, nan=0.0)
        x_target = self.norm_fn.fit_transform(safe_x, mask=ts_mask)

        x_input = torch.where(ts_mask, torch.zeros_like(x_target), x_target)

        n_patch = T // d
        T_aligned = n_patch * d
        x_aligned    = x_input[:, :T_aligned]
        mask_aligned = ts_mask[:, :T_aligned].float()
        pad_aligned  = pad_flat[:, :T_aligned].float()

        x_patch    = x_aligned.reshape(B * C, n_patch, d)
        mask_patch = mask_aligned.reshape(B * C, n_patch, d)
        pad_patch  = pad_aligned.reshape(B * C, n_patch, d)

        attn_mask = self._build_attention_mask(pad_patch)

        x = self.in_layer(torch.cat([x_patch, 1 - mask_patch], dim=-1))

        x, _ = self.encoder(x, attn_mask=attn_mask)

        raw = self.out_layer(x)

        q_pred = self._decode_quantiles(raw, T_aligned)

        if T_aligned < T:
            pad = T - T_aligned
            q_pred = F.pad(q_pred, (pad, 0), value=0.0)

        Q = self.num_quantile

        if not return_raw:
            q_asinh = q_pred.reshape(B, C, Q, T).permute(0, 3, 1, 2)
            x_tgt = x_target.reshape(B, C, T).permute(0, 2, 1)
            return q_asinh, x_tgt

        q_flat = q_pred.reshape(B * C * Q, T)

        saved_mean = self.norm_fn.mean.repeat_interleave(Q, dim=0)
        saved_std  = self.norm_fn.std.repeat_interleave(Q, dim=0)
        self.norm_fn.mean, self.norm_fn.std = saved_mean, saved_std
        q_inv = self.norm_fn.inverse_transform(q_flat)

        q_inv = q_inv.reshape(B, C, Q, T).permute(0, 3, 1, 2)
        return q_inv

    def forward(self, x_enc, mask=None, miss_mask=None, pad_mask=None, return_raw=True):
        return self.imputation(x_enc, mask, miss_mask=miss_mask, pad_mask=pad_mask,
                               return_raw=return_raw)
