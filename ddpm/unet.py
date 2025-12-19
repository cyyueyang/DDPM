import torch
import torch.nn.functional as F
import torch.nn as nn
import math
from torch.nn.modules.normalization import GroupNorm

def get_norm(norm, num_channels, num_groups):
    if norm == "bn":
        return nn.BatchNorm2d(num_channels)
    elif norm == "in":
        return nn.InstanceNorm2d(num_channels, affine=True)
    elif norm == "ln":
        return nn.LayerNorm(num_channels)
    elif norm == "gn":
        return nn.GroupNorm(num_channels, num_groups)
    else:
        return nn.Identity()

class PositionalEmbedding(nn.Module):
    def __init__(self, dim, scale=1.0):
        super(PositionalEmbedding, self).__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.scale = scale

    def forward(self, x):
        half_dim = self.dim // 2
        emb = math.log(10000) / half_dim
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = torch.outer(x, emb)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class Downsample(nn.Module):
    def __init__(self, in_channels):
        super(Downsample, self).__init__()
        self.downsample = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        if x.shape[-1] % 2 == 1:
            raise ValueError("w must be even")
        if x.shape[-2] % 2 == 1:
            raise ValueError("h must be even")

        return self.downsample(x)

class Upsample(nn.Module):
    def __init__(self, in_channels):
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        return self.upsample(x)

class Attention(nn.Module):
    def __init__(self, in_channels, norm="gn", groups=32):
        super(Attention, self).__init__()
        self.in_channels = in_channels
        self.norm = get_norm(norm, in_channels, groups)
        self.to_qkv = nn.Conv2d(in_channels, 3 * in_channels, kernel_size=1)
        self.to_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = torch.split(self.norm(self.to_qkv(x)), self.in_channels, dim=1)
        q = q.permute(0, 2, 3, 1).view(b, h*w, c)
        k = k.permute(0, 2, 3, 1).view(b, h*w, c)
        v = v.permute(0, 2, 3, 1).view(b, h*w, c)

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(c)
        scores = torch.softmax(scores, dim=-1)
        out = torch.matmul(scores, v)
        out = self.to_out(out.view(b, c, h, w).permute(0, 3, 1, 2)) + x
        return out

class ResidualBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 norm="gn",
                 groups=32,
                 dropout=0.1,
                 activation=F.relu,
                 time_emb_dim=None,
                 use_attention=False):
        super(ResidualBlock, self).__init__()

        self.activation = activation
        self.norm_1 = get_norm(norm, in_channels, groups)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm_2 = get_norm(norm, out_channels, groups)
        self.conv_2 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

        self.time_bias = nn.Linear(time_emb_dim, out_channels) if time_emb_dim is not None else None

        self.residual_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.attention = Attention(out_channels, norm=norm, groups=groups) if use_attention else nn.Identity()

    def forward(self, x, time_emb=None):
        out = self.conv_1(self.activation(self.norm_1(x)))

        if self.time_bias is not None:
            if time_emb is not None:
                out += self.time_bias(self.activation(time_emb))[:, :, None, None]
            else :
                raise ValueError("time_emb must not be None if time_bias is True")

        out = self.activation(self.norm_2(out))
        out = self.conv_2(out) + self.residual_connection(out)
        out = self.attention(out)
        return out

class UNet(nn.Module):
    def __init__(self,
                 img_channels,
                 base_channels,
                 channel_mults=(1, 2, 4, 8),
                 num_res_blocks=2,
                 time_emb_dim=None,
                 time_emb_scale=1.0,
                 activation=F.relu,
                 dropout=0.1,
                 attention_resolution=(),
                 norm="gn",
                 groups=32,
                 init_padding=0,
                 use_attention=False):
        super(UNet, self).__init__()

        self.activation = activation
        self.init_padding = init_padding

        self.time_mlp = nn.Sequential(
            PositionalEmbedding(base_channels, time_emb_scale),
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        ) if time_emb_dim is not None else None

        self.init_conv = nn.Conv2d(img_channels, base_channels, kernel_size=3, padding=1)

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        channels = [base_channels]
        now_channels = base_channels

        for i, mult in enumerate(channel_mults):
            out_channels = base_channels * mult
            for _ in range(num_res_blocks):
                self.downs.append(ResidualBlock(
                    now_channels,
                    out_channels,
                    norm=norm,
                    groups=groups,
                    dropout=dropout,
                    activation=self.activation,
                    time_emb_dim=time_emb_dim,
                    use_attention=i in attention_resolution()
                ))

                now_channels = out_channels
                channels.append(out_channels)

            if i != len(channel_mults) - 1:
                self.downs.append(Downsample(now_channels))
                channels.append(now_channels)

        self.mid = nn.ModuleList([
            ResidualBlock(
                now_channels,
                now_channels,
                norm=norm,
                groups=groups,
                dropout=dropout,
                activation=self.activation,
                time_emb_dim=time_emb_dim,
                use_attention=False
            ),
            ResidualBlock(
                now_channels,
                now_channels,
                norm=norm,
                groups=groups,
                dropout=dropout,
                activation=self.activation,
                time_emb_dim=time_emb_dim,
                use_attention=False
            )
        ])

        for i, mult in reversed(list(enumerate(channels))):
            out_channels = base_channels * mult

            for _ in range(num_res_blocks+1):
                self.ups.append(
                    ResidualBlock(
                        channels.pop() + now_channels,
                        out_channels,
                        norm=norm,
                        groups=groups,
                        dropout=dropout,
                        activation=self.activation,
                        time_emb_dim=time_emb_dim,
                        use_attention=i in attention_resolution()
                    )
                )
                now_channels = out_channels

                if i != 0:
                    self.ups.append(
                        Upsample(now_channels)
                    )
        assert len(channels) == 0

        self.out_norm = get_norm(norm, base_channels, groups)
        self.out_conv = nn.Conv2d(base_channels, img_channels, kernel_size=3, padding=1)


    def forward(self, x, time=None):
        ip = self.init_padding

        if ip != 0:
            x = F.pad(x, (ip,) * 4)

        if self.time_mlp is not None:
            if time is not None:
                time_emb = self.time_mlp(time)
            else:
                raise ValueError("time_emb must not be None if time_mlp is True")
        else:
            time_emb = None

        x = self.init_conv(x)

        skips = [x]

        for layer in self.downs:
            x = layer(x, time_emb=time_emb)
            skips.append(x)
        for layer in self.mid:
            x = layer(x, time_emb=time_emb)
        for layer in self.ups:
            if isinstance(layer, ResidualBlock):
                x = torch.cat([x, skips.pop()], dim=1)
            x = layer(x, time_emb=time_emb)

        x = self.activation(self.out_norm(x))
        x = self.out_conv(x)
        if self.init_padding != 0:
            return x[:, :, ip:-ip, ip:-ip]
        return x





