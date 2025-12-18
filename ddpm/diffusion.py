from functools import partial
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import Tuple, List
from copy import deepcopy
from .ema import EMA
from .utils import extract

class GaussianDiffusion(nn.Module):
    def __init__(self,
                 model: nn.Module,
                 img_size: Tuple[int, int] | int,
                 img_channels: int,
                 betas: np.ndarray,
                 loss_type: str = 'l2',
                 ema_decay: float = 0.99,
                 ema_start: int = 1000,
                 ema_update_rate: int = 1,
                 ):

        super(GaussianDiffusion, self).__init__()

        self.model = model
        self.ema_model = deepcopy(model)
        self.ema_decay = ema_decay
        self.ema = EMA(self.ema_decay)
        self.ema_start = ema_start
        self.ema_update_rate = ema_update_rate
        self.step = 0

        self.img_channels = img_channels
        if isinstance(img_size, int):
            self.img_size = (img_size, img_size)
        elif isinstance(img_size, tuple):
            self.img_size = img_size

        if loss_type not in ['l2', 'l1']:
            raise ValueError('loss_type must be either "l2" or "l1"')
        self.loss_type = loss_type

        self.time_steps = len(betas)

        alphas = 1 - betas
        alphas_cumprod = np.cumprod(alphas)
        to_tensor = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_tensor(betas))
        self.register_buffer('alphas', to_tensor(alphas))
        self.register_buffer('alphas_cumprod', to_tensor(alphas_cumprod))

        self.register_buffer('sqrt_alphas_cumprod', to_tensor(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_sqrt_alphas', to_tensor(np.sqrt(1 - alphas_cumprod)))
        self.register_buffer('reciprocal_sqrt_alphas', to_tensor(1 / np.sqrt(alphas)))

        self.register_buffer('remove_noise_coeff', to_tensor(betas / np.sqrt(1 - alphas_cumprod)))
        self.register_buffer('sigma', to_tensor(np.sqrt(betas)))

    def update_ema(self):
        self.step += 1
        if self.step % self.ema_update_rate == 0:
            if self.step < self.ema_start:
                self.ema_model.load_state_dict(self.model.state_dict())
            else:
                self.ema.update_model_average(self.ema_model, self.model)

    @torch.no_grad()
    def remove_noise(self, x, t, use_ema=True):
        if use_ema:
            return (x - extract(self.remove_noise_coeff, t, x.shape) * self.ema_model(x, t)) * extract(self.reciprocal_sqrt_alphas, t, x.shape)
        else:
            return (x - extract(self.remove_noise_coeff, t, x.shape) * self.model(x, t)) * extract(self.reciprocal_sqrt_alphas, t, x.shape)

    @torch.no_grad()
    def sample(self, batch_size, device, use_ema=True):
        x = torch.randn(batch_size, self.img_channels, *(self.img_size)).to(device)

        for t in range(self.time_steps, -1, -1):
            t_batch = torch.tensor([t]).repeat(batch_size).to(device)
            x = self.remove_noise(x, t_batch, use_ema=use_ema)
            if t > 0:
                x += extract(self.sigma, t_batch, x.shape) * torch.randn_like(x)

        return x.detach().cpu()

    @torch.no_grad()
    def sample_diffusion_sequence(self, batch_size, device, use_ema=True):
        x = torch.randn(batch_size, self.img_channels, *(self.img_size)).to(device)
        diffusion_sequence = [x.detach().cpu()]

        for t in range(self.time_steps, -1, -1):
            t_batch = torch.tensor([t]).repeat(batch_size).to(device)
            x = self.remove_noise(x, t_batch, use_ema=use_ema)
            if t > 0:
                x += extract(self.sigma, t_batch, x.shape) * torch.randn_like(x)

            diffusion_sequence.append(x.detach().cpu())

        return diffusion_sequence

    def perturb_x(self, x, t, noise):
        return extract(self.sqrt_alphas_cumprod, t, x.shape) * x + extract(self.sqrt_one_minus_sqrt_alphas, t, x.shape) * noise

    def get_loss(self, x, t):
        noise = torch.randn_like(x)

        perturbed_x = self.perturb_x(x, t, noise)
        estimated_noise = self.model(perturbed_x, t)
        loss = None
        if self.loss_type == 'l2':
            loss = F.mse_loss(estimated_noise, noise)
        elif self.loss_type == 'l1':
            loss = F.l1_loss(estimated_noise, noise)

        return loss

    def forward(self, x):
        b, c, h, w = x.shape
        assert c == self.img_channels and h == self.img_size[0] and w == self.img_size[1]
        t = torch.randint(0, self.time_steps-1, (b, ), device=x.device)

        return self.get_loss(x, t)

def generate_cosine_schedule(T, s=0.008):
    def f(t, T):
        return np.cos((t / T + s) / (1 + s) * (np.pi / 2.0)) ** 2

    alphas = []
    f_0 = f(0, T)
    for t in range(T + 1):
        alphas.append(f(t, T) / f_0)

    betas = []
    for t in range(1, T + 1):
        betas.append(min(1 - alphas[t] / alphas[t-1], 0.999))

    return np.array(betas)

def generate_linear_schedule(T, low, high):
    return torch.linspace(low, high, T, device=T.device)








