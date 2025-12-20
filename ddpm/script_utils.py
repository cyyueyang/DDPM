import torchvision
import argparse
import torch.nn.functional as F
from .unet import UNet
from .diffusion import generate_cosine_schedule, generate_linear_schedule, GaussianDiffusion
def cycle(dl):
    while True:
        for x in dl:
            yield x

def get_transform():
    class ReScale():
        def __call__(self, x):
            return 2 * x - 1

    return torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        ReScale()
    ])

def str2bool(v):
    return v.lower() in ('true', '1')

def add_dict_to_argparser(parser, dict):
    for k, v in dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v_type, bool):
            v_type = str2bool
        parser.add_argument(f'--{k}', type=v_type, default=v)

def diffusion_defaults():
    defaults = dict(
        time_steps = 1000,
        scheduler = "linear",
        loss_type = "l2",
        base_channels = 128,
        channel_mults = (1, 2, 2, 2),
        num_res_blocks = 2,
        time_emb_dim = 128*4,
        norm = "gn",
        dropout = 0.1,
        activation = "silu",
        attention_resolutions = (1,),
        ema_decay = 0.99,
        ema_update_rate = 1
    )
    return defaults

def get_diffusion_from_args(args):
    activation = {
        "relu": F.relu,
        "silu": F.silu,
    }

    model = UNet(
        img_channels=3,
        base_channels=args.base_channels,
        channel_mults=args.channel_mults,
        num_res_blocks=args.num_res_blocks,
        time_emb_dim=args.time_emb_dim,
        norm=args.norm,
        dropout=args.dropout,
        activation=activation[args.activation],
        init_padding=0,
        attention_resolution=args.attention_resolutions
    )
    betas = None
    if args.scheduler == "linear":
        betas = generate_linear_schedule(args.num_timesteps, args.schedule_low * 1000 / args.num_timesteps, args.schedule_high * 1000 / args.num_timesteps)
    elif args.scheduler == "cosine":
        betas = generate_cosine_schedule(args.num_timesteps)

    diffusion = GaussianDiffusion(
        model,
        img_size=(32, 32),
        img_channels=3,
        betas=betas,
        loss_type=args.loss_type,
        ema_decay=args.ema_decay,
        ema_update_rate=args.ema_update_rate,
        ema_start=2000
    )

    return diffusion






