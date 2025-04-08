from src import model_torch
from src import model_original
from src.trainer import Trainer
from src.diffusion import GaussianDiffusion, DDIM_Sampler
import yaml
import argparse
import math
from torch.optim.lr_scheduler import LambdaLR


def get_scheduler(optimizer, warmup_epochs, total_epochs):
    def warmup_cosine_schedule_fn(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        progress = (epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        return 0.5 * (1 + math.cos(math.pi * progress))
    
    return LambdaLR(optimizer, lr_lambda=warmup_cosine_schedule_fn)


def main(args):
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    unet_cfg = config['unet']
    ddim_cfg = config['ddim']
    trainer_cfg = config['trainer']
    image_size = unet_cfg['image_size']

    if config['type'] == 'original':
        unet = model_original.Unet(**unet_cfg).to(args.device)
    elif config['type'] == 'torch':
        unet = model_torch.Unet(**unet_cfg).to(args.device)
    else:
        print("Unet type must be one of ['original', 'torch']")
        exit()

    diffusion = GaussianDiffusion(unet, image_size=image_size).to(args.device)

    ddim_samplers = []
    for sampler_cfg in ddim_cfg.values():
        ddim_samplers.append(DDIM_Sampler(diffusion, **sampler_cfg))

    trainer = Trainer(
        diffusion,
        ddim_samplers=ddim_samplers,
        exp_name=args.exp_name,
        cpu_percentage=args.cpu_percentage,
        **trainer_cfg
    )

    # Add LR scheduler using trainer's optimizer
    optimizer = trainer.optimizer  # Make sure Trainer defines self.optimizer
    total_epochs = trainer_cfg.get('epochs', 100)  # adjust fallback
    warmup_epochs = trainer_cfg.get('warmup_epochs', 5)

    trainer.scheduler = get_scheduler(optimizer, warmup_epochs, total_epochs)

    # Load checkpoint if needed
    if args.load is not None:
        trainer.load(args.load, args.tensorboard, args.no_prev_ddim_setting)

    # Start training loop
    trainer.train()


if __name__ == '__main__':
    parse = argparse.ArgumentParser(description='DDPM & DDIM')
    parse.add_argument('-c', '--config', type=str, default='./config/cifar10.yaml')
    parse.add_argument('-l', '--load', type=str, default=None)
    parse.add_argument('-t', '--tensorboard', type=str, default=None)
    parse.add_argument('--exp_name', default=None)
    parse.add_argument('--device', type=str, choices=['cuda', 'cpu'], default='cuda')
    parse.add_argument('--cpu_percentage', type=float, default=0)
    parse.add_argument('--no_prev_ddim_setting', action='store_true')
    args = parse.parse_args()
    main(args)
