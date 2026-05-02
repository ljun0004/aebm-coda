import argparse
import datetime
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from util.crop import center_crop_arr
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.loader import CachedFolder

from models.vae import AutoencoderKL
from models import mar
from engine_mar import train_one_epoch, evaluate, generate
import copy

# from taming.models.vqgan import VQModel
from ldm.models.autoencoder import VQModel
from omegaconf import OmegaConf

from coda.models import (
    MARAutoencoderKL,
    CODAQuantizer,
    build_peft_from_vae,
)

def get_args_parser():
    parser = argparse.ArgumentParser('MAR training with Diffusion Loss', add_help=False)
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int, 
                        help='Accumulate gradient iterations (for increasing effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mar_large', type=str, metavar='MODEL',
                        help='Name of model to train')

    # VAE or VQGAN parameters
    parser.add_argument('--img_size', default=256, type=int,
                        help='images input size')
    parser.add_argument('--vae_path', default="pretrained_models/vae/kl16.ckpt", type=str,
                        help='vae checkpoint path')
    parser.add_argument('--vae_lora_path', default="pretrained_models/vae/kl16.ckpt", type=str,
                        help='fine-tunedvae checkpoint path')
    parser.add_argument('--vae_quantizer_path', default="pretrained_models/vae/kl16.ckpt", type=str,
                        help='vae quantizer checkpoint path')
    parser.add_argument('--vae_cfg', default="pretrained_models/vae/kl16.ckpt", type=str,
                        help='vqgan configuration path')
    parser.add_argument('--vae_embed_dim', default=16, type=int,
                        help='vae output embedding dimension')
    parser.add_argument('--vae_stride', default=16, type=int,
                        help='tokenizer stride, default use KL16')
    parser.add_argument('--patch_size', default=1, type=int,
                        help='number of tokens to group as a patch')
    parser.add_argument('--vae_mode', type=str, default='vq',
                        help='vae mode: vq or kl')

    # Generation parameters
    parser.add_argument('--num_iter', default=64, type=int,
                        help='number of autoregressive iterations to generate an image')
    parser.add_argument('--eval_num_images', default=50000, type=int,
                        help='number of images to generate')
    parser.add_argument('--gen_num_images', default=50000, type=int,
                        help='number of images to generate')
    parser.add_argument('--cfg', default=1.0, type=float, help="classifier-free guidance")
    parser.add_argument('--cfg_schedule', default="linear", type=str)
    parser.add_argument('--label_drop_prob', default=0.1, type=float)
    parser.add_argument('--eval_freq', type=int, default=40, help='evaluation frequency')
    parser.add_argument('--gen_freq', type=int, default=40, help='evaluation frequency')
    parser.add_argument('--save_freq', type=int, default=5, help='save frequency')
    parser.add_argument('--save_last_freq', type=int, default=5, help='save last frequency')
    parser.add_argument('--online_eval', action='store_true')
    parser.add_argument('--online_gen', action='store_true')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--generate', action='store_true')
    parser.add_argument('--gen_bsz', type=int, default=1, help='generation batch size')
    parser.add_argument('--eval_bsz', type=int, default=64, help='evaluation batch size')
    parser.add_argument('--sampling_mode', type=str, default='diffusion', help='sampling mode: diffusion or reconstruction')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.02,
                        help='weight decay (default: 0.02)')

    parser.add_argument('--grad_checkpointing', action='store_true')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--lr_schedule', type=str, default='constant',
                        help='learning rate schedule')
    parser.add_argument('--warmup_epochs', type=int, default=100, metavar='N',
                        help='epochs to warmup LR')
    parser.add_argument('--ema_rate', default=0.9999, type=float)

    parser.add_argument('--celoss_scale', type=float, default=1.0,
                        help='cross entropy loss scale (default: 1.0)')
    parser.add_argument('--ddpmloss_scale', type=float, default=1.0,
                        help='diffusion loss scale (default: 1.0)')
    parser.add_argument('--reloss_scale', type=float, default=1.0,
                        help='q regularization scale (default: 1.0)')                       
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='reg term scale (default: 1.0)')
    parser.add_argument('--beta', type=float, default=1.0,
                        help='lse term scale (default: 1.0)')
    parser.add_argument('--min_logit_scale', type=float, default=0.0,
                        help='min logits scale (default: 0.0)')
    parser.add_argument('--max_logit_scale', type=float, default=1.0,
                        help='max logits scale (default: 1.0)')

    parser.add_argument('--encoder_adaln_mod', action='store_true', dest='encoder_adaln_mod',
                        help='MAE encoder use adaln modulation')
    parser.add_argument('--decoder_adaln_mod', action='store_true', dest='decoder_adaln_mod',
                        help='MAE decoder use adaln modulation')
    parser.add_argument('--final_layer_adaln_mod', action='store_true', dest='final_layer_adaln_mod',
                        help='Final layer use adaln modulation')

    # MAGE params
    parser.add_argument('--mask_ratio_min', type=float, default=0.5,
                        help='Minimum mask ratio')
    parser.add_argument('--mask_ratio_max', type=float, default=1.0,
                        help='Maximum mask ratio')
    parser.add_argument('--mask_ratio_mu', type=float, default=0.55,
                        help='Mask ratio distribution peak')
    parser.add_argument('--mask_ratio_std', type=float, default=0.25,
                        help='Mask ratio distribution std')

    # MAR params
    parser.add_argument('--grad_clip', type=float, default=3.0,
                        help='Gradient clip')
    parser.add_argument('--attn_dropout', type=float, default=0.1,
                        help='attention dropout')
    parser.add_argument('--proj_dropout', type=float, default=0.1,
                        help='projection dropout')
    parser.add_argument('--buffer_size', type=int, default=64)

    # Codebook Adapter params
    parser.add_argument('--adapter_depth', type=int, default=12)
    parser.add_argument('--adapter_embed_dim', type=int, default=1536)
    parser.add_argument('--adapter_num_heads', type=int, default=16)
    parser.add_argument('--adapter_mlp_ratio', type=int, default=4)

    # Diffusion Loss params
    parser.add_argument('--num_sampling_steps', type=int, default="100")
    parser.add_argument('--diffusion_batch_mul', type=int, default=1)
    parser.add_argument('--temperature', default=1.0, type=float, help='diffusion loss sampling temperature')

    # Dataset parameters
    parser.add_argument('--data_path', default='./data/imagenet', type=str,
                        help='dataset path')
    parser.add_argument('--class_num', default=1000, type=int)

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval_epoch', default=0, type=int, metavar='N',
                        help='eval epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    # caching latents
    parser.add_argument('--use_cached', action='store_true', dest='use_cached',
                        help='Use cached latents')
    parser.set_defaults(use_cached=False)
    parser.add_argument('--cached_path', default='', help='path to cached latents')

    return parser


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    print(f"Main - seed: {seed}")

    cudnn.benchmark = True

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    print(f"Main - num_tasks: {num_tasks}, global_rank: {global_rank}")

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    print(f"Main - log_dir: {args.log_dir}, log_writer: {log_writer}")

    # augmentation following DiT and ADM
    transform_train = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    if args.use_cached:
        dataset_train = CachedFolder(args.cached_path, args.vae_mode)
        print("Using cached dataset")
    else:
        # dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
        dataset_train = datasets.ImageFolder(args.data_path, transform=transform_train)
    print(dataset_train)

    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    print("Sampler_train = %s" % str(sampler_train))

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    sampler_val = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=False
    )
    print("Sampler_val = %s" % str(sampler_val))

    data_loader_val = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_val,
        batch_size=args.gen_bsz,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    # define the vae
    print(f"VAE mode: {args.vae_mode}")
    cookbook_size = None
    if args.vae_mode == "kl":
        vae = AutoencoderKL(embed_dim=args.vae_embed_dim, ch_mult=(1, 1, 2, 2, 4), ckpt_path=args.vae_path).cuda().eval()

        for param in vae_lora.parameters():
            param.requires_grad = False
        
    elif args.vae_mode == "coda":
        vae =  MARAutoencoderKL(embed_dim=args.vae_embed_dim, ch_mult=(1, 1, 2, 2, 4), ckpt_path=args.vae_path).cuda().eval()

        vae_lora = build_peft_from_vae(
            vae, 
            rank=32, 
            lora_module_list=['conv1', 'conv2', 'upsample', 'downsample'],
        ).cuda().eval()

        multi_quantizer = CODAQuantizer(
            hidden_dim=16,
            codebook_size=16384,
            patch_size_list=[1, 2, 3, 4, 5, 6, 8, 10, 13, 16],
            beta=0.05,
            attn_norm_type='rms_norm',
            attn_dim=32,
        ).cuda().eval()

        cookbook_size = multi_quantizer.codebook_size

        sd = torch.load(args.vae_lora_path, map_location='cpu')
        msg = vae_lora.load_state_dict(sd, strict=False)
        print("Loading pre-trained LoRA finetuned KL-VAE")
        print("Missing keys:")
        print(msg.missing_keys)
        print("Unexpected keys:")
        print(msg.unexpected_keys)
        print(f"Restored from {args.vae_lora_path}")

        sd = torch.load(args.vae_quantizer_path, map_location='cpu')
        msg = multi_quantizer.load_state_dict(sd, strict=False)
        print("Loading pre-trained Multi-Quantizer")
        print("Missing keys:")
        print(msg.missing_keys)
        print("Unexpected keys:")
        print(msg.unexpected_keys)
        print(f"Restored from {args.vae_quantizer_path}")

        for param in vae_lora.parameters():
            param.requires_grad = False

        for param in multi_quantizer.parameters():
            param.requires_grad = False

    elif args.vae_mode == "vq":
        config = OmegaConf.load(args.vae_cfg).model
        vae = VQModel(ddconfig=config.params.ddconfig,
                        # lossconfig=config.params.lossconfig,
                        n_embed=config.params.n_embed,
                        embed_dim=config.params.embed_dim,
                        ckpt_path=args.vae_path,
                        ).cuda().eval()
        cookbook_size = vae.n_embed

        for param in vae.parameters():
            param.requires_grad = False

    else:
        raise NotImplementedError

    print(f"VQGAN codebook size: {cookbook_size}")

    torch.set_float32_matmul_precision('high')
    
    model = mar.__dict__[args.model](
        img_size=args.img_size,
        vae_stride=args.vae_stride,
        patch_size=args.patch_size,
        vae_embed_dim=args.vae_embed_dim,
        cookbook_size=cookbook_size,
        mask_ratio_min=args.mask_ratio_min,
        mask_ratio_max=args.mask_ratio_max,
        mask_ratio_mu=args.mask_ratio_mu,
        mask_ratio_std=args.mask_ratio_std,
        label_drop_prob=args.label_drop_prob,
        class_num=args.class_num,
        grad_checkpointing=args.grad_checkpointing,
        num_sampling_steps=args.num_sampling_steps,
        celoss_scale=args.celoss_scale,
        ddpmloss_scale=args.ddpmloss_scale,
        reloss_scale=args.reloss_scale,
        alpha=args.alpha,
        beta=args.beta,
    )

    print("Model = %s" % str(model))
    # following timm: set wd as 0 for bias and norm layers
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of trainable parameters: {}M".format(n_params / 1e6))

    model.to(device)
    model_without_ddp = model

    eff_batch_size = args.batch_size * misc.get_world_size() * args.accum_iter

    if args.lr is None:
        args.lr = args.blr * (eff_batch_size / 256) 

    print("base lr: %.2e" % args.blr)
    print("actual lr: %.2e" % args.lr)
    print("warmup epochs: %.2e" % args.warmup_epochs)
    print("effective batch size: %d" % eff_batch_size)

    # ---------------------------------------------------------
    # PHASE 1: LOAD WEIGHTS (BEFORE DDP)
    # ---------------------------------------------------------
    checkpoint = None
    if args.resume and os.path.exists(args.resume):
        ckpt_path = args.resume
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        
        # 1. Load weights into bare model using the mapper
        # We map the dict first, then load. This makes strict=False effective.
        model_state_dict = checkpoint['model']
        msg = model_without_ddp.load_state_dict(model_state_dict, strict=False)
        
        print("Loading pre-trained model and converting to Weight Norm...")
        print("Missing keys:", msg.missing_keys)
        print("Unexpected keys:", msg.unexpected_keys)
        print(f"Restored from {ckpt_path}")

        # ---> ADD THIS LINE BACK HERE <---
        model_params = list(model_without_ddp.parameters())

        # 2. Setup EMA from checkpoint
        # We must also map the EMA state dict, otherwise 'name in ema_state_dict' 
        # will fail because the names in the checkpoint are the old ones.
        ema_state_dict = checkpoint['model_ema']
        ema_params = [
            (ema_state_dict[name] if name in ema_state_dict else p.detach().clone()).cuda() 
            for name, p in model_without_ddp.named_parameters()
        ]
        
        args.start_epoch = checkpoint['epoch'] + 1
        print("Resume checkpoint %s" % args.resume)
    else:
        print("Training from scratch")
        model_params = list(model_without_ddp.parameters())
        ema_params = [p.detach().clone() for p in model_params]

    # ---------------------------------------------------------
    # PHASE 2: WRAP IN DDP (Hooks the parameters correctly)
    # ---------------------------------------------------------
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module

    # ---------------------------------------------------------
    # PHASE 3: INITIALIZE OPTIMIZER (Tracks the DDP-hooked params)
    # ---------------------------------------------------------
    param_groups = misc.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.999))
    print(optimizer)
    loss_scaler = NativeScaler(enabled=False)

    # ---------------------------------------------------------
    # PHASE 4: LOAD OPTIMIZER STATE
    # ---------------------------------------------------------
    if checkpoint is not None:
        if 'optimizer' in checkpoint and 'epoch' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer'])
                print(">> Optimizer state loaded successfully.")
            except ValueError as e:
                print(f">> WARNING: Optimizer load failed. Skipping optimizer state load.")
                pass

            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
            print("With optim & sched!")
            
        del checkpoint
        torch.cuda.empty_cache() # Flush the massive checkpoint
    
    # evaluate FID and IS
    if args.evaluate:
        torch.cuda.empty_cache()
        evaluate(model_without_ddp, 
                 vae_lora if args.vae_mode == "coda" else vae, 
                 multi_quantizer if args.vae_mode == "coda" else None, 
                 ema_params, args, args.start_epoch, batch_size=args.eval_bsz,
                 log_writer=log_writer, cfg=args.cfg, use_ema=True)
        torch.cuda.empty_cache()
        return

    # evaluate FID and IS
    if args.generate:
        torch.cuda.empty_cache()
        generate(model_without_ddp, 
                 vae_lora if args.vae_mode == "coda" else vae, 
                 multi_quantizer if args.vae_mode == "coda" else None, 
                 ema_params, args, args.start_epoch, batch_size=args.gen_bsz, log_writer=log_writer, 
                 cfg=args.cfg, use_ema=True, data_loader=data_loader_val)
        torch.cuda.empty_cache()
        return

    # training
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(
            model, 
            vae_lora if args.vae_mode == "coda" else vae,
            multi_quantizer if args.vae_mode == "coda" else None,
            model_params, ema_params,
            data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )

        # save checkpoint
        if epoch % args.save_last_freq == 0 or epoch + 1 == args.epochs:
            # save/Overwrite the "last" version for easy resuming
            misc.save_model(args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                            loss_scaler=loss_scaler, epoch=epoch, ema_params=ema_params, epoch_name="last")
        if epoch % args.save_freq == 0 or epoch + 1 == args.epochs:
            # save the numbered version for your history
            misc.save_model(args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                            loss_scaler=loss_scaler, epoch=epoch, ema_params=ema_params, epoch_name=None)

        # online evaluation
        if args.online_eval and (epoch % args.eval_freq == 0 or epoch + 1 == args.epochs):
            torch.cuda.empty_cache()
            evaluate(model_without_ddp, 
                     vae if args.vae_mode == "coda" else vae, 
                     multi_quantizer if args.vae_mode == "coda" else None, 
                     ema_params, args, epoch, batch_size=args.eval_bsz,
                     log_writer=log_writer, cfg=args.cfg, use_ema=True)
            torch.cuda.empty_cache()

        # online unconditional generation
        if args.online_gen and (epoch % args.gen_freq == 0 or epoch + 1 == args.epochs):
            torch.cuda.empty_cache()
            generate(model_without_ddp, 
                     vae if args.vae_mode == "coda" else vae, 
                     multi_quantizer if args.vae_mode == "coda" else None, 
                     ema_params, args, epoch, batch_size=args.gen_bsz, log_writer=log_writer, 
                     cfg=args.cfg, use_ema=True, data_loader=data_loader_val)
            torch.cuda.empty_cache()

        if misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    # args.log_dir = args.output_dir
    main(args)