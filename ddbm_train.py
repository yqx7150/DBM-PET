"""
Train a diffusion model on images.
"""

import argparse
import multiprocessing
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from ddbm import dist_util, logger
from datasets import load_data
from ddbm.resample import create_named_schedule_sampler
from ddbm.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    sample_defaults,
    args_to_dict,
    add_dict_to_argparser,
    get_workdir
)
from ddbm.train_util import TrainLoop

import torch.distributed as dist

from pathlib import Path

import wandb
import numpy as np

from glob import glob
import os
from datasets.augment import AugmentPipe


# wandb.login(key='f4b4a196791a685930bae36930565e7599dcc5d8')
# 检查 GPU 可用性并设置设备
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):

    workdir = get_workdir(args.exp)
    Path(workdir).mkdir(parents=True, exist_ok=True)
    # 初始化分布式训练工具
    dist_util.setup_dist()
    logger.configure(dir=workdir)  # 配置日志记录器，将日志保存到工作目录

    if dist.get_rank() == 0:
        name = args.exp if args.resume_checkpoint == "" else args.exp + '_resume'
        logger.log("creating model and diffusion...")

    data_image_size = args.image_size  # 获取图像尺寸

    # 如果没有指定恢复检查点，则查找最新的检查点
    if args.resume_checkpoint == "":
        model_ckpts = list(glob(f'{workdir}/*model*[0-9].*'))  # 列出模型检查点文件
        # 根据文件名中的数字后缀查找最新检查点
        if len(model_ckpts) > 0:
            max_ckpt = max(model_ckpts, key=lambda x: int(x.split('model_')[-1].split('.')[0]))
            if os.path.exists(max_ckpt):
                args.resume_checkpoint = max_ckpt  # 设置为要恢复的检查点
                if dist.get_rank() == 0:
                    logger.log('Resuming from checkpoint: ', max_ckpt)

    # 创建模型和扩散过程
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    # 将模型转移到适当的设备
    device = get_device()
    model.to(dist_util.dev())  # 将模型移动到适当的设备


    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)  # 创建调度采样器

    if args.batch_size == -1:
        batch_size = args.global_batch_size // dist.get_world_size()
        if args.global_batch_size % dist.get_world_size() != 0:
            logger.log(
                f"warning, using smaller global_batch_size of {dist.get_world_size() * batch_size} instead of {args.global_batch_size}"
            )
    else:
        batch_size = args.batch_size  # 使用指定的批量大小
    print('batch_size', batch_size)
    # 记录数据加载器的创建
    if dist.get_rank() == 0:
        logger.log("creating data loader...")
    # logger.log("creating data loader...")

    # 加载训练和测试数据
    data, test_data = load_data(
        data_dir=args.data_dir,
        dataset=args.dataset,
        batch_size=batch_size,
        image_size=data_image_size,
        # num_workers=args.num_workers,
    )
    # 如果启用数据增强，设置增强管道
    if args.use_augment:
        augment = AugmentPipe(
            p=0.12, xflip=1e8, yflip=1, scale=1, rotate_frac=1, aniso=1, translate_frac=1
        )
    else:
        augment = None

    logger.log("training...")  # 开始训练过程
    # 初始化并运行训练循环
    TrainLoop(
        model=model,
        diffusion=diffusion,
        train_data=data,
        test_data=test_data,
        batch_size=batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        test_interval=args.test_interval,
        save_interval=args.save_interval,
        save_interval_for_preemption=args.save_interval_for_preemption,
        resume_checkpoint=args.resume_checkpoint,
        workdir=workdir,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        augment_pipe=augment,
        **sample_defaults()
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="",
        dataset='edges2handbags',
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        global_batch_size=2,
        batch_size=-1,
        microbatch=-1,
        ema_rate="0.9999",
        log_interval=50,
        test_interval=500,
        save_interval=50000,
        save_interval_for_preemption=50000,
        resume_checkpoint="",
        exp='',
        use_fp16=False,
        fp16_scale_growth=1e-3,
        debug=False,
        num_workers=2,
        use_augment=False
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


# if __name__ == "__main__":
#     args = create_argparser().parse_args()
#     # 设置多进程启动方法
#     multiprocessing.set_start_method('spawn')
#     main(args)
if __name__ == "__main__":
    args = create_argparser().parse_args()
    main(args)
