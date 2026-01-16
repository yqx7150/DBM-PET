"""
Train a diffusion model on images.
"""

import argparse

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
wandb.login(key='f4b4a196791a685930bae36930565e7599dcc5d8')
def main(args):
    # 根据实验名称设置工作目录
    workdir = get_workdir(args.exp)
    Path(workdir).mkdir(parents=True, exist_ok=True)
    # 初始化分布式训练工具
    dist_util.setup_dist()
    logger.configure(dir=workdir)       # 配置日志记录器，将日志保存到工作目录

    # 如果是主进程，初始化Weights & Biases用于实验跟踪
    if dist.get_rank() == 0:
        name = args.exp if args.resume_checkpoint == "" else args.exp + '_resume'       #根据是否有恢复检查点来设置实验名称。如果没有恢复检查点，使用 args.exp，否则在名称后加上 _resume 以指示这是恢复训练
        wandb.init(project="bridge", group=args.exp,name=name, config=vars(args), mode='online' if not args.debug else 'disabled')
        logger.log("creating model and diffusion...")


    data_image_size = args.image_size       # 获取图像尺寸

    # 如果没有指定恢复检查点，则查找最新的检查点
    if args.resume_checkpoint == "":
        model_ckpts = list(glob(f'{workdir}/*model*[0-9].*'))   # 列出模型检查点文件
        # 根据文件名中的数字后缀查找最新检查点
        if len(model_ckpts) > 0:
            max_ckpt = max(model_ckpts, key=lambda x: int(x.split('model_')[-1].split('.')[0]))
            if os.path.exists(max_ckpt):
                args.resume_checkpoint = max_ckpt       # 设置为要恢复的检查点
                if dist.get_rank() == 0:
                    logger.log('Resuming from checkpoint: ', max_ckpt)

    # 创建模型和扩散过程
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())       # 将模型移动到适当的设备

    # 如果是主进程，监控模型以便记录
    if dist.get_rank() == 0:
        wandb.watch(model, log='all')
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)      # 创建调度采样器

    # 根据全局批量大小和分布式进程数量确定批量大小
    if args.batch_size == -1:
        batch_size = args.global_batch_size // dist.get_world_size()
        if args.global_batch_size % dist.get_world_size() != 0:
            logger.log(
                f"warning, using smaller global_batch_size of {dist.get_world_size()*batch_size} instead of {args.global_batch_size}"
            )
    else:
        batch_size = args.batch_size        # 使用指定的批量大小

    # 记录数据加载器的创建
    if dist.get_rank() == 0:
        logger.log("creating data loader...")
    # 加载训练和测试数据
    data, test_data = load_data(
        data_dir=args.data_dir,
        dataset=args.dataset,
        batch_size=batch_size,
        image_size=data_image_size,
        num_workers=args.num_workers,
    )
    # 如果启用数据增强，设置增强管道
    if args.use_augment:
        augment = AugmentPipe(
                p=0.12,xflip=1e8, yflip=1, scale=1, rotate_frac=1, aniso=1, translate_frac=1
            )
    else:
        augment = None

    logger.log("training...")   # 开始训练过程
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

# 定义默认参数字典，包含训练和模型配置的超参数
def create_argparser():
    defaults = dict(
        data_dir="",                    #数据目录
        dataset='edges2handbags',       #使用的数据集
        schedule_sampler="uniform",     #调度采样器类型
        lr=1e-4,                        #学习率
        weight_decay=0.0,               #权重衰减值
        lr_anneal_steps=0,              #学习率衰减步数
        global_batch_size=2048,         #全局批次大小
        batch_size=-1,                  #本地批次大小 -1表示不使用
        microbatch=-1,  # -1 disables microbatches      微批次
        ema_rate="0.9999",  # comma-separated list of EMA values    指数移动平均率
        log_interval=50,                #日志记录间隔
        test_interval=500,              #测试间隔
        save_interval=10000,            #模型保存间隔
        save_interval_for_preemption=50000,         #恢复时的保存间隔
        resume_checkpoint="",               #恢复训练的检查点路径
        exp='',                         #额外的实验标识符
        use_fp16=False,                 #是否使用混合精度训练
        fp16_scale_growth=1e-3,         #混合精度的缩放增长
        debug=False,                    #是否处于调试模式
        num_workers=2,                  #数据加载的工作线程数量
        use_augment=False               #是否使用数据增强
    )
    defaults.update(model_and_diffusion_defaults())     # 更新默认参数字典，加入模型和扩散相关的默认值
    parser = argparse.ArgumentParser()                  # 创建命令行参数解析器
    add_dict_to_argparser(parser, defaults)             # 将默认参数添加到解析器中
    return parser


if __name__ == "__main__":
    args = create_argparser().parse_args()
    main(args)
