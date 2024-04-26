import os
import random
import torch
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
from torch.nn import SyncBatchNorm
from torch.nn.parallel import DistributedDataParallel

from lib.core.trainer import Trainer
from lib.models.CFST import CFST
from lib.core.config import parse_args
from lib.utils.utils import prepare_output_dir, create_logger, setup_for_distributed, get_optimizer
from lib.dataset.loaders import get_data_loaders
from lr_scheduler import CosineAnnealingWarmupRestarts

def main(gpu, args, cfg):
    if cfg.GPUS > 1:
        dist.init_process_group(backend='nccl', init_method='tcp://localhost:1493', world_size=cfg.GPUS, rank=gpu)

    if cfg.SEED_VALUE >= 0:
        print(f'Seed value for the experiment {cfg.SEED_VALUE}')
        os.environ['PYTHONHASHSEED'] = str(cfg.SEED_VALUE)
        random.seed(cfg.SEED_VALUE)
        torch.manual_seed(cfg.SEED_VALUE)
        np.random.seed(cfg.SEED_VALUE)
        torch.cuda.manual_seed(cfg.SEED_VALUE)
        torch.cuda.manual_seed_all(cfg.SEED_VALUE)

    print("GPU index : ", gpu)
    setup_for_distributed(gpu==0)
    torch.cuda.set_device(gpu)

    logger = create_logger(cfg.LOGDIR, phase='train')
    logger.info(f'GPU name -> {torch.cuda.get_device_name()}')
    logger.info(f'GPU feat -> {torch.cuda.get_device_properties("cuda")}')

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    # ========= Dataloaders ========= #
    data_loaders = get_data_loaders(cfg, gpu)
    
    
    # ========= Initialize networks, optimizers and lr_schedulers ========= #
    model = CFST(
        seqlen=cfg.DATASET.SEQLEN,
        n_layers=cfg.MODEL.n_layers,
        d_model=cfg.MODEL.d_model,
        num_head=cfg.MODEL.num_head,
        dropout=cfg.MODEL.dropout,
        drop_path_r=cfg.MODEL.drop_path_r,
        atten_drop=cfg.MODEL.atten_drop,
        mask_ratio=cfg.MODEL.mask_ratio,
        stride_short=cfg.MODEL.stride_short,
        short_n_layers=cfg.MODEL.short_n_layers,
        device=torch.device(gpu)
    )
    model = SyncBatchNorm.convert_sync_batchnorm(model).to(gpu)
    model = DistributedDataParallel(model, device_ids=[gpu], broadcast_buffers=False)

    gen_optimizer = get_optimizer(
        model=model,
        optim_type=cfg.TRAIN.GEN_OPTIM,
        lr=cfg.TRAIN.GEN_LR,
        weight_decay=cfg.TRAIN.GEN_WD,
        momentum=cfg.TRAIN.GEN_MOMENTUM,
    )

    lr_scheduler = CosineAnnealingWarmupRestarts(
        gen_optimizer,
        first_cycle_steps = cfg.TRAIN.END_EPOCH,
        max_lr=cfg.TRAIN.GEN_LR,
        min_lr=cfg.TRAIN.GEN_LR * 0.1,
        warmup_steps=cfg.TRAIN.LR_PATIENCE,
    )
    # ========= Start Training ========= #
    

if __name__ == '__main__':
    cfg, cfg_file, args = parse_args()
    cfg = prepare_output_dir(cfg, cfg_file)

    mp.spawn(main, nprocs=cfg.GPUS, args=(args, cfg))