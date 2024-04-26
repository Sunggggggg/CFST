import os
import random
import torch
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn

from lib.core.trainer import Trainer
from lib.core.config import parse_args
from lib.utils.utils import prepare_output_dir, create_logger, setup_for_distributed
from lib.dataset.loaders import get_data_loaders

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
    train_2d_loader, train_3d_loader, valid_loader = data_loaders
    
    # ========= Initialize networks, optimizers and lr_schedulers ========= #
    
    ## 배치놈 싱크로 나이즈 해야함

    # ========= Start Training ========= #
    

if __name__ == '__main__':
    cfg, cfg_file, args = parse_args()
    cfg = prepare_output_dir(cfg, cfg_file)

    mp.spawn(main, nprocs=cfg.GPUS, args=(args, cfg))