from torch.utils.data import ConcatDataset, DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from lib.dataset import *
import torch
import joblib
import os.path as osp

def get_data_loaders(cfg, gpu):
    overlap = cfg.TRAIN.OVERLAP

    def get_3d_datasets(dataset_names):
        datasets = []
        for dataset_name in dataset_names:
            db = eval(dataset_name)(load_opt=cfg.TITLE, set='train', seqlen=cfg.DATASET.SEQLEN, overlap=overlap, debug=cfg.DEBUG)
            datasets.append(db)
        return ConcatDataset(datasets)
    # ===== Train dataset (2D dataset)===== #

    # ===== Train dataset (3D dataset)===== #
    if cfg.TRAIN.DATASETS_3D :
        train_3d_dataset_names = cfg.TRAIN.DATASETS_3D
        data_3d_batch_size = cfg.TRAIN.BATCH_SIZE
        train_3d_db = get_3d_datasets(train_3d_dataset_names)
        train_3d_sampler = DistributedSampler(train_3d_db, rank=gpu, num_replicas=cfg.GPUS)

        train_3d_loader = DataLoader(
            dataset=train_3d_db,
            batch_size=data_3d_batch_size,
            num_workers=cfg.NUM_WORKERS,
            sampler=train_3d_sampler,
            pin_memory=True
        )

    # ===== Evaluation dataset ===== #
    overlap = ((cfg.DATASET.SEQLEN - 1)/float(cfg.DATASET.SEQLEN))
    valid_db = eval(cfg.TRAIN.DATASET_EVAL)(load_opt=cfg.TITLE, set='val', seqlen=cfg.DATASET.SEQLEN, overlap=overlap, debug=cfg.DEBUG)
    valid_sampler = DistributedSampler(valid_db, rank=gpu, num_replicas=cfg.GPUS)

    valid_loader = DataLoader(
        dataset=valid_db,
        batch_size=data_3d_batch_size,
        num_workers=cfg.NUM_WORKERS,
        sampler=valid_sampler,
        pin_memory=True
    )

    return None, train_3d_loader, valid_loader
