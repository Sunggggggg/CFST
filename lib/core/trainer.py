# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import time
import torch
import shutil
import logging
import numpy as np
import os.path as osp
from progress.bar import Bar

from lib.core.config import BASE_DATA_DIR
from lib.utils.utils import move_dict_to_device, AverageMeter, check_data_pararell
from lib.utils.eval_utils import (
    compute_accel,
    compute_error_accel,
    compute_error_verts,
    batch_compute_similarity_transform_torch,
)

logger = logging.getLogger(__name__)


class Trainer():
    def __init__(
            self,
            cfg,
            data_loaders,
            generator,
            gen_optimizer,
            criterion,
            lr_scheduler=None,
            writer=None,
            performance_type='min',
            clip_norm_num=None,
            val_epoch=5
    ):
        dis_motion_update_steps=cfg.TRAIN.MOT_DISCR.UPDATE_STEPS
        start_epoch=cfg.TRAIN.START_EPOCH
        end_epoch=cfg.TRAIN.END_EPOCH
        device=cfg.DEVICE
        debug=cfg.DEBUG
        logdir=cfg.LOGDIR
        resume=cfg.TRAIN.RESUME
        num_iters_per_epoch=cfg.TRAIN.NUM_ITERS_PER_EPOCH
        debug_freq=cfg.DEBUG_FREQ
        self.table_name = cfg.TITLE

        self.train_2d_loader, self.train_3d_loader, self.valid_loader = data_loaders
        self.clip_norm_num = clip_norm_num
        self.train_2d_iter = self.train_3d_iter = None

        if self.train_2d_loader:
            self.train_2d_iter = iter(self.train_2d_loader)

        if self.train_3d_loader:
            self.train_3d_iter = iter(self.train_3d_loader)

        # Models and optimizers
        self.generator = generator
        self.gen_optimizer = gen_optimizer

        # Training parameters
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.criterion = criterion
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.writer = writer
        self.debug = debug
        self.debug_freq = debug_freq
        self.logdir = logdir
        self.val_epoch = val_epoch
        self.dis_motion_update_steps = dis_motion_update_steps

        self.performance_type = performance_type
        self.train_global_step = 0
        self.valid_global_step = 0
        self.epoch = 0
        self.best_performance = float('inf') if performance_type == 'min' else -float('inf')

        self.evaluation_accumulators = dict.fromkeys(['pred_j3d', 'target_j3d', 'target_theta', 'pred_verts'])

        self.num_iters_per_epoch = num_iters_per_epoch

        if self.writer is None:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=self.logdir)

        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Resume from a pretrained model
        if resume is not None:
            self.resume_pretrained(resume)

    def train(self):
        # Single epoch training routine

        losses = AverageMeter()
        kp_2d_loss_short = AverageMeter()
        kp_3d_loss_short = AverageMeter()
        accel_loss_short_2d = AverageMeter()
        accel_loss_short_3d = AverageMeter()
        timer = {
            'data': 0,
            'forward': 0,
            'loss': 0,
            'backward': 0,
            'batch': 0,
        }

        self.generator.train()

        start = time.time()

        summary_string = ''

        bar = Bar(f'Epoch {self.epoch + 1}/{self.end_epoch}', fill='#', max=self.num_iters_per_epoch)

        for i in range(self.num_iters_per_epoch):
            # Dirty solution to reset an iterator
            target_2d = target_3d = None
            if self.train_2d_iter:
                try:
                    target_2d = next(self.train_2d_iter)
                except StopIteration:
                    self.train_2d_iter = iter(self.train_2d_loader)
                    target_2d = next(self.train_2d_iter)

                move_dict_to_device(target_2d, self.device)

            if self.train_3d_iter:
                try:
                    target_3d = next(self.train_3d_iter)
                except StopIteration:
                    self.train_3d_iter = iter(self.train_3d_loader)
                    target_3d = next(self.train_3d_iter)

                move_dict_to_device(target_3d, self.device)


            # <======= Feedforward generator and discriminator
            if target_2d and target_3d:
                pass
            elif target_3d:
                inp = target_3d['video']
            else:
                pass
    
            # Replay 
            if False :
                timer['data'] = time.time() - start
                start = time.time()

                inv_inp = inp.flip(1)               # [B, hT, 2048]
                inv_vitpose = inp_vitpose.flip(1)   # [B, T, J, 3] 
                preds, mask_ids, pred_mae = self.generator(inv_inp, inv_vitpose, is_train=True)
                
                timer['forward'] = time.time() - start
                start = time.time()

                gen_loss, loss_dict = self.criterion(
                    generator_outputs_mae=pred_mae,
                    generator_outputs_short=preds,
                    data_2d=target_2d,
                    data_3d=target_3d,
                    scores=None, 
                    mask_ids=mask_ids
                )
                
                timer['loss'] = time.time() - start
                start = time.time()

                # <======= Backprop generator and discriminator
                self.gen_optimizer.zero_grad()
                gen_loss.backward()
                self.gen_optimizer.step()

            timer['data'] = time.time() - start
            start = time.time()

            preds = self.generator(inp, is_train=True)
            
            timer['forward'] = time.time() - start
            start = time.time()

            gen_loss, loss_dict = self.criterion(
                preds=preds,
                data_2d=target_2d,
                data_3d=target_3d,
            )
            
            timer['loss'] = time.time() - start
            start = time.time()

            # <======= Backprop generator and discriminator
            self.gen_optimizer.zero_grad()
            gen_loss.backward()
            self.gen_optimizer.step()

            # <======= Log training info
            total_loss = gen_loss
            # exclude motion discriminator
            # total_loss = gen_loss + motion_dis_loss

            losses.update(total_loss.item(), inp.size(0))
            kp_2d_loss_short.update(loss_dict['loss_kp_2d_short'].item(), inp.size(0))
            kp_3d_loss_short.update(loss_dict['loss_kp_3d_short'].item(), inp.size(0))

            accel_loss_short_2d.update(loss_dict['loss_accel_2d_short'].item(), inp.size(0))
            accel_loss_short_3d.update(loss_dict['loss_accel_3d_short'].item(), inp.size(0))

            timer['backward'] = time.time() - start
            timer['batch'] = timer['data'] + timer['forward'] + timer['loss'] + timer['backward']
            start = time.time()

            summary_string = f'({i + 1}/{self.num_iters_per_epoch}) | Total: {bar.elapsed_td} | ' \
                             f'ETA: {bar.eta_td:} | loss: {losses.avg:.2f}' \
                             f'| 2d_short: {kp_2d_loss_short.avg:.2f} ' \
                             f'| 3d_short: {kp_3d_loss_short.avg:.2f} ' \
                             f'| 2d_short_accel: {accel_loss_short_2d.avg:.2f} ' \
                             f'| 3d_short_accel: {accel_loss_short_3d.avg:.2f} '

            for k, v in loss_dict.items():
                summary_string += f' | {k}: {v:.3f}'
                self.writer.add_scalar('train_loss/'+k, v, global_step=self.train_global_step)
            for k,v in timer.items():
                summary_string += f' | {k}: {v:.2f}'

            self.writer.add_scalar('train_loss/loss', total_loss.item(), global_step=self.train_global_step)
            if self.debug:
                print('==== Visualize ====')
                from lib.utils.vis import batch_visualize_vid_preds
                video = target_3d['video']
                dataset = 'spin'
                vid_tensor = batch_visualize_vid_preds(video, preds[-1], target_3d.copy(),
                                                       vis_hmr=False, dataset=dataset)
                self.writer.add_video('train-video', vid_tensor, global_step=self.train_global_step, fps=10)

            self.train_global_step += 1
            bar.suffix = summary_string
            bar.next()

            if torch.isnan(total_loss):
                exit('Nan value in loss, exiting!...')
            # =======>

        bar.finish()

        logger.info(summary_string)

    def validate(self):
        self.generator.eval()

        start = time.time()

        summary_string = ''

        bar = Bar('Validation', fill='#', max=len(self.valid_loader))

        if self.evaluation_accumulators is not None:
            for k,v in self.evaluation_accumulators.items():
                self.evaluation_accumulators[k] = []

        J_regressor = torch.from_numpy(np.load(osp.join(BASE_DATA_DIR, 'J_regressor_h36m.npy'))).float()
        with torch.no_grad():
            for i, target in enumerate(self.valid_loader):
                move_dict_to_device(target, self.device)
                # <=============
                inp = target['video']
                batch = len(inp)
                preds = self.generator(inp, is_train=False, J_regressor=J_regressor)

                # convert to 14 keypoint format for evaluation
                n_kp = preds[-1]['kp_3d'].shape[-2]
                pred_j3d = preds[-1]['kp_3d'].view(-1, n_kp, 3).cpu().numpy()
                target_j3d = target['kp_3d'].view(-1, n_kp, 3).cpu().numpy()
                pred_verts = preds[-1]['verts'].view(-1, 6890, 3).cpu().numpy()
                target_theta = target['theta'].view(-1, 85).cpu().numpy()
                self.evaluation_accumulators['pred_verts'].append(pred_verts)
                self.evaluation_accumulators['target_theta'].append(target_theta)

                self.evaluation_accumulators['pred_j3d'].append(pred_j3d)
                self.evaluation_accumulators['target_j3d'].append(target_j3d)

                # =============>
                batch_time = time.time() - start

                summary_string = f'({i + 1}/{len(self.valid_loader)}) | batch: {batch_time * 10.0:.4}ms | ' \
                                f'Total: {bar.elapsed_td} | ETA: {bar.eta_td:}'

                self.valid_global_step += 1
                bar.suffix = summary_string
                bar.next()

        bar.finish()

        logger.info(summary_string)

    def fit(self):

        for epoch in range(self.start_epoch, self.end_epoch):
            self.epoch = epoch
            self.train()
            self.validate()
            if (epoch + 1) % 10 == 0 :
                performance = self.evaluate()

            # log the learning rate
            for param_group in self.gen_optimizer.param_groups:
                print(f'Learning rate {param_group["lr"]}')
                self.writer.add_scalar('lr/gen_lr', param_group['lr'], global_step=self.epoch)
            
            if (epoch + 1) % 10 == 0 :
                logger.info(f'Epoch {epoch+1} performance: {performance:.4f}')
                self.save_model(performance, epoch)

            # lr decay
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        self.writer.close()

    def save_model(self, performance, epoch):
        save_dict = {
            'epoch': epoch,
            'gen_state_dict': self.generator.state_dict(),
            'performance': performance,
            'gen_optimizer': self.gen_optimizer.state_dict(),
        }

        filename = osp.join(self.logdir, f'checkpoint.pth.tar')
        torch.save(save_dict, filename)

        if self.performance_type == 'min':
            is_best = performance < self.best_performance
        else:
            is_best = performance > self.best_performance

        if is_best:
            logger.info('Best performance achived, saving it!')
            self.best_performance = performance
            shutil.copyfile(filename, osp.join(self.logdir, f'model_best_{epoch}.pth.tar'))

            with open(osp.join(self.logdir, 'best.txt'), 'w') as f:
                f.write(str(float(performance)))

    def resume_pretrained(self, model_path):
        if osp.isfile(model_path):
            checkpoint = torch.load(model_path)
            self.start_epoch = checkpoint['epoch'] + 1
            self.generator.load_state_dict(checkpoint['gen_state_dict'])
            self.gen_optimizer.load_state_dict(checkpoint['gen_optimizer'])
            self.best_performance = checkpoint['performance']

            logger.info(f"=> loaded checkpoint '{model_path}' "
                  f"(epoch {self.start_epoch}, performance {self.best_performance})")
        else:
            logger.info(f"=> no checkpoint found at '{model_path}'")

    def evaluate(self):

        for k, v in self.evaluation_accumulators.items():
            self.evaluation_accumulators[k] = np.vstack(v)

        pred_j3ds = self.evaluation_accumulators['pred_j3d']
        target_j3ds = self.evaluation_accumulators['target_j3d']

        pred_j3ds = torch.from_numpy(pred_j3ds).float()
        target_j3ds = torch.from_numpy(target_j3ds).float()

        print(f'Evaluating on {pred_j3ds.shape[0]} number of poses...')
        pred_pelvis = (pred_j3ds[:,[2],:] + pred_j3ds[:,[3],:]) / 2.0
        target_pelvis = (target_j3ds[:,[2],:] + target_j3ds[:,[3],:]) / 2.0

        pred_j3ds -= pred_pelvis
        target_j3ds -= target_pelvis

        errors = torch.sqrt(((pred_j3ds - target_j3ds) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
        S1_hat = batch_compute_similarity_transform_torch(pred_j3ds, target_j3ds)
        errors_pa = torch.sqrt(((S1_hat - target_j3ds) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
        pred_verts = self.evaluation_accumulators['pred_verts']
        target_theta = self.evaluation_accumulators['target_theta']

        m2mm = 1000

        pve = np.mean(compute_error_verts(target_theta=target_theta, pred_verts=pred_verts)) * m2mm
        accel = np.mean(compute_accel(pred_j3ds)) * m2mm
        accel_err = np.mean(compute_error_accel(joints_pred=pred_j3ds, joints_gt=target_j3ds)) * m2mm
        mpjpe = np.mean(errors) * m2mm
        pa_mpjpe = np.mean(errors_pa) * m2mm

        eval_dict = {
            'mpjpe': mpjpe,
            'pa-mpjpe': pa_mpjpe,
            'accel': accel,
            'pve': pve,
            'accel_err': accel_err
        }

        log_str = f'Epoch {self.epoch}, '
        log_str += ' '.join([f'{k.upper()}: {v:.4f},'for k,v in eval_dict.items()])
        logger.info(log_str)

        for k,v in eval_dict.items():
            self.writer.add_scalar(f'error/{k}', v, global_step=self.epoch)
        # return accel_err
        return pa_mpjpe