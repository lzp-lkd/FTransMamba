import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from tools.cfg import py2cfg

import os
import torch
from torch import nn
import cv2
import numpy as np
import argparse
from pathlib import Path
from tools.metric import Evaluator
from pytorch_lightning.loggers import CSVLogger
import random


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, help="Path to the config.", required=True)
    return parser.parse_args()


class Supervision_Train(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.net = config.net
        self.loss = config.loss

        self.metrics_train = Evaluator(num_class=config.num_classes)
        self.metrics_val = Evaluator(num_class=config.num_classes)

    def forward(self, x):
        seg_pre = self.net(x)
        return seg_pre

    def training_step(self, batch, batch_idx):
        img, mask = batch['img'], batch['gt_semantic_seg']
        prediction = self.net(img)
        loss = self.loss(prediction, mask)

        if self.config.use_aux_loss:
            pre_mask = nn.Softmax(dim=1)(prediction[0])
        else:
            pre_mask = nn.Softmax(dim=1)(prediction)

        pre_mask = pre_mask.argmax(dim=1)
        for i in range(mask.shape[0]):
            self.metrics_train.add_batch(mask[i].cpu().numpy(), pre_mask[i].cpu().numpy())

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return {"loss": loss}

    def on_train_epoch_end(self):
        if 'vaihingen' in self.config.log_name or 'potsdam' in self.config.log_name or \
                'whubuilding' in self.config.log_name or 'massbuilding' in self.config.log_name or \
                'cropland' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_train.F1()[:-1])
        else:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union())
            F1 = np.nanmean(self.metrics_train.F1())

        OA = np.nanmean(self.metrics_train.OA())
        eval_value = {'mIoU': mIoU, 'F1': F1, 'OA': OA}
        print('train:', eval_value)

        self.metrics_train.reset()
        log_dict = {'train_mIoU': mIoU, 'train_F1': F1, 'train_OA': OA}
        self.log_dict(log_dict, on_epoch=True, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        img, mask = batch['img'], batch['gt_semantic_seg']
        prediction = self.forward(img)
        pre_mask = nn.Softmax(dim=1)(prediction)
        pre_mask = pre_mask.argmax(dim=1)
        for i in range(mask.shape[0]):
            self.metrics_val.add_batch(mask[i].cpu().numpy(), pre_mask[i].cpu().numpy())

        loss_val = self.loss(prediction, mask)
        self.log("val_loss", loss_val, on_step=True, on_epoch=True, prog_bar=True)
        return {"loss_val": loss_val}

    def on_validation_epoch_end(self):
        if 'vaihingen' in self.config.log_name or 'potsdam' in self.config.log_name or \
                'whubuilding' in self.config.log_name or 'massbuilding' in self.config.log_name or \
                'cropland' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_val.F1()[:-1])
        else:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union())
            F1 = np.nanmean(self.metrics_val.F1())

        OA = np.nanmean(self.metrics_val.OA())
        eval_value = {'mIoU': mIoU, 'F1': F1, 'OA': OA}
        print('val:', eval_value)

        self.metrics_val.reset()
        log_dict = {'val_mIoU': mIoU, 'val_F1': F1, 'val_OA': OA}
        self.log_dict(log_dict, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = self.config.optimizer
        lr_scheduler = self.config.lr_scheduler
        return [optimizer], [lr_scheduler]

    def train_dataloader(self):
        return self.config.train_loader

    def val_dataloader(self):
        return self.config.val_loader

def main():
    args = get_args()
    config = py2cfg(args.config_path)
    seed_everything(42)

    # 模型检查点回调：保存最佳模型
    checkpoint_callback = ModelCheckpoint(
        save_top_k=config.save_top_k,
        monitor=config.monitor,  # 监控 val_mIoU
        save_last=config.save_last,
        mode=config.monitor_mode,  # 'max' 表示越大越好
        dirpath=config.weights_path,
        filename=config.weights_name
    )

    # 早停回调：当 val_mIoU 连续 10 个 epoch 不提升时停止训练
    early_stopping_callback = EarlyStopping(
        monitor='val_mIoU',  # 监控验证集的 mIoU
        patience=10,         # 耐心值：连续 10 个 epoch 不提升则停止
        mode='max',          # 'max' 表示 mIoU 越大越好
        verbose=True         # 打印早停信息
    )

    # 日志记录器
    logger = CSVLogger('lightning_logs', name=config.log_name)

    # 初始化模型
    model = Supervision_Train(config)
    if config.pretrained_ckpt_path:
        model = Supervision_Train.load_from_checkpoint(config.pretrained_ckpt_path, config=config)

    # 配置训练器
    trainer = pl.Trainer(
        devices=config.gpus,
        max_epochs=config.max_epoch,
        accelerator='auto',
        check_val_every_n_epoch=config.check_val_every_n_epoch,
        callbacks=[checkpoint_callback, early_stopping_callback],  # 添加早停回调
        strategy='auto',
        logger=logger,
        num_sanity_val_steps=2,
        # 防止过拟合：启用梯度裁剪
        gradient_clip_val=0.5  # 添加梯度裁剪，防止梯度爆炸
    )

    # 开始训练
    trainer.fit(model=model, ckpt_path=config.resume_ckpt_path)


if __name__ == "__main__":
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    main()
