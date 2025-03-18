# -*- coding: utf-8 -*-
import argparse
import os
import shutil
import time
import yaml

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.utilities.seed import seed_everything

from config.load_config import load_yaml, DotDict
from data.dataset import SynthTextDataSet, CustomDataset
from loss.mseloss import Maploss_v2, Maploss_v3
from model.craft import CRAFT
from eval import main_eval
from metrics.eval_det_iou import DetectionIoUEvaluator
from utils.util import copyStateDict, save_parser


class CRAFTLightningModule(pl.LightningModule):
    def __init__(self, config):
        super(CRAFTLightningModule, self).__init__()
        self.config = config
        self.net_param = self.get_load_param()
        self.criterion = self.get_loss()
        self.craft = CRAFT(pretrained=False, amp=self.config.train.amp, dropout=0.5)
        if self.config.train.ckpt_path is not None:
            self.craft.load_state_dict(copyStateDict(self.net_param["craft"]))
        self.craft = nn.SyncBatchNorm.convert_sync_batchnorm(self.craft)

    def get_load_param(self):
        
        if self.config.train.ckpt_path is not None:
            param = torch.load(self.config.train.ckpt_path, map_location="cpu")
        else:
            param = None
        return param

    def get_loss(self):
        if self.config.train.loss == 2:
            criterion = Maploss_v2()
        elif self.config.train.loss == 3:
            criterion = Maploss_v3()
        else:
            raise Exception("Undefined loss")
        return criterion

    def forward(self, x):
        return self.craft(x)

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.craft.parameters(),
            lr=self.config.train.lr,
            weight_decay=self.config.train.weight_decay,
        )
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        images, region_scores, affinity_scores, confidence_masks = batch
        images = images.cuda(non_blocking=True)
        region_scores = region_scores.cuda(non_blocking=True)
        affinity_scores = affinity_scores.cuda(non_blocking=True)
        confidence_masks = confidence_masks.cuda(non_blocking=True)

        output, _ = self.craft(images)
        out1 = output[:, :, :, 0]
        out2 = output[:, :, :, 1]
        loss = self.criterion(
            region_scores,
            affinity_scores,
            out1,
            out2,
            confidence_masks,
            self.config.train.neg_rto,
        )
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, region_scores, affinity_scores, confidence_masks = batch
        images = images.cuda(non_blocking=True)
        region_scores = region_scores.cuda(non_blocking=True)
        affinity_scores = affinity_scores.cuda(non_blocking=True)
        confidence_masks = confidence_masks.cuda(non_blocking=True)

        output, _ = self.craft(images)
        out1 = output[:, :, :, 0]
        out2 = output[:, :, :, 1]
        loss = self.criterion(
            region_scores,
            affinity_scores,
            out1,
            out2,
            confidence_masks,
            self.config.train.neg_rto,
        )
        self.log("val_loss", loss)
        return loss

    def setup_dataloader(self, dataset):
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.train.batch_size,
            shuffle=True if self.training else False,
            num_workers=self.config.train.num_workers,
            pin_memory=True,
        )
        return loader

    def train_dataloader(self):
        if self.config.train.use_synthtext:
            dataset = SynthTextDataSet(
                output_size=self.config.train.data.output_size,
                data_dir=self.config.train.synth_data_dir,
                saved_gt_dir=None,
                mean=self.config.train.data.mean,
                variance=self.config.train.data.variance,
                gauss_init_size=self.config.train.data.gauss_init_size,
                gauss_sigma=self.config.train.data.gauss_sigma,
                enlarge_region=self.config.train.data.enlarge_region,
                enlarge_affinity=self.config.train.data.enlarge_affinity,
                aug=self.config.train.data.syn_aug,
                vis_test_dir=self.config.vis_test_dir,
                vis_opt=self.config.train.data.vis_opt,
                sample=self.config.train.data.syn_sample,
            )
        else:
            dataset = CustomDataset(
                output_size=self.config.train.data.output_size,
                data_dir=self.config.data_root_dir,
                saved_gt_dir=None,
                mean=self.config.train.data.mean,
                variance=self.config.train.data.variance,
                gauss_init_size=self.config.train.data.gauss_init_size,
                gauss_sigma=self.config.train.data.gauss_sigma,
                enlarge_region=self.config.train.data.enlarge_region,
                enlarge_=self.config.train.data.enlarge_affinity,
                watershed_param=self.config.train.data.watershed,
                aug=self.config.train.data.custom_aug,
                vis_test_dir=self.config.vis_test_dir,
                sample=self.config.train.data.custom_sample,
                vis_opt=self.config.train.data.vis_opt,
                pseudo_vis_opt=self.config.train.data.pseudo_vis_opt,
                do_not_care_label=self.config.train.data.do_not_care_label,
            )
        return self.setup_dataloader(dataset)

    def val_dataloader(self):
        dataset = CustomDataset(
            output_size=self.config.train.data.output_size,
            data_dir=self.config.data_root_dir,
            saved_gt_dir=None,
            mean=self.config.train.data.mean,
            variance=self.config.train.data.variance,
            gauss_init_size=self.config.train.data.gauss_init_size,
            gauss_sigma=self.config.train.data.gauss_sigma,
            enlarge_region=self.config.train.data.enlarge_region,
            enlarge_=self.config.train.data.enlarge_affinity,
            watershed_param=self.config.train.data.watershed,
            aug=self.config.train.data.custom_aug,
            vis_test_dir=self.config.vis_test_dir,
            sample=self.config.train.data.custom_sample,
            vis_opt=self.config.train.data.vis_opt,
            pseudo_vis_opt=self.config.train.data.pseudo_vis_opt,
            do_not_care_label=self.config.train.data.do_not_care_label,
        )
        return self.setup_dataloader(dataset)


def main():
    parser = argparse.ArgumentParser(description="CRAFT custom data train")
    parser.add_argument(
        "--yaml",
        "--yaml_file_name",
        default="custom_data_train",
        type=str,
        help="Load configuration",
    )
    parser.add_argument(
        "--port", "--use ddp port", default="2346", type=str, help="Port number"
    )
    parser.add_argument(
        "--gradient_clip_val",
        type=float,
        default=0.5,
        help="Gradient clipping value",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay value",
    )

    args = parser.parse_args()

    # load configure
    exp_name = args.yaml
    config = load_yaml(args.yaml)
    config["gradient_clip_val"] = args.gradient_clip_val
    config["weight_decay"] = args.weight_decay

    print("-" * 20 + " Options " + "-" * 20)
    print(yaml.dump(config))
    print("-" * 40)

    # Make result_dir
    res_dir = os.path.join(config["results_dir"], args.yaml)
    config["results_dir"] = res_dir
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    # Duplicate yaml file to result_dir
    shutil.copy(
        "config/" + args.yaml + ".yaml", os.path.join(res_dir, args.yaml) + ".yaml"
    )

    # Set seed for reproducibility
    seed_everything(42)

    tensorboard_logger = TensorBoardLogger("tb_logs", name=exp_name)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(
        dirpath=config["results_dir"],
        filename="{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        mode="min",
        monitor="val_loss",
    )
    model = CRAFTLightningModule(config)
    trainer = Trainer(
        max_epochs=config["train"]["end_iter"],
        gpus=torch.cuda.device_count(),
        logger=tensorboard_logger,
        strategy="ddp",
        precision=16,  # Enable mixed precision training
        callbacks=[lr_monitor, checkpoint_callback],
        gradient_clip_val=config["gradient_clip_val"],
    )

    # Find the best learning rate
    tuner = pl.tuner.Tuner(trainer)
    lr_finder = tuner.lr_find(model, trainer.train_dataloader(), trainer.val_dataloader(), num_training=1000)
    
    # Pick the best learning rate
    new_lr = lr_finder.suggestion()
    model.hparams.lr = new_lr

    trainer.fit(model)


if __name__ == "__main__":
    main()
