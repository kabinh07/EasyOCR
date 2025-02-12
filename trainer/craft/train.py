# -*- coding: utf-8 -*-
import argparse
import os
import shutil
import time
import multiprocessing as mp
import yaml
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

import torch.multiprocessing as mp

from config.load_config import load_yaml, DotDict
from data.dataset import SynthTextDataSet, CustomDataset
from loss.mseloss import Maploss_v2, Maploss_v3
from model.craft import CRAFT
from eval import main_eval
from metrics.eval_det_iou import DetectionIoUEvaluator
from utils.util import copyStateDict, save_parser

from PIL import Image, ImageDraw
from torch.utils.tensorboard import SummaryWriter
import cv2

#DDP
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

def ddp_setup(rank: int, world_size: int):
   """
   Args:
       rank: Unique identifier of each process
      world_size: Total number of processes
   """
   os.environ["MASTER_ADDR"] = "localhost"
   os.environ["MASTER_PORT"] = "12355"
   torch.cuda.set_device(rank)
   init_process_group(backend="nccl", rank=rank, world_size=world_size)

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')
        self.best_model=None
    def early_stop(self, validation_loss,model,iteration):
        # print(f"MIN Validation LOSS={self.min_validation_loss:.7f},CURRENT LOSS:{validation_loss:.7f},CURRENT COUNTER: {self.counter}, ITERATION {iteration} ")
        if validation_loss < self.min_validation_loss:
            self.best_model = model
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class Trainer(object):
    def __init__(self, config, mode):
        self.config = config
        if self.config.tensorboard:
            self.writer = SummaryWriter()
        self.mode = mode
        self.early_stopper = EarlyStopper(patience=int(self.config.train.patience) if self.config.train.patience else 5)

    def get_synth_loader(self):

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

        syn_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.train.batch_size // self.config.train.synth_ratio,
            shuffle=False,
            num_workers=self.config.train.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return syn_loader

    def get_custom_dataset(self, data_type = "train"):

        custom_dataset = CustomDataset(
            output_size=self.config.train.data.output_size,
            data_dir=self.config.data_root_dir,
            data_type = data_type,
            saved_gt_dir=None,
            mean=self.config.train.data.mean,
            variance=self.config.train.data.variance,
            gauss_init_size=self.config.train.data.gauss_init_size,
            gauss_sigma=self.config.train.data.gauss_sigma,
            enlarge_region=self.config.train.data.enlarge_region,
            enlarge_affinity=self.config.train.data.enlarge_affinity,
            watershed_param=self.config.train.data.watershed,
            aug=self.config.train.data.custom_aug,
            vis_test_dir=self.config.vis_test_dir,
            sample=self.config.train.data.custom_sample,
            vis_opt=self.config.train.data.vis_opt,
            pseudo_vis_opt=self.config.train.data.pseudo_vis_opt,
            do_not_care_label=self.config.train.data.do_not_care_label,
        )

        return custom_dataset

    def get_load_param(self, gpu):
        rank = torch.distributed.get_rank()  # Get current process rank

        if rank == 0:
            # Load checkpoint ONLY on rank 0
            if self.config.train.ckpt_path is not None:
                print(f"Rank {rank}: Loading checkpoint from {self.config.train.ckpt_path}")
                map_location = f"cuda:{gpu}"
                param = torch.load(self.config.train.ckpt_path, map_location=map_location)
            else:
                param = None
        else:
            param = None  # Other ranks do not load from disk

        # Broadcast the loaded parameters from rank 0 to all processes
        param_list = [param]  # Convert to list for broadcast
        torch.distributed.broadcast_object_list(param_list, src=0)  
        param = param_list[0]  # Extract back from list

        return param

    def adjust_learning_rate(self, optimizer, gamma, step, lr):
        lr = lr * (gamma ** step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        return param_group["lr"]

    def get_loss(self):
        if self.config.train.loss == 2:
            criterion = Maploss_v2()
        elif self.config.train.loss == 3:
            criterion = Maploss_v3()
        else:
            raise Exception("Undefined loss")
        return criterion

    def iou_eval(self, dataset, train_step, buffer, model):
        test_config = DotDict(self.config.test[dataset])

        val_result_dir = os.path.join(
            self.config.results_dir, "{}/{}".format(dataset + "_iou", str(train_step))
        )

        evaluator = DetectionIoUEvaluator()

        metrics = main_eval(
            None,
            self.config.train.backbone,
            test_config,
            evaluator,
            val_result_dir,
            buffer,
            model,
            self.mode,
        )
        if self.config.tensorboard:    
            self.writer.add_scalar("Evaluation precision", np.round(metrics['precision'], 3), train_step)
            self.writer.add_scalar("Evaluation recall", np.round(metrics['recall'], 3), train_step)
            self.writer.add_scalar("Evaluation hmean", np.round(metrics['hmean'], 3), train_step)

        if self.gpu == 0 and self.config.wandb_opt:
            wandb.log(
                {
                    "{} iou Recall".format(dataset): np.round(metrics["recall"], 3),
                    "{} iou Precision".format(dataset): np.round(
                        metrics["precision"], 3
                    ),
                    "{} iou F1-score".format(dataset): np.round(metrics["hmean"], 3),
                }
            )

    def train(self, rank, world_size, buffer_dict):
        ddp_setup(rank, world_size)
        # MODEL -------------------------------------------------------------------------------------------------------#
        # SUPERVISION model
        supervision_device = rank
        supervision_param = self.get_load_param(supervision_device)


        if self.config.mode == "weak_supervision":
            if self.config.train.backbone == "vgg":
                supervision_model = CRAFT(pretrained=False, amp=self.config.train.amp)
            else:
                raise Exception("Undefined architecture")
            # print(f"prtining from train.py | devices count: {torch.cuda.device_count()}")
            if self.config.train.ckpt_path is not None:
                supervision_model.load_state_dict(
                    copyStateDict(supervision_param['craft'])
                )
                supervision_model = supervision_model.to(f"cuda:{supervision_device}")
            print(f"Supervision model loading on : gpu {supervision_device}")
        else:
            supervision_model, supervision_device = None, None

        # TRAIN model
        if self.config.train.backbone == "vgg":
            craft = CRAFT(pretrained=False, amp=self.config.train.amp)
        else:
            raise Exception("Undefined architecture")

        if self.config.train.ckpt_path is not None:
            craft.load_state_dict(copyStateDict(supervision_param['craft']))

        craft = craft.cuda()
        craft = DDP(craft)
        

        torch.backends.cudnn.benchmark = True

        # DATASET -----------------------------------------------------------------------------------------------------#

        if self.config.train.use_synthtext:
            trn_syn_loader = self.get_synth_loader()
            batch_syn = iter(trn_syn_loader)

        if self.config.train.real_dataset == "custom":
            trn_real_dataset = self.get_custom_dataset()
            valid_dataset = self.get_custom_dataset(data_type='valid')
        else:
            raise Exception("Undefined dataset")

        if self.config.mode == "weak_supervision":
            trn_real_dataset.update_model(supervision_model)
            trn_real_dataset.update_device(supervision_device)

            valid_dataset.update_model(supervision_model)
            valid_dataset.update_device(supervision_device)

        trn_real_loader = torch.utils.data.DataLoader(
            trn_real_dataset,
            batch_size=self.config.train.batch_size,
            shuffle=False,
            num_workers=self.config.train.num_workers,
            drop_last=False,
            pin_memory=True,
            sampler=DistributedSampler(trn_real_dataset),
        )
        valid_data_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=self.config.train.batch_size,
            shuffle=False,
            num_workers=self.config.train.num_workers,
            drop_last=False,
            pin_memory=True,
            sampler=DistributedSampler(valid_dataset),
        )

        # OPTIMIZER ---------------------------------------------------------------------------------------------------#
        optimizer = optim.Adam(
            craft.parameters(),
            lr=self.config.train.lr,
            weight_decay=self.config.train.weight_decay,
        )

        if self.config.train.ckpt_path is not None and self.config.train.st_iter != 0:
            optimizer.load_state_dict(copyStateDict(supervision_param["optimizer"]))
            self.config.train.st_iter = supervision_param["optimizer"]["state"][0]["step"]
            self.config.train.lr = supervision_param["optimizer"]["param_groups"][0]["lr"]

        # LOSS --------------------------------------------------------------------------------------------------------#
        # mixed precision
        if self.config.train.amp:
            scaler = torch.cuda.amp.GradScaler()

            if (
                    self.config.train.ckpt_path is not None
                    and self.config.train.st_iter != 0
            ):
                scaler.load_state_dict(copyStateDict(supervision_param["scaler"]))
        else:
            scaler = None

        criterion = self.get_loss()

        # TRAIN -------------------------------------------------------------------------------------------------------#
        train_step = self.config.train.st_iter
        whole_training_step = self.config.train.end_iter
        update_lr_rate_step = 0
        training_lr = self.config.train.lr
        loss_value = 0
        val_loss = 0
        batch_time = 0
        best_val_loss = float('inf')
        start_time = time.time()
        print(
            "================================ Train start ================================"
        )

        while train_step < whole_training_step:
            for (
                    index,
                    (
                            images,
                            region_scores,
                            affinity_scores,
                            confidence_masks,
                    ),
            ) in enumerate(trn_real_loader):
                craft.train()
                trn_real_loader.sampler.set_epoch(train_step)
                valid_data_loader.sampler.set_epoch(train_step)
                if train_step > 0 and train_step % self.config.train.lr_decay == 0:
                    update_lr_rate_step += 1
                    training_lr = self.adjust_learning_rate(
                        optimizer,
                        self.config.train.gamma,
                        update_lr_rate_step,
                        self.config.train.lr,
                    )

                images = images.cuda(non_blocking=True)
                region_scores = region_scores.cuda(non_blocking=True)
                affinity_scores = affinity_scores.cuda(non_blocking=True)
                confidence_masks = confidence_masks.cuda(non_blocking=True)

                if self.config.train.use_synthtext:
                    # Synth image load
                    syn_image, syn_region_label, syn_affi_label, syn_confidence_mask = next(
                        batch_syn
                    )
                    syn_image = syn_image.cuda(non_blocking=True)
                    syn_region_label = syn_region_label.cuda(non_blocking=True)
                    syn_affi_label = syn_affi_label.cuda(non_blocking=True)
                    syn_confidence_mask = syn_confidence_mask.cuda(non_blocking=True)

                    # concat syn & custom image
                    images = torch.cat((syn_image, images), 0)
                    region_image_label = torch.cat(
                        (syn_region_label, region_scores), 0
                    )
                    affinity_image_label = torch.cat((syn_affi_label, affinity_scores), 0)
                    confidence_mask_label = torch.cat(
                        (syn_confidence_mask, confidence_masks), 0
                    )
                else:
                    region_image_label = region_scores
                    affinity_image_label = affinity_scores
                    confidence_mask_label = confidence_masks

                if self.config.train.amp:
                    with torch.cuda.amp.autocast():

                        output, _ = craft(images)
                        out1 = output[:, :, :, 0]
                        out2 = output[:, :, :, 1]
                        
                        if train_step % 500 == 0:
                            # print(images[0].squeeze().permute(1,2,0).cpu().numpy().shape)
                            img = Image.fromarray(images[0].permute(1,2,0).cpu().detach().numpy().astype(np.uint8))
                            reg_scaled = (cv2.resize(region_image_label[0].cpu().detach().numpy(), (768, 768), interpolation=cv2.INTER_LINEAR)*255).astype(np.uint8)
                            reg = Image.fromarray(reg_scaled, mode = "L")
                            aff_scaled = (cv2.resize(affinity_image_label[0].cpu().detach().numpy(), (768, 768), interpolation=cv2.INTER_LINEAR)*255).astype(np.uint8)
                            aff = Image.fromarray(aff_scaled, mode = "L")
                            cnf_scaled = (cv2.resize(confidence_mask_label[0].cpu().detach().numpy(), (768, 768), interpolation=cv2.INTER_LINEAR)*255).astype(np.uint8)
                            cnf = Image.fromarray(cnf_scaled, mode = "L")
                            # print(f"Printing from trianing loop | size of output: {out1[0].cpu().detach().numpy().shape} | type: {type(out1[0].cpu().detach().numpy())}")
                            ot1_scaled = cv2.resize((out1[0].cpu().detach().numpy()*255).astype(np.uint8), (768, 768), interpolation=cv2.INTER_LINEAR).astype(np.uint8)
                            # print(f"Printing from trianing loop | size of output: {ot1_scaled.shape}")
                            ot1 = Image.fromarray(ot1_scaled, mode = "L")
                            ot2_scaled = cv2.resize((out2[0].cpu().detach().numpy()*255).astype(np.uint8), (768, 768), interpolation=cv2.INTER_LINEAR).astype(np.uint8)
                            ot2 = Image.fromarray(ot2_scaled, mode = "L")
                            img.save(os.path.join('/home/EasyOCR/example_data', f"{train_step}_image.jpg"))
                            reg.save(os.path.join('/home/EasyOCR/example_data', f"{train_step}_region.jpg"))
                            aff.save(os.path.join('/home/EasyOCR/example_data', f"{train_step}_affinity.jpg"))
                            cnf.save(os.path.join('/home/EasyOCR/example_data', f"{train_step}_confidence.jpg"))
                            ot1.save(os.path.join('/home/EasyOCR/example_data', f"{train_step}_output_reg.jpg"))
                            ot2.save(os.path.join('/home/EasyOCR/example_data', f"{train_step}_output_aff.jpg"))

                        loss = criterion(
                            region_image_label,
                            affinity_image_label,
                            out1,
                            out2,
                            confidence_mask_label,
                            self.config.train.neg_rto,
                            self.config.train.n_min_neg,
                        )

                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                else:
                    output, _ = craft(images)
                    out1 = output[:, :, :, 0]
                    out2 = output[:, :, :, 1]
                    loss = criterion(
                        region_image_label,
                        affinity_image_label,
                        out1,
                        out2,
                        confidence_mask_label,
                        self.config.train.neg_rto,
                    )

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                end_time = time.time()
                loss_value += loss.item()
                batch_time += end_time - start_time

                if train_step > 0 and train_step % 5 == 0:
                    mean_loss = loss_value / 5
                    loss_value = 0
                    avg_batch_time = batch_time / 5
                    batch_time = 0
                    start_time = time.time()

                    print(
                        "{}, training_step: {}|{}, learning rate: {:.8f}, "
                        "training_loss: {:.5f}, avg_batch_time: {:.5f}".format(
                            time.strftime(
                                "%Y-%m-%d:%H:%M:%S", time.localtime(time.time())
                            ),
                            train_step,
                            whole_training_step,
                            training_lr,
                            mean_loss,
                            avg_batch_time,
                        )
                    )
                    if self.config.tensorboard:
                        self.writer.add_scalar("Training loss", mean_loss, train_step)
                    if self.config.wandb_opt:
                        wandb.log({"train_step": train_step, "mean_loss": mean_loss})

                if (
                        train_step % self.config.train.eval_interval == 0
                        and train_step != 0
                ):

                    craft.eval()

                    print("Saving state, index:", train_step)
                    save_param_dic = {
                        "iter": train_step,
                        "craft": craft.module.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    }
                    save_param_path = (
                            self.config.results_dir
                            + "/CRAFT_clr_best.pt"
                    )

                    if self.config.train.amp:
                        save_param_dic["scaler"] = scaler.state_dict()
                        save_param_path = (
                                self.config.results_dir
                                + "/CRAFT_clr_best.pt"
                        )
                    
                    torch.save(save_param_dic, self.config.results_dir+"/CRAFT_clr_last.pt")

                    # validation
                    for (index_val,(images, region_scores, affinity_scores, confidence_masks,),) in enumerate(valid_data_loader):
                        images = images.cuda(non_blocking=True)
                        region_scores = region_scores.cuda(non_blocking=True)
                        affinity_scores = affinity_scores.cuda(non_blocking=True)
                        confidence_masks = confidence_masks.cuda(non_blocking=True)
                        region_image_label = region_scores
                        affinity_image_label = affinity_scores
                        confidence_mask_label = confidence_masks

                        if self.config.train.amp:
                            with torch.cuda.amp.autocast():

                                # for name, param in craft.named_parameters():
                                #     if param.requires_grad:
                                #         print(name)
                                with torch.no_grad():
                                    output, _ = craft(images)
                                    out1 = output[:, :, :, 0]
                                    out2 = output[:, :, :, 1]

                                # print(f"out1 shape: {out1.shape}")
                                # print(f"out2 shape: {out2.shape}")

                                loss = criterion(
                                    region_image_label,
                                    affinity_image_label,
                                    out1,
                                    out2,
                                    confidence_mask_label,
                                    self.config.train.neg_rto,
                                    self.config.train.n_min_neg,
                                )
                            optimizer.zero_grad()
                        val_loss += loss.item()
                    mean_val_loss = val_loss/len(valid_data_loader)
                    if self.config.tensorboard:
                        self.writer.add_scalar("Validation loss", mean_val_loss , train_step)
                    # if index == self.config.eval_interval:
                    #     best_val_loss = mean_val_loss
                    #     torch.save(save_param_dic, save_param_path)
                    if mean_val_loss < best_val_loss:
                        best_val_loss = mean_val_loss
                        torch.save(save_param_dic, save_param_path)
                    val_loss = 0

                    self.iou_eval(
                        "custom_data",
                        train_step,
                        buffer_dict["custom_data"],
                        craft,
                    )

                train_step += 1
                if train_step >= whole_training_step:
                    break

            if self.config.mode == "weak_supervision":
                state_dict = craft.module.state_dict()
                supervision_model.load_state_dict(state_dict)
                trn_real_dataset.update_model(supervision_model)

            ea = self.early_stopper.early_stop(mean_val_loss, craft, index)
            # print(f"ITERATION: {train_step} -> TRAIN LOSS: {mean_loss:.7f} , MIN LOSS: {self.early_stopper.min_validation_loss:.7f} STEP {self.early_stopper.counter}")
            if ea:
                torch.save(
                    self.early_stopper.best_model.module.state_dict(), save_param_path)
                print(f'end the training : early stop: {ea} MIN_LOSS: {self.early_stopper.min_validation_loss:.7f}')
                sys.exit()

        # save last model
        save_param_dic = {
            "iter": train_step,
            "craft": craft.module.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_param_path = (
                self.config.results_dir + "/CRAFT_clr_" + repr(train_step) + ".pth"
        )

        if self.config.train.amp:
            save_param_dic["scaler"] = scaler.state_dict()
            save_param_path = (
                    self.config.results_dir
                    + "/CRAFT_clr_amp_"
                    + repr(train_step)
                    + ".pth"
            )
        torch.save(save_param_dic, save_param_path)
        if self.config.tensorboard:
            self.writer.flush()
        destroy_process_group()


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

    args = parser.parse_args()

    # load configure
    exp_name = args.yaml
    config = load_yaml(args.yaml)

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

    if config["mode"] == "weak_supervision":
        mode = "weak_supervision"
    else:
        mode = None


    # Apply config to wandb
    if config["wandb_opt"]:
        wandb.init(project="craft-stage2", entity="user_name", name=exp_name)
        wandb.config.update(config)

    config = DotDict(config)

    # Start train
    buffer_dict = {"custom_data":None}

    trainer = Trainer(config, mode)
    world_size = torch.cuda.device_count()
    mp.spawn(trainer.train, args=(world_size, buffer_dict,), nprocs=world_size)

    if config["wandb_opt"]:
        wandb.finish()
    if config["tensorboard"]:
        trainer.writer.close()
    sys.exit()


if __name__ == "__main__":
    main()
