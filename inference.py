"""INSID3 inference script."""

import argparse
import datetime
import os
import random
import sys
import time
from os.path import join

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import opts
from datasets import build_dataset
from models import build_foris_from_args
from utils.metrics import Evaluator, AverageMeter
from utils.visualization import save_episode_visualizations


def main(args: argparse.Namespace) -> float:
    print(args)

    # ──────── Reproducibility and logging setup ────────
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    log_file = join(args.output_dir, 'log.txt')
    with open(log_file, 'w') as fp:
        fp.write(" ".join(sys.argv) + '\n')
        fp.write(str(vars(args)) + '\n\n')

    # ──────── Model setup ────────
    model = build_foris_from_args(args)
    model.to(args.device)
    model.eval()

    print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')
    print('Start inference')

    start_time = time.time()
    miou = evaluate(args, model, log_file)
    print(f'Total inference time: {time.time() - start_time:.1f}s')
    return miou


def evaluate(args: argparse.Namespace, model: torch.nn.Module, log_file: str) -> float:
    # ──────── Dataset and loader setup ────────
    ds = build_dataset(args.dataset, args=args)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.num_workers,
                        collate_fn=lambda x: x[0])
    meter = AverageMeter(args.dataset, ds.class_ids)

    # ──────── Evaluation loop ────────
    pbar = tqdm(loader, ncols=80)
    for idx, batch in enumerate(pbar):

        ref_imgs = batch['ref_imgs']    # list of PIL Images
        ref_masks = batch['ref_masks']  # list of tensors
        tgt_img = batch['tgt_img']      # PIL Image
        tgt_mask = batch['tgt_mask']    # tensor

        # Set all references
        model._ref_images = None  # Ensure reset
        model._ref_masks = None
        for i in range(len(ref_imgs)):
            model.set_reference(ref_imgs[i], ref_masks[i])
        # Set target (pass GT mask for dataflow PNG / debug when dataflow_image_path is set)
        model.set_target(tgt_img, gt_mask=tgt_mask)
        # Segment
        pred_mask = model.segment()
        pm = pred_mask.detach()
        while pm.dim() > 2:
            pm = pm.squeeze(0)
        pred_hw = (pm.shape[0], pm.shape[1])

        tgt_mask = F.interpolate(
            tgt_mask.unsqueeze(0).unsqueeze(0).float(),
            size=pred_hw, mode='nearest',
        ).squeeze(0).squeeze(0) > 0.5

        tgt_ignore_idx = batch.get('tgt_ignore_idx')
        if tgt_ignore_idx is not None:
            tgt_ignore_idx = F.interpolate(
                tgt_ignore_idx.unsqueeze(0).unsqueeze(0).float(),
                size=pred_hw, mode='nearest',
            ).squeeze(0).squeeze(0) > 0.5

        area_inter, area_union = Evaluator.classify_prediction(
            pred_mask, tgt_mask,
            tgt_ignore_idx=tgt_ignore_idx,
        )
        meter.update(area_inter, area_union, batch['class_id'].cuda())

        fg_union = area_union[1].clamp_min(1.0)
        episode_iou = (area_inter[1] / fg_union * 100.0).item()
        # save_episode_visualizations(
        #     reference_image=ref_imgs[0],
        #     reference_mask=ref_masks[0],
        #     target_image=tgt_img,
        #     target_gt_mask=tgt_mask,
        #     predicted_mask=pred_mask,
        #     output_dir=args.output_dir,
        #     stem=f"{idx:06d}",
        #     iou=episode_iou,
        # )

        if (idx + 1) % 50 == 0:
            miou = meter.compute_iou()[0]
            pbar.set_description(f'mIoU: {miou:.1f}')
            print(' ')

    # ──────── Final results ────────
    miou = meter.compute_iou()[0].item()
    out_str = f'mIoU = {miou:.1f}'
    print(out_str)
    with open(log_file, 'a') as fp:
        fp.write(out_str + '\n')
    return miou


if __name__ == '__main__':
    parser = argparse.ArgumentParser('INSID3 inference', parents=[opts.get_args_parser()])
    args = parser.parse_args()
    timestamp = datetime.datetime.now().strftime('%m%d_%H%M')
    args.output_dir = join(args.output_dir, f'{args.exp_name}_{timestamp}')
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
