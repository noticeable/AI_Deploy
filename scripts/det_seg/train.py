#!/usr/bin/env python

import argparse
import pathlib

import torch
from fvcore.common.checkpoint import Checkpointer

from pytorch_det_seg import (
    apply_data_parallel_wrapper,
    create_dataloader,
    create_loss,
    create_model,
    create_multitask_weighting,
    create_optimizer,
    create_scheduler,
    get_default_config,
    update_config,
)
from pytorch_det_seg.utils import create_logger
from pytorch_det_seg.models.qat import (
    apply_qat_epoch_controls,
    is_qat_enabled,
    is_qat_supported_model,
    prepare_model_for_qat,
)
from pytorch_image_classification.models.qat_common import unwrap_model
from pytorch_image_classification.utils import set_seed, setup_cudnn


def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    config = get_default_config()
    config.merge_from_file(args.config)
    config.merge_from_list(args.options)
    config = update_config(config)
    config.freeze()
    return config


def train_one_epoch(config, model, criterion, weighting, optimizer, scheduler, train_loader, logger, epoch=0):
    model.train()
    total_loss = 0.0
    total_det_loss = 0.0
    total_seg_loss = 0.0
    device = torch.device(config.device)
    weighting.before_epoch(epoch) if hasattr(weighting, 'before_epoch') else None
    for step, (images, targets) in enumerate(train_loader, 1):
        images = images.to(device)
        targets = [{key: value.to(device) if hasattr(value, 'to') else value for key, value in target.items()}
                   for target in targets]
        model_outputs = model(images, targets, return_outputs=True)
        loss_dict = model_outputs['losses']
        weighted = criterion(loss_dict)
        optimizer.zero_grad()
        task_weights = weighting.backward(weighted, model_outputs=model_outputs)
        loss = weighted['loss_total']
        if config.train.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.gradient_clip)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        total_det_loss += weighted['det_loss'].item()
        total_seg_loss += weighted['seg_loss'].item()
        det_weight = task_weights['det_loss'] if task_weights is not None else float(config.loss.det_weight)
        seg_weight = task_weights['seg_loss'] if task_weights is not None else float(config.loss.seg_weight)
        if step % config.train.log_period == 0 or step == len(train_loader):
            logger.info(
                f"step={step} total_loss={loss.item():.4f} "
                f"loss_box={weighted['loss_box'].item():.4f} "
                f"loss_cls={weighted['loss_cls'].item():.4f} "
                f"loss_seg={weighted['loss_seg'].item():.4f} "
                f"det_weight={det_weight:.4f} seg_weight={seg_weight:.4f}")
    average = {
        'loss_total': total_loss / max(1, len(train_loader)),
        'det_loss': total_det_loss / max(1, len(train_loader)),
        'seg_loss': total_seg_loss / max(1, len(train_loader)),
    }
    weighting.after_epoch(epoch, average) if hasattr(weighting, 'after_epoch') else None
    return average['loss_total']


def main():
    config = load_config()
    set_seed(config)
    setup_cudnn(config)

    output_dir = pathlib.Path(config.train.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    logger = create_logger(__name__, distributed_rank=0, output_dir=output_dir)

    train_loader = create_dataloader(config, is_train=True)
    model = create_model(config)
    qat_state = None
    if is_qat_enabled(config):
        if not is_qat_supported_model(config):
            raise ValueError(f'QAT is not supported for det-seg model family {(config.model.meta_architecture, config.model.name)}')
        model, qat_state = prepare_model_for_qat(config, model)
    model = apply_data_parallel_wrapper(config, model)
    criterion = create_loss(config)
    weighting = create_multitask_weighting(config, model)
    optimizer = create_optimizer(config, model)
    scheduler = create_scheduler(config, optimizer, steps_per_epoch=len(train_loader))
    checkpointer = Checkpointer(model, optimizer=optimizer, scheduler=scheduler, save_dir=output_dir)

    for epoch in range(config.scheduler.epochs):
        apply_qat_epoch_controls(config, unwrap_model(model), epoch + 1, qat_state)
        loss = train_one_epoch(config, model, criterion, weighting, optimizer, scheduler, train_loader, logger, epoch=epoch)
        logger.info(f'epoch={epoch + 1} loss={loss:.4f}')
        if (epoch + 1) % config.train.checkpoint_period == 0:
            checkpointer.save(f'checkpoint_{epoch + 1:05d}', epoch=epoch + 1, config=config.as_dict())


if __name__ == '__main__':
    main()
