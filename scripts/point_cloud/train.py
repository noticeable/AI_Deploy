#!/usr/bin/env python

import argparse
import pathlib
import time

import numpy as np
import torch

from fvcore.common.checkpoint import Checkpointer

from pytorch_point_cloud import (
    apply_data_parallel_wrapper,
    create_dataloader,
    create_loss,
    create_model,
    create_model_from_checkpoint,
    get_default_config,
    update_config,
)
from pytorch_image_classification.scheduler import create_scheduler
from pytorch_image_classification.utils import (
    AverageMeter,
    DummyWriter,
    create_logger,
    create_tensorboard_writer,
    save_config,
    set_seed,
    setup_cudnn,
)
from pytorch_image_classification.optim import create_optimizer
from pytorch_image_classification.models.qat_common import unwrap_model
from pytorch_point_cloud.models.qat import (
    apply_qat_epoch_controls,
    is_qat_enabled,
    is_qat_supported_model,
    prepare_model_for_qat,
)



def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    config = get_default_config()
    config.merge_from_file(args.config)
    config.merge_from_list(args.options)
    if args.resume:
        config_path = pathlib.Path(args.resume) / 'config.yaml'
        if config_path.exists():
            config.merge_from_file(config_path.as_posix())
        config.merge_from_list(['train.resume', True, 'train.output_dir', args.resume])
    config = update_config(config)
    if not torch.cuda.is_available():
        config.device = 'cpu'
    config.freeze()
    return config



def _to_device(config, points, targets, device):
    points = points.to(device)
    if config.task == 'classification':
        targets = targets.to(device)
    else:
        targets = {k: v.to(device) for k, v in targets.items()}
    return points, targets



def _extract_main_logits(config, outputs):
    if config.task == 'classification':
        return outputs['logits'] if isinstance(outputs, dict) else outputs
    return outputs['seg_logits']



def _compute_metric(config, logits, targets):
    if config.task == 'classification':
        preds = logits.argmax(dim=1)
        return preds.eq(targets).float().mean().item()
    preds = logits.argmax(dim=-1)
    return preds.eq(targets['seg_label']).float().mean().item()



def train_one_epoch(epoch, config, model, optimizer, scheduler, loss_func,
                    train_loader, logger, writer):
    model.train()
    device = torch.device(config.device)
    loss_meter = AverageMeter()
    metric_meter = AverageMeter()
    start = time.time()

    for step, (points, targets) in enumerate(train_loader, 1):
        points, targets = _to_device(config, points, targets, device)
        optimizer.zero_grad()
        outputs = model(points)
        loss = loss_func(outputs, targets)
        loss.backward()
        if config.train.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           config.train.gradient_clip)
        optimizer.step()
        scheduler.step()

        logits = _extract_main_logits(config, outputs)
        metric = _compute_metric(config, logits, targets)

        loss_meter.update(loss.item(), points.size(0))
        metric_meter.update(metric, points.size(0))

        if step % config.train.log_period == 0 or step == len(train_loader):
            logger.info(
                f'Epoch {epoch} Step {step}/{len(train_loader)} '
                f'lr {scheduler.get_last_lr()[0]:.6f} '
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                f'metric {metric_meter.val:.4f} ({metric_meter.avg:.4f})')

    elapsed = time.time() - start
    logger.info(f'Train epoch {epoch} elapsed {elapsed:.2f}')
    writer.add_scalar('Train/Loss', loss_meter.avg, epoch)
    writer.add_scalar('Train/Metric', metric_meter.avg, epoch)
    writer.add_scalar('Train/LearningRate', scheduler.get_last_lr()[0], epoch)
    return loss_meter.avg, metric_meter.avg



def validate(epoch, config, model, loss_func, val_loader, logger, writer):
    model.eval()
    device = torch.device(config.device)
    loss_meter = AverageMeter()
    metric_meter = AverageMeter()
    start = time.time()

    with torch.no_grad():
        for points, targets in val_loader:
            points, targets = _to_device(config, points, targets, device)
            outputs = model(points)
            loss = loss_func(outputs, targets)
            logits = _extract_main_logits(config, outputs)
            metric = _compute_metric(config, logits, targets)
            loss_meter.update(loss.item(), points.size(0))
            metric_meter.update(metric, points.size(0))

    elapsed = time.time() - start
    logger.info(
        f'Val epoch {epoch} loss {loss_meter.avg:.4f} metric {metric_meter.avg:.4f}')
    logger.info(f'Val epoch {epoch} elapsed {elapsed:.2f}')
    writer.add_scalar('Val/Loss', loss_meter.avg, epoch)
    writer.add_scalar('Val/Metric', metric_meter.avg, epoch)
    return loss_meter.avg, metric_meter.avg



def main():
    config = load_config()
    set_seed(config)
    setup_cudnn(config)

    output_dir = pathlib.Path(config.train.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    save_config(config, output_dir / 'config.yaml')

    logger = create_logger(__name__, distributed_rank=0, output_dir=output_dir)
    logger.info(config)

    train_loader = create_dataloader(config, is_train=True)
    val_loader = create_dataloader(config, is_train=False)

    if config.train.checkpoint and not config.train.resume:
        model, config, _ = create_model_from_checkpoint(config, config.train.checkpoint, create_model)
    else:
        model = create_model(config)
    qat_state = None
    if is_qat_enabled(config):
        if not is_qat_supported_model(config):
            raise ValueError(f'QAT is not supported for point-cloud model {(config.model.type, config.model.name)}')
        model, qat_state = prepare_model_for_qat(config, model)
    model = apply_data_parallel_wrapper(config, model)
    optimizer = create_optimizer(config, model)
    scheduler = create_scheduler(config, optimizer, steps_per_epoch=len(train_loader))
    checkpointer = Checkpointer(model,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                save_dir=output_dir,
                                save_to_disk=True)

    if config.train.resume:
        checkpointer.resume_or_load('', resume=True)

    writer = create_tensorboard_writer(config, output_dir, purge_step=1) \
        if config.train.use_tensorboard else DummyWriter()

    train_loss, val_loss = create_loss(config)

    for epoch in range(1, config.scheduler.epochs + 1):
        apply_qat_epoch_controls(config, unwrap_model(model), epoch, qat_state)
        train_one_epoch(epoch, config, model, optimizer, scheduler, train_loss,
                        train_loader, logger, writer)
        validate(epoch, config, model, val_loss, val_loader, logger, writer)
        if epoch % config.train.checkpoint_period == 0 or epoch == config.scheduler.epochs:
            checkpointer.save(f'checkpoint_{epoch:05d}', epoch=epoch, config=config.as_dict())
        writer.flush()


if __name__ == '__main__':
    main()
