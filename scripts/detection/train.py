#!/usr/bin/env python

import argparse
import pathlib

import torch
import torch.nn.functional as F
from fvcore.common.checkpoint import Checkpointer

from pytorch_object_detection import (
    apply_data_parallel_wrapper,
    create_dataloader,
    create_model,
    create_optimizer,
    create_scheduler,
    get_default_config,
    update_config,
)
from pytorch_object_detection.utils import create_logger
from pytorch_object_detection.utils.checkpoint import create_model_from_checkpoint
from pytorch_object_detection.models.qat import (
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


def load_model_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model_state = checkpoint.get('model', checkpoint)
    model.load_state_dict(model_state)


def create_teacher_model(config, device):
    teacher_checkpoint = config.distill.teacher_checkpoint
    if not teacher_checkpoint:
        raise ValueError('Distillation enabled but distill.teacher_checkpoint is empty.')
    teacher_base_config = config.clone()
    teacher_base_config.defrost()
    teacher_base_config.model.yolo.channels = []
    teacher_base_config.freeze()
    teacher_model, _, _ = create_model_from_checkpoint(teacher_base_config, teacher_checkpoint, create_model)
    teacher_model.to(device)
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False
    return teacher_model


def compute_distill_losses(student_outputs, teacher_outputs, distill_config):
    temperature = float(distill_config.temperature)
    if temperature <= 0:
        raise ValueError(f'distill.temperature must be > 0, got {temperature}')

    student_logits = student_outputs['pred_logits']
    teacher_logits = teacher_outputs['pred_logits']
    kd_cls = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=-1),
        F.softmax(teacher_logits / temperature, dim=-1),
        reduction='batchmean',
    ) * (temperature ** 2)
    kd_box = F.smooth_l1_loss(student_outputs['pred_boxes'], teacher_outputs['pred_boxes'])
    return kd_cls, kd_box


def train_one_epoch(config, model, optimizer, scheduler, train_loader, logger, teacher_model=None):
    model.train()
    total_loss = 0.0
    device = torch.device(config.device)
    distill_enabled = config.distill.enabled and teacher_model is not None
    model_for_calls = unwrap_model(model)
    for step, (images, targets) in enumerate(train_loader, 1):
        images = images.to(device)
        if distill_enabled:
            student_outputs = model_for_calls(images, targets, return_outputs=True)
            hard_loss = sum(student_outputs['losses'].values())
            with torch.no_grad():
                teacher_outputs = teacher_model(images, targets, return_outputs=True)
            kd_cls, kd_box = compute_distill_losses(student_outputs, teacher_outputs, config.distill)
            soft_loss = config.distill.cls_weight * kd_cls + config.distill.box_weight * kd_box
            loss = config.distill.hard_loss_weight * hard_loss + config.distill.soft_loss_weight * soft_loss
            log_message = (
                f'step={step} total_loss={loss.item():.4f} '
                f'hard_loss={hard_loss.item():.4f} '
                f'kd_cls={kd_cls.item():.4f} '
                f'kd_box={kd_box.item():.4f}'
            )
        else:
            loss_dict = model(images, targets)
            hard_loss = sum(loss_dict.values())
            loss = hard_loss
            log_message = f'step={step} loss={loss.item():.4f}'
        optimizer.zero_grad()
        loss.backward()
        if config.train.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.gradient_clip)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        if step % config.train.log_period == 0 or step == len(train_loader):
            logger.info(log_message)
    return total_loss / max(1, len(train_loader))


def main():
    config = load_config()
    set_seed(config)
    setup_cudnn(config)

    output_dir = pathlib.Path(config.train.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    logger = create_logger(__name__, distributed_rank=0, output_dir=output_dir)

    train_loader = create_dataloader(config, is_train=True)
    if config.train.checkpoint:
        model, config, _ = create_model_from_checkpoint(config, config.train.checkpoint, create_model)
    else:
        model = create_model(config)
    qat_state = None
    if is_qat_enabled(config):
        if not is_qat_supported_model(config):
            raise ValueError(
                f'QAT is not supported for detection model family {(config.model.meta_architecture, config.model.name)}')
        model, qat_state = prepare_model_for_qat(config, model)
    device = torch.device(config.device)
    teacher_model = create_teacher_model(config, device) if config.distill.enabled else None
    model = apply_data_parallel_wrapper(config, model)
    optimizer = create_optimizer(config, model)
    scheduler = create_scheduler(config, optimizer, steps_per_epoch=len(train_loader))
    checkpointer = Checkpointer(model, optimizer=optimizer, scheduler=scheduler, save_dir=output_dir)

    for epoch in range(config.scheduler.epochs):
        apply_qat_epoch_controls(config, unwrap_model(model), epoch + 1, qat_state)
        loss = train_one_epoch(config, model, optimizer, scheduler, train_loader, logger, teacher_model=teacher_model)
        logger.info(f'epoch={epoch + 1} loss={loss:.4f}')
        if (epoch + 1) % config.train.checkpoint_period == 0:
            checkpointer.save(f'checkpoint_{epoch + 1:05d}', epoch=epoch + 1, config=config.as_dict())


if __name__ == '__main__':
    main()
