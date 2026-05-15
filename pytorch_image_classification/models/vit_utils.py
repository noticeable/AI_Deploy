import math
import re

import torch
import torch.nn.functional as F
import torchvision


def _extract_state_dict(checkpoint):
    """Extract the actual weight dictionary / 提取 checkpoint 中真正的权重字典。"""
    if isinstance(checkpoint, dict):
        for key in ['model', 'state_dict']:
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    return checkpoint


def _resize_pos_embedding(pos_embedding, target_length, has_class_token):
    """Resize positional embeddings across grid sizes / 在不同 patch 网格大小间重采样位置编码。"""
    if pos_embedding.shape[1] == target_length:
        return pos_embedding

    if has_class_token:
        cls_pos = pos_embedding[:, :1]
        patch_pos = pos_embedding[:, 1:]
        target_patch_length = target_length - 1
    else:
        cls_pos = None
        patch_pos = pos_embedding
        target_patch_length = target_length

    source_size = int(math.sqrt(patch_pos.shape[1]))
    target_size = int(math.sqrt(target_patch_length))
    if source_size**2 != patch_pos.shape[1] or target_size**2 != target_patch_length:
        raise ValueError('Position embedding length is not a square number')

    patch_pos = patch_pos.reshape(1, source_size, source_size, -1).permute(
        0, 3, 1, 2)
    patch_pos = F.interpolate(patch_pos,
                              size=(target_size, target_size),
                              mode='bicubic',
                              align_corners=False)
    patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, target_patch_length,
                                                      -1)
    if cls_pos is not None:
        return torch.cat([cls_pos, patch_pos], dim=1)
    return patch_pos


def _load_checkpoint_state_dict(model_config):
    """Load checkpoint weights from disk / 从磁盘加载 checkpoint 权重。"""
    checkpoint = torch.load(model_config.pretrained_path, map_location='cpu')
    return dict(_extract_state_dict(checkpoint))


def _normalize_prefix(key):
    """Strip common wrapper prefixes / 去掉常见封装前缀。"""
    prefixes = (
        'module.',
        'backbone.',
        'model.',
        'student.',
        'network.',
    )
    for prefix in prefixes:
        if key.startswith(prefix):
            return key[len(prefix):]
    return key


def _rename_stage_key(key):
    """Map stage naming variants / 统一 stage 命名差异。"""
    return re.sub(r'layers\.(\d+)\.', r'stages.\1.', key)


def _rename_block_key(key):
    """Map block naming variants / 统一 block 命名差异。"""
    key = re.sub(r'blocks\.(\d+)\.(\d+)\.', r'stages.\1.\1.', key)
    key = re.sub(r'layers\.(\d+)\.blocks\.(\d+)\.', r'stages.\1.\1.', key)
    key = re.sub(r'stage(\d+)\.(\d+)\.',
                 lambda match: f'stages.{int(match.group(1)) - 1}.{match.group(2)}.',
                 key)
    return key


def _rename_downsample_key(key):
    """Map downsample naming variants / 统一下采样层命名差异。"""
    key = re.sub(r'layers\.(\d+)\.downsample\.', r'downsamples.\1.', key)
    key = re.sub(r'downsample_layers\.(\d+)\.',
                 lambda match: f'downsamples.{int(match.group(1)) - 1}.', key)
    return key


def _rename_head_key(key):
    """Map classifier head names / 统一分类头命名差异。"""
    if key.startswith('classifier.'):
        return 'head.' + key[len('classifier.'):]
    if key.startswith('last_linear.'):
        return 'head.' + key[len('last_linear.'):]
    return key


def _generate_candidate_keys(key):
    """Generate possible remapped parameter names / 生成一组候选参数名。"""
    candidates = []
    queue = [key]
    seen = set()
    while queue:
        current = queue.pop(0)
        if current in seen:
            continue
        seen.add(current)
        candidates.append(current)
        renamed = [
            _rename_head_key(current),
            _rename_stage_key(current),
            _rename_block_key(current),
            _rename_downsample_key(current),
        ]
        for item in renamed:
            if item not in seen:
                queue.append(item)
    return candidates


def _remap_generic_state_dict(model, state_dict):
    """Match foreign checkpoint keys to local keys / 将外部 checkpoint 键名映射到当前模型。"""
    model_state = model.state_dict()
    remapped = {}
    used_targets = set()
    for key, value in state_dict.items():
        normalized = _normalize_prefix(key)
        for candidate in _generate_candidate_keys(normalized):
            if candidate in model_state and candidate not in used_targets:
                remapped[candidate] = value
                used_targets.add(candidate)
                break
        else:
            tail_matches = [target for target in model_state
                            if target.endswith(normalized)
                            and target not in used_targets]
            if len(tail_matches) == 1:
                remapped[tail_matches[0]] = value
                used_targets.add(tail_matches[0])
    return remapped


def _drop_classifier_if_needed(state_dict, reset_classifier):
    """Optionally drop classifier weights / 按需移除分类头权重以便重置类别数。"""
    if reset_classifier:
        for key in ['head.weight', 'head.bias', 'fc.weight', 'fc.bias']:
            state_dict.pop(key, None)
    return state_dict


def _load_with_fallback(model, state_dict, strict):
    """Load weights with configurable strictness / 按 strict 选项加载权重。"""
    if strict:
        model.load_state_dict(state_dict, strict=True)
    else:
        model.load_state_dict(state_dict, strict=False)


def load_vit_pretrained(model, config, model_name='vit'):
    """Load ViT/DeiT pretrained weights / 加载 ViT 或 DeiT 预训练权重。"""
    model_config = getattr(config.model, model_name)
    if not model_config.pretrained:
        return

    if model_config.pretrained_source == 'torchvision':
        weights = model_config.pretrained_path
        if weights.lower() == 'none' or weights == '':
            weights = None
        source_model = torchvision.models.vision_transformer._vision_transformer(
            patch_size=model_config.patch_size,
            num_layers=model_config.num_layers,
            num_heads=model_config.num_heads,
            hidden_dim=model_config.emb_dim,
            mlp_dim=model_config.mlp_dim,
            weights=weights,
            progress=True,
            image_size=config.dataset.image_size,
            num_classes=config.dataset.n_classes,
            representation_size=(model_config.representation_size
                                 if model_config.representation_size > 0 else
                                 None),
            dropout=model_config.dropout,
            attention_dropout=model_config.attention_dropout)
        state_dict = dict(source_model.state_dict())
    elif model_config.pretrained_source == 'checkpoint':
        state_dict = _load_checkpoint_state_dict(model_config)
    else:
        raise ValueError(
            f'model.{model_name}.pretrained_source must be torchvision or checkpoint')

    if (model_config.interpolate_position_embedding
            and 'encoder.pos_embedding' in state_dict
            and hasattr(model, 'encoder')
            and hasattr(model.encoder, 'pos_embedding')):
        state_dict['encoder.pos_embedding'] = _resize_pos_embedding(
            state_dict['encoder.pos_embedding'],
            model.encoder.pos_embedding.shape[1], model.class_token is not None)

    state_dict = _drop_classifier_if_needed(state_dict,
                                            model_config.reset_classifier)
    _load_with_fallback(model, state_dict,
                        model_config.strict_pretrained_load
                        and not model_config.reset_classifier)


def load_generic_pretrained(model, config, model_name):
    """Load pretrained weights for non-ViT transformers / 为非 ViT Transformer 加载通用预训练权重。"""
    model_config = getattr(config.model, model_name)
    if not model_config.pretrained:
        return
    if model_config.pretrained_source != 'checkpoint':
        raise ValueError(
            f'model.{model_name}.pretrained_source must be checkpoint')
    state_dict = _load_checkpoint_state_dict(model_config)
    state_dict = _remap_generic_state_dict(model, state_dict)
    state_dict = _drop_classifier_if_needed(state_dict,
                                            model_config.reset_classifier)
    _load_with_fallback(model, state_dict,
                        model_config.strict_pretrained_load
                        and not model_config.reset_classifier)
