import torch


_YOLO_DENSE_HEAD_FIELDS = (
    'enabled',
    'type',
    'per_cell_boxes',
    'use_objectness',
    'neck_channels',
)


def _collect_head_weights(model_state, prefix):
    return {name: weight for name, weight in model_state.items() if name.startswith(prefix)}


def _infer_tinyyolo_channels(model_state, fallback_channels):
    channels = list(fallback_channels)
    if not channels:
        return channels
    if len(channels) == 4:
        stage_prefixes = ('stem.block.0', 'stage2.blocks.0.block.0', 'stage3.blocks.0.block.0', 'stage4.blocks.0.block.0')
        inferred_channels = []
        for prefix, fallback in zip(stage_prefixes, channels):
            weight = model_state.get(f'{prefix}.weight')
            if weight is not None and getattr(weight, 'ndim', 0) >= 4:
                inferred_channels.append(int(weight.shape[0]))
            else:
                inferred_channels.append(int(fallback))
        return inferred_channels
    head_weight = model_state.get('neck.heads.2.box_head.weight')
    if head_weight is not None and getattr(head_weight, 'ndim', 0) >= 4:
        channels[-1] = int(head_weight.shape[1])
    return channels


def _infer_tinyyolo_neck_channels(model_state, fallback_neck_channels):
    head_weight = model_state.get('neck.heads.2.box_head.weight')
    if head_weight is not None and getattr(head_weight, 'ndim', 0) >= 4:
        return int(head_weight.shape[1])
    return int(fallback_neck_channels)


def _resize_detection_head(module, weight):
    if getattr(weight, 'ndim', 0) == 4:
        return torch.nn.Conv2d(int(weight.shape[1]), int(weight.shape[0]), kernel_size=1)
    return torch.nn.Linear(int(weight.shape[1]), int(weight.shape[0]))


def _maybe_resize_head_module(model, module_name, weight):
    module = getattr(model, module_name, None)
    if module is None or weight is None:
        return
    current_shape = tuple(module.weight.shape)
    target_shape = tuple(weight.shape)
    if current_shape != target_shape:
        setattr(model, module_name, _resize_detection_head(module, weight))


def load_checkpoint_and_update_config(config, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    saved_config = checkpoint.get('config', {})
    model_state = checkpoint.get('model', checkpoint)
    updated_config = config.clone()
    updated_config.defrost()

    dataset_config = saved_config.get('dataset', {})
    class_names = dataset_config.get('class_names', [])
    if class_names:
        updated_config.dataset.class_names = class_names

    model_config = saved_config.get('model', {})
    if model_config.get('meta_architecture') is not None:
        updated_config.model.meta_architecture = model_config['meta_architecture']
    if model_config.get('name') is not None:
        updated_config.model.name = model_config['name']
    yolo_config = model_config.get('yolo', {})
    if yolo_config.get('block') is not None:
        updated_config.model.yolo.block = yolo_config['block']
    if yolo_config.get('block_kernel_size') is not None:
        updated_config.model.yolo.block_kernel_size = yolo_config['block_kernel_size']
    if yolo_config.get('num_candidates') is not None:
        updated_config.model.yolo.num_candidates = yolo_config['num_candidates']
    if yolo_config.get('min_box_size') is not None:
        updated_config.model.yolo.min_box_size = yolo_config['min_box_size']
    dense_head_config = yolo_config.get('dense_head', {})
    for field_name in _YOLO_DENSE_HEAD_FIELDS:
        if dense_head_config.get(field_name) is not None:
            setattr(updated_config.model.yolo.dense_head,
                    field_name,
                    dense_head_config[field_name])
    updated_config.model.yolo.dense_head.type = 'fpn_multi'
    channels = yolo_config.get('channels', [])
    if channels:
        updated_config.model.yolo.channels = _infer_tinyyolo_channels(model_state, channels)
    elif yolo_config.get('width_mult') is not None:
        updated_config.model.yolo.width_mult = yolo_config['width_mult']
    updated_config.model.yolo.dense_head.neck_channels = _infer_tinyyolo_neck_channels(
        model_state,
        getattr(updated_config.model.yolo.dense_head, 'neck_channels', 128),
    )

    updated_config.freeze()
    return checkpoint, updated_config


def create_model_from_checkpoint(config, checkpoint_path, create_model_fn):
    checkpoint, updated_config = load_checkpoint_and_update_config(config, checkpoint_path)
    model = create_model_fn(updated_config)
    model_state = checkpoint.get('model', checkpoint)
    _maybe_resize_head_module(model, 'box_head', model_state.get('box_head.weight'))
    _maybe_resize_head_module(model, 'score_head', model_state.get('score_head.weight'))
    _maybe_resize_head_module(model, 'objectness_head', model_state.get('objectness_head.weight'))

    head_prefixes = [
        'neck.heads.0.box_head',
        'neck.heads.0.score_head',
        'neck.heads.0.objectness_head',
        'neck.heads.1.box_head',
        'neck.heads.1.score_head',
        'neck.heads.1.objectness_head',
        'neck.heads.2.box_head',
        'neck.heads.2.score_head',
        'neck.heads.2.objectness_head',
    ]
    for prefix in head_prefixes:
        weight = model_state.get(f'{prefix}.weight')
        if weight is None:
            continue
        module = model
        parts = prefix.split('.')
        for part in parts[:-1]:
            module = module[int(part)] if part.isdigit() else getattr(module, part)
        last_name = parts[-1]
        head_module = getattr(module, last_name)
        if head_module is not None and tuple(head_module.weight.shape) != tuple(weight.shape):
            setattr(module, last_name, _resize_detection_head(head_module, weight))

    if hasattr(model, 'config'):
        model.config = updated_config
    model.load_state_dict(model_state, strict=False)
    model.to(updated_config.device)
    return model, updated_config, checkpoint
