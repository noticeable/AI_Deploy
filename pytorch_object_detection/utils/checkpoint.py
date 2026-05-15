import torch



def _infer_tinyyolo_channels(model_state, fallback_channels):
    channels = list(fallback_channels)
    if not channels:
        return channels
    box_head_weight = model_state.get('box_head.weight')
    score_head_weight = model_state.get('score_head.weight')
    inferred_last_channel = None
    if box_head_weight is not None and getattr(box_head_weight, 'ndim', 0) >= 2:
        inferred_last_channel = int(box_head_weight.shape[1])
    elif score_head_weight is not None and getattr(score_head_weight, 'ndim', 0) >= 2:
        inferred_last_channel = int(score_head_weight.shape[1])
    if inferred_last_channel is not None:
        channels[-1] = inferred_last_channel
    return channels



def _resize_linear(module, weight):
    return torch.nn.Linear(int(weight.shape[1]), int(weight.shape[0]))



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
    channels = yolo_config.get('channels', [])
    if channels:
        updated_config.model.yolo.channels = _infer_tinyyolo_channels(model_state, channels)
    elif yolo_config.get('width_mult') is not None:
        updated_config.model.yolo.width_mult = yolo_config['width_mult']

    updated_config.freeze()
    return checkpoint, updated_config



def create_model_from_checkpoint(config, checkpoint_path, create_model_fn):
    checkpoint, updated_config = load_checkpoint_and_update_config(config, checkpoint_path)
    model = create_model_fn(updated_config)
    model_state = checkpoint.get('model', checkpoint)
    box_head_weight = model_state.get('box_head.weight')
    score_head_weight = model_state.get('score_head.weight')
    if hasattr(model, 'box_head') and box_head_weight is not None:
        if (getattr(model.box_head, 'in_features', None) != int(box_head_weight.shape[1]) or
                getattr(model.box_head, 'out_features', None) != int(box_head_weight.shape[0])):
            model.box_head = _resize_linear(model.box_head, box_head_weight)
    if hasattr(model, 'score_head') and score_head_weight is not None:
        if (getattr(model.score_head, 'in_features', None) != int(score_head_weight.shape[1]) or
                getattr(model.score_head, 'out_features', None) != int(score_head_weight.shape[0])):
            model.score_head = _resize_linear(model.score_head, score_head_weight)
    if hasattr(model, 'config'):
        model.config = updated_config
    model.load_state_dict(model_state)
    model.to(updated_config.device)
    return model, updated_config, checkpoint
