import torch



def _resize_detection_head(module, box_weight, score_weight):
    in_features = int(box_weight.shape[1])
    n_classes = int(score_weight.shape[0] - 1)
    head_cls = type(module)
    return head_cls(in_features, n_classes)



def _resize_detection_neck(module, weight):
    neck_cls = type(module)
    return neck_cls(int(weight.shape[1]), int(weight.shape[0]))



def _resize_segmentation_neck(module, first_weight, second_weight=None):
    hidden_channels = int(second_weight.shape[1]) if second_weight is not None else int(first_weight.shape[0])
    neck_cls = type(module)
    return neck_cls(int(first_weight.shape[1]), hidden_channels)



def _resize_segmentation_head(module, conv_weight):
    in_channels = int(conv_weight.shape[1])
    n_classes = int(conv_weight.shape[0])
    head_cls = type(module)
    return head_cls(in_channels, n_classes)



def _maybe_resize_det_neck(model, model_state):
    det_neck_weight = model_state.get('det_neck.block.block.0.weight')
    if det_neck_weight is None or not hasattr(model, 'det_neck'):
        return model
    current_shape = tuple(model.det_neck.block.block[0].weight.shape)
    target_shape = tuple(det_neck_weight.shape)
    if current_shape != target_shape:
        model.det_neck = _resize_detection_neck(model.det_neck, det_neck_weight)
    return model



def _maybe_resize_seg_neck(model, model_state):
    seg_neck_first = model_state.get('seg_neck.block.0.block.0.weight')
    seg_neck_second = model_state.get('seg_neck.block.2.block.0.weight')
    if seg_neck_first is None or not hasattr(model, 'seg_neck'):
        return model
    current_first = tuple(model.seg_neck.block[0].block[0].weight.shape)
    current_second = tuple(model.seg_neck.block[2].block[0].weight.shape)
    target_first = tuple(seg_neck_first.shape)
    target_second = tuple(seg_neck_second.shape) if seg_neck_second is not None else current_second
    if current_first != target_first or current_second != target_second:
        model.seg_neck = _resize_segmentation_neck(model.seg_neck, seg_neck_first, seg_neck_second)
    return model



def _maybe_resize_det_head(model, model_state):
    det_box_weight = model_state.get('det_head.box_head.weight')
    det_score_weight = model_state.get('det_head.score_head.weight')
    if det_box_weight is None or det_score_weight is None or not hasattr(model, 'det_head'):
        return model
    current_box = tuple(model.det_head.box_head.weight.shape)
    current_score = tuple(model.det_head.score_head.weight.shape)
    if current_box != tuple(det_box_weight.shape) or current_score != tuple(det_score_weight.shape):
        model.det_head = _resize_detection_head(model.det_head, det_box_weight, det_score_weight)
    return model



def _maybe_resize_seg_head(model, model_state):
    seg_head_weight = model_state.get('seg_head.block.1.weight')
    if seg_head_weight is None or not hasattr(model, 'seg_head'):
        return model
    current_shape = tuple(model.seg_head.block[1].weight.shape)
    target_shape = tuple(seg_head_weight.shape)
    if current_shape != target_shape:
        model.seg_head = _resize_segmentation_head(model.seg_head, seg_head_weight)
    return model



def _rebuild_modules_from_state_shapes(model, model_state):
    model = _maybe_resize_det_neck(model, model_state)
    model = _maybe_resize_seg_neck(model, model_state)
    model = _maybe_resize_det_head(model, model_state)
    model = _maybe_resize_seg_head(model, model_state)
    return model



def load_checkpoint_and_update_config(config, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    checkpoint_config = checkpoint.get('config', {})
    updated_config = config.clone()
    updated_config.defrost()

    dataset_cfg = checkpoint_config.get('dataset', {})
    if dataset_cfg.get('class_names'):
        updated_config.dataset.class_names = list(dataset_cfg['class_names'])
    if dataset_cfg.get('n_classes') is not None:
        updated_config.dataset.n_classes = int(dataset_cfg['n_classes'])

    model_cfg = checkpoint_config.get('model', {})
    backbone_cfg = model_cfg.get('backbone', {})
    if backbone_cfg.get('channels'):
        updated_config.model.backbone.channels = list(backbone_cfg['channels'])

    updated_config.freeze()
    return checkpoint, updated_config



def create_model_from_checkpoint(config, checkpoint_path, create_model_fn):
    checkpoint, updated_config = load_checkpoint_and_update_config(config, checkpoint_path)
    model = create_model_fn(updated_config)
    model_state = checkpoint.get('model', checkpoint)
    model = _rebuild_modules_from_state_shapes(model, model_state)
    if hasattr(model, 'config'):
        model.config = updated_config
    model = model.to(updated_config.device)
    model.load_state_dict(model_state)
    model.to(updated_config.device)
    return model, updated_config, checkpoint
