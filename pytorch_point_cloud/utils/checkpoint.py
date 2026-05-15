import torch
import torch.nn as nn


def _resize_linear_from_weight(weight, bias=True):
    return nn.Linear(int(weight.shape[1]), int(weight.shape[0]), bias=bias)


def _resize_bn1d_from_weight(weight):
    return nn.BatchNorm1d(int(weight.shape[0]))


def _maybe_resize_pointnet_classifier(model, model_state):
    fc1_weight = model_state.get('fc1.weight')
    fc2_weight = model_state.get('fc2.weight')
    fc3_weight = model_state.get('fc3.weight')
    bn1_weight = model_state.get('bn1.weight')
    bn2_weight = model_state.get('bn2.weight')
    if fc1_weight is not None:
        current = tuple(model.fc1.weight.shape)
        target = tuple(fc1_weight.shape)
        if current != target:
            model.fc1 = _resize_linear_from_weight(fc1_weight, bias=model.fc1.bias is not None)
    if fc2_weight is not None:
        current = tuple(model.fc2.weight.shape)
        target = tuple(fc2_weight.shape)
        if current != target:
            model.fc2 = _resize_linear_from_weight(fc2_weight, bias=model.fc2.bias is not None)
    if fc3_weight is not None:
        current = tuple(model.fc3.weight.shape)
        target = tuple(fc3_weight.shape)
        if current != target:
            model.fc3 = _resize_linear_from_weight(fc3_weight, bias=model.fc3.bias is not None)
    if bn1_weight is not None:
        current = int(model.bn1.weight.shape[0])
        target = int(bn1_weight.shape[0])
        if current != target:
            model.bn1 = _resize_bn1d_from_weight(bn1_weight)
    if bn2_weight is not None:
        current = int(model.bn2.weight.shape[0])
        target = int(bn2_weight.shape[0])
        if current != target:
            model.bn2 = _resize_bn1d_from_weight(bn2_weight)
    return model


def _maybe_resize_dgcnn_backbone(model, model_state, config):
    fusion_weight = model_state.get('backbone.fusion.0.weight')
    fusion_bn_weight = model_state.get('backbone.fusion.1.weight')
    if fusion_weight is None or fusion_bn_weight is None:
        return model
    current_fusion = tuple(model.backbone.fusion[0].weight.shape)
    target_fusion = tuple(fusion_weight.shape)
    current_bn = int(model.backbone.fusion[1].weight.shape[0])
    target_bn = int(fusion_bn_weight.shape[0])
    if current_fusion == target_fusion and current_bn == target_bn:
        return model
    edge_dims = list(config.model.dgcnn.edge_dims)
    emb_dims = int(fusion_weight.shape[0])
    model.backbone = type(model.backbone)(config.dataset.n_channels, edge_dims, emb_dims, config.model.dgcnn.k)
    return model


def _maybe_resize_dgcnn_classifier(model, model_state, config):
    first_weight = model_state.get('classifier.0.weight')
    second_weight = model_state.get('classifier.4.weight')
    third_weight = model_state.get('classifier.8.weight')
    if first_weight is None or second_weight is None or third_weight is None:
        return model
    current_shapes = (
        tuple(model.classifier[0].weight.shape),
        tuple(model.classifier[4].weight.shape),
        tuple(model.classifier[8].weight.shape),
    )
    target_shapes = (
        tuple(first_weight.shape),
        tuple(second_weight.shape),
        tuple(third_weight.shape),
    )
    if current_shapes == target_shapes:
        return model
    dropout = config.model.dgcnn.dropout
    model.classifier = nn.Sequential(
        nn.Linear(int(first_weight.shape[1]), int(first_weight.shape[0]), bias=False),
        nn.BatchNorm1d(int(model_state['classifier.1.weight'].shape[0])),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Dropout(dropout),
        nn.Linear(int(second_weight.shape[1]), int(second_weight.shape[0]), bias=False),
        nn.BatchNorm1d(int(model_state['classifier.5.weight'].shape[0])),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Dropout(dropout),
        nn.Linear(int(third_weight.shape[1]), int(third_weight.shape[0]), bias=True),
    )
    return model


def _rebuild_modules_from_state_shapes(model, model_state, config):
    model_key = (config.model.type, config.model.name)
    if model_key == ('pointnet', 'pointnet_cls'):
        return _maybe_resize_pointnet_classifier(model, model_state)
    if model_key == ('dgcnn', 'dgcnn_cls'):
        model = _maybe_resize_dgcnn_backbone(model, model_state, config)
        return _maybe_resize_dgcnn_classifier(model, model_state, config)
    return model


def load_checkpoint_and_update_config(config, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    checkpoint_config = checkpoint.get('config', {})
    updated_config = config.clone()
    updated_config.defrost()

    dataset_cfg = checkpoint_config.get('dataset', {})
    if dataset_cfg.get('n_classes') is not None:
        updated_config.dataset.n_classes = int(dataset_cfg['n_classes'])

    updated_config.freeze()
    return checkpoint, updated_config


def create_model_from_checkpoint(config, checkpoint_path, create_model_fn):
    checkpoint, updated_config = load_checkpoint_and_update_config(config, checkpoint_path)
    model = create_model_fn(updated_config)
    model_state = checkpoint.get('model', checkpoint)
    prune_meta = checkpoint.get('prune', {})
    if bool(prune_meta.get('rebuilt', False)):
        model = _rebuild_modules_from_state_shapes(model, model_state, updated_config)
    if hasattr(model, 'config'):
        model.config = updated_config
    model = model.to(updated_config.device)
    model.load_state_dict(model_state)
    model.to(updated_config.device)
    return model, updated_config, checkpoint
