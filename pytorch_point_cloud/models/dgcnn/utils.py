from typing import Dict, Iterable, List, Tuple

import pathlib
import torch



def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ['model', 'state_dict', 'net', 'module']:
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    return checkpoint



def _strip_prefixes(state_dict: Dict[str, torch.Tensor],
                    prefixes: Iterable[str] = ('module.', 'model.', 'backbone.')):
    cleaned = {}
    for key, value in state_dict.items():
        new_key = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
                    changed = True
        cleaned[new_key] = value
    return cleaned



def _rename_keys(state_dict: Dict[str, torch.Tensor], rules: Iterable[Tuple[str, str]]):
    mapped = {}
    renamed = []
    for key, value in state_dict.items():
        new_key = key
        for old, new in rules:
            if old in new_key:
                new_key = new_key.replace(old, new)
        mapped[new_key] = value
        if new_key != key:
            renamed.append((key, new_key))
    return mapped, renamed



def _classification_rules():
    return [
        ('transform_net', 'input_transform'),
        ('edge_convs.0', 'backbone.blocks.0.net'),
        ('edge_convs.1', 'backbone.blocks.1.net'),
        ('edge_convs.2', 'backbone.blocks.2.net'),
        ('edge_convs.3', 'backbone.blocks.3.net'),
        ('conv1', 'backbone.blocks.0.net.0'),
        ('conv2', 'backbone.blocks.1.net.0'),
        ('conv3', 'backbone.blocks.2.net.0'),
        ('conv4', 'backbone.blocks.3.net.0'),
        ('conv5', 'backbone.fusion.0'),
        ('bn1', 'backbone.blocks.0.net.1'),
        ('bn2', 'backbone.blocks.1.net.1'),
        ('bn3', 'backbone.blocks.2.net.1'),
        ('bn4', 'backbone.blocks.3.net.1'),
        ('bn5', 'backbone.fusion.1'),
        ('linear1', 'classifier.0'),
        ('linear2', 'classifier.3'),
        ('linear3', 'classifier.6'),
        ('bn6', 'classifier.1'),
        ('bn7', 'classifier.4'),
        ('dp1', 'classifier.2'),
        ('dp2', 'classifier.5'),
    ]



def _segmentation_rules():
    return [
        ('edge_convs.0', 'backbone.blocks.0.net'),
        ('edge_convs.1', 'backbone.blocks.1.net'),
        ('edge_convs.2', 'backbone.blocks.2.net'),
        ('conv1', 'backbone.blocks.0.net.0'),
        ('conv2', 'backbone.blocks.1.net.0'),
        ('conv3', 'backbone.blocks.2.net.0'),
        ('conv6', 'backbone.fusion.0'),
        ('bn1', 'backbone.blocks.0.net.1'),
        ('bn2', 'backbone.blocks.1.net.1'),
        ('bn3', 'backbone.blocks.2.net.1'),
        ('bn6', 'backbone.fusion.1'),
        ('conv7', 'seg_head.0'),
        ('conv8', 'seg_head.3'),
        ('conv9', 'seg_head.6'),
        ('bn7', 'seg_head.1'),
        ('bn8', 'seg_head.4'),
        ('dp1', 'seg_head.2'),
        ('dp2', 'seg_head.5'),
    ]



def _filter_head_keys(state_dict, task):
    prefixes = ('classifier.',) if task == 'classification' else ('seg_head.',)
    filtered = {}
    dropped = []
    for key, value in state_dict.items():
        if key.startswith(prefixes):
            dropped.append(key)
            continue
        filtered[key] = value
    return filtered, dropped



def _build_mapping_stats(original_state_dict, mapped_state_dict, renamed_pairs, dropped_classifier_keys):
    renamed_from_keys = {source for source, _ in renamed_pairs}
    renamed_to_keys = {target for _, target in renamed_pairs}
    dropped_key_set = set(dropped_classifier_keys)

    exact_match_keys = []
    renamed_match_keys = []
    unmatched_input_keys = []
    for key in original_state_dict.keys():
        if key in dropped_key_set:
            continue
        if key in renamed_from_keys:
            renamed_match_keys.append(key)
            continue
        if key in mapped_state_dict:
            exact_match_keys.append(key)
            continue
        unmatched_input_keys.append(key)

    unmatched_categories = {
        'head_keys': [],
        'edgeconv_related': [],
        'fusion_related': [],
        'other': [],
    }
    for key in unmatched_input_keys:
        lowered = key.lower()
        if any(token in lowered for token in ['classifier', 'seg_head', 'head', 'linear3', 'conv9']):
            unmatched_categories['head_keys'].append(key)
        elif any(token in lowered for token in ['edge', 'graph', 'conv1', 'conv2', 'conv3', 'conv4']):
            unmatched_categories['edgeconv_related'].append(key)
        elif any(token in lowered for token in ['conv5', 'conv6', 'fusion', 'emb']):
            unmatched_categories['fusion_related'].append(key)
        else:
            unmatched_categories['other'].append(key)

    return {
        'input_key_count': len(original_state_dict),
        'mapped_key_count': len(mapped_state_dict),
        'exact_match_key_count': len(exact_match_keys),
        'renamed_key_count': len(renamed_pairs),
        'dropped_key_count': len(dropped_classifier_keys),
        'unmatched_input_key_count': len(unmatched_input_keys),
        'exact_match_keys': exact_match_keys,
        'renamed_target_keys': sorted(renamed_to_keys),
        'unmatched_input_keys': unmatched_input_keys,
        'unmatched_input_key_categories': unmatched_categories,
    }



def _collect_shape_mismatches(model_state_dict, incoming_state_dict):
    mismatches = []
    compatible_state_dict = {}
    for key, value in incoming_state_dict.items():
        if key not in model_state_dict:
            compatible_state_dict[key] = value
            continue
        if model_state_dict[key].shape != value.shape:
            mismatches.append({
                'key': key,
                'model_shape': tuple(model_state_dict[key].shape),
                'checkpoint_shape': tuple(value.shape),
            })
            continue
        compatible_state_dict[key] = value
    return compatible_state_dict, mismatches



def _map_dgcnn_state_dict(state_dict, pretrained_source, reset_classifier, task, rules):
    original_state_dict = dict(state_dict)
    state_dict = _strip_prefixes(state_dict)
    renamed_pairs: List[Tuple[str, str]] = []

    if pretrained_source == 'official':
        state_dict, renamed_pairs = _rename_keys(state_dict, rules)

    dropped_classifier_keys = []
    if reset_classifier:
        state_dict, dropped_classifier_keys = _filter_head_keys(state_dict, task)

    mapping_stats = _build_mapping_stats(
        original_state_dict,
        state_dict,
        renamed_pairs,
        dropped_classifier_keys,
    )
    return {
        'task': task,
        'state_dict': state_dict,
        'renamed_pairs': renamed_pairs,
        'dropped_classifier_keys': dropped_classifier_keys,
        'mapping_stats': mapping_stats,
    }



def map_dgcnn_cls_state_dict(state_dict, pretrained_source='official', reset_classifier=False):
    return _map_dgcnn_state_dict(
        state_dict, pretrained_source, reset_classifier, 'classification', _classification_rules())



def map_dgcnn_seg_state_dict(state_dict, pretrained_source='official', reset_classifier=False):
    return _map_dgcnn_state_dict(
        state_dict, pretrained_source, reset_classifier, 'segmentation', _segmentation_rules())



def _load_mapped_state_dict(model, config, mapper):
    dgcnn_cfg = config.model.dgcnn
    if not dgcnn_cfg.pretrained:
        return

    checkpoint_path = pathlib.Path(dgcnn_cfg.pretrained_path).expanduser()
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    raw_state_dict = _extract_state_dict(checkpoint)
    mapped = mapper(
        raw_state_dict,
        pretrained_source=dgcnn_cfg.pretrained_source,
        reset_classifier=dgcnn_cfg.reset_classifier,
    )
    model_state_dict = model.state_dict()
    compatible_state_dict, shape_mismatches = _collect_shape_mismatches(
        model_state_dict,
        mapped['state_dict'],
    )

    missing, unexpected = model.load_state_dict(
        compatible_state_dict,
        strict=dgcnn_cfg.strict_pretrained_load and not dgcnn_cfg.reset_classifier and not shape_mismatches,
    )
    return {
        'task': mapped['task'],
        'missing_keys': list(missing),
        'unexpected_keys': list(unexpected),
        'shape_mismatches': shape_mismatches,
        'renamed_pairs': mapped['renamed_pairs'],
        'dropped_classifier_keys': mapped['dropped_classifier_keys'],
        'loaded_key_count': len(compatible_state_dict),
        'mapping_stats': mapped['mapping_stats'],
    }



def load_dgcnn_cls_pretrained(model, config):
    return _load_mapped_state_dict(model, config, map_dgcnn_cls_state_dict)



def load_dgcnn_seg_pretrained(model, config):
    return _load_mapped_state_dict(model, config, map_dgcnn_seg_state_dict)



def diagnose_dgcnn_state_dict_mapping(task, state_dict, reset_classifier=False,
                                      pretrained_source='official'):
    if task == 'classification':
        mapped = map_dgcnn_cls_state_dict(
            state_dict,
            pretrained_source=pretrained_source,
            reset_classifier=reset_classifier,
        )
    elif task == 'segmentation':
        mapped = map_dgcnn_seg_state_dict(
            state_dict,
            pretrained_source=pretrained_source,
            reset_classifier=reset_classifier,
        )
    else:
        raise ValueError(f'Unsupported DGCNN task for mapping diagnosis: {task}')

    return {
        'task': mapped['task'],
        'mapped_keys': sorted(mapped['state_dict'].keys()),
        'renamed_pairs': mapped['renamed_pairs'],
        'dropped_classifier_keys': mapped['dropped_classifier_keys'],
        'mapped_key_count': len(mapped['state_dict']),
        'mapping_stats': mapped['mapping_stats'],
    }
