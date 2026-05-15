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
                    prefixes: Iterable[str] = ('module.', 'model.')):
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
            if new_key.startswith(old):
                new_key = new + new_key[len(old):]
                break
        mapped[new_key] = value
        if new_key != key:
            renamed.append((key, new_key))
    return mapped, renamed



def _segmentation_rules():
    return [
        ('stem', 'encoder.input_proj'),
        ('encoder_blocks.0', 'encoder.blocks.0'),
        ('encoder_blocks.1', 'encoder.blocks.1'),
        ('encoder_blocks.2', 'encoder.blocks.2'),
        ('encoder_transitions.0', 'encoder.transitions.0'),
        ('encoder_transitions.1', 'encoder.transitions.1'),
        ('classifier.', 'seg_head.'),
        ('head.', 'seg_head.'),
    ]



def _filter_head_keys(state_dict):
    filtered = {}
    dropped = []
    for key, value in state_dict.items():
        if key.startswith('seg_head.'):
            dropped.append(key)
            continue
        filtered[key] = value
    return filtered, dropped



def _build_mapping_stats(original_state_dict, mapped_state_dict, renamed_pairs, dropped_classifier_keys):
    renamed_from_keys = {source for source, _ in renamed_pairs}
    renamed_to_keys = {target for _, target in renamed_pairs}
    dropped_key_set = set(dropped_classifier_keys)

    exact_match_keys = []
    unmatched_input_keys = []
    for key in original_state_dict.keys():
        if key in dropped_key_set:
            continue
        if key in renamed_from_keys:
            continue
        if key in mapped_state_dict:
            exact_match_keys.append(key)
            continue
        unmatched_input_keys.append(key)

    unmatched_categories = {
        'head_keys': [],
        'encoder_related': [],
        'kernel_related': [],
        'other': [],
    }
    for key in unmatched_input_keys:
        lowered = key.lower()
        if any(token in lowered for token in ['head', 'classifier', 'seg']):
            unmatched_categories['head_keys'].append(key)
        elif any(token in lowered for token in ['encoder', 'block', 'transition', 'stem']):
            unmatched_categories['encoder_related'].append(key)
        elif any(token in lowered for token in ['kernel', 'kpconv', 'neighbor', 'point']):
            unmatched_categories['kernel_related'].append(key)
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



def map_kpconv_seg_state_dict(state_dict, pretrained_source='official', reset_classifier=False):
    original_state_dict = dict(state_dict)
    state_dict = _strip_prefixes(state_dict)
    renamed_pairs: List[Tuple[str, str]] = []

    if pretrained_source == 'official':
        state_dict, renamed_pairs = _rename_keys(state_dict, _segmentation_rules())

    dropped_classifier_keys = []
    if reset_classifier:
        state_dict, dropped_classifier_keys = _filter_head_keys(state_dict)

    mapping_stats = _build_mapping_stats(
        original_state_dict,
        state_dict,
        renamed_pairs,
        dropped_classifier_keys,
    )
    return {
        'task': 'segmentation',
        'state_dict': state_dict,
        'renamed_pairs': renamed_pairs,
        'dropped_classifier_keys': dropped_classifier_keys,
        'mapping_stats': mapping_stats,
    }



def load_kpconv_seg_pretrained(model, config):
    kpconv_cfg = config.model.kpconv
    if not kpconv_cfg.pretrained:
        return

    checkpoint_path = pathlib.Path(kpconv_cfg.pretrained_path).expanduser()
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    raw_state_dict = _extract_state_dict(checkpoint)
    mapped = map_kpconv_seg_state_dict(
        raw_state_dict,
        pretrained_source=kpconv_cfg.pretrained_source,
        reset_classifier=kpconv_cfg.reset_classifier,
    )
    model_state_dict = model.state_dict()
    compatible_state_dict, shape_mismatches = _collect_shape_mismatches(
        model_state_dict,
        mapped['state_dict'],
    )

    missing, unexpected = model.load_state_dict(
        compatible_state_dict,
        strict=kpconv_cfg.strict_pretrained_load and not kpconv_cfg.reset_classifier and not shape_mismatches,
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



def diagnose_kpconv_state_dict_mapping(state_dict, reset_classifier=False,
                                       pretrained_source='official'):
    mapped = map_kpconv_seg_state_dict(
        state_dict,
        pretrained_source=pretrained_source,
        reset_classifier=reset_classifier,
    )
    return {
        'task': mapped['task'],
        'mapped_keys': sorted(mapped['state_dict'].keys()),
        'renamed_pairs': mapped['renamed_pairs'],
        'dropped_classifier_keys': mapped['dropped_classifier_keys'],
        'mapped_key_count': len(mapped['state_dict']),
        'mapping_stats': mapped['mapping_stats'],
    }
