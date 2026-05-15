import pathlib
from typing import Dict, Iterable, List, Tuple

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



def _shared_stage_rules():
    return [
        ('q_proj', 'to_query'),
        ('query_proj', 'to_query'),
        ('k_proj', 'to_key'),
        ('key_proj', 'to_key'),
        ('v_proj', 'to_value'),
        ('value_proj', 'to_value'),
        ('pos_mlp.net', 'position_encoding.net'),
        ('pos_mlp', 'position_encoding'),
        ('position_mlp.net', 'position_encoding.net'),
        ('position_mlp', 'position_encoding'),
        ('attn_mlp', 'attention_mlp'),
        ('attention_proj', 'attention_mlp'),
        ('out_proj', 'output_proj'),
        ('output_linear', 'output_proj'),
        ('proj', 'feature_proj'),
        ('down_proj', 'feature_proj'),
        ('linear1', 'upsample_proj'),
        ('up_proj', 'upsample_proj'),
        ('linear2', 'skip_proj'),
        ('skip_linear', 'skip_proj'),
        ('fuse', 'fusion'),
        ('fusion_mlp', 'fusion'),
    ]



def _v1_rules():
    return [
        ('patch_embed', 'input_proj'),
        ('fc1', 'net.1'),
        ('fc2', 'net.4'),
        ('transformer1', 'encoder_stages.0.blocks'),
        ('transformer2', 'encoder_stages.1.blocks'),
        ('transformer3', 'encoder_stages.2.blocks'),
        ('linear_q', 'to_query'),
        ('linear_k', 'to_key'),
        ('linear_v', 'to_value'),
        ('pos_mlp', 'position_encoding.net'),
        ('attn_mlp', 'attention_mlp'),
        ('proj', 'output_proj'),
        ('down1.linear', 'downsamples.0.feature_proj'),
        ('down2.linear', 'downsamples.1.feature_proj'),
        ('down1.pos_mlp', 'downsamples.0.position_encoding.net'),
        ('down2.pos_mlp', 'downsamples.1.position_encoding.net'),
        ('down1.fusion', 'downsamples.0.fusion'),
        ('down2.fusion', 'downsamples.1.fusion'),
        ('down1.norm', 'downsamples.0.norm'),
        ('down2.norm', 'downsamples.1.norm'),
        ('down1', 'downsamples.0'),
        ('down2', 'downsamples.1'),
        ('cls_head', 'head.net'),
        ('head', 'head.net'),
    ]



def _v2_rules():
    return _shared_stage_rules() + [
        ('encoders', 'encoder_stages'),
        ('encoder', 'encoder_stages'),
        ('downs', 'downsamples'),
        ('downsamples', 'downsamples'),
        ('decoders', 'decoder_stages'),
        ('decoder', 'decoder_stages'),
        ('decoder_blocks', 'decoder_stages'),
        ('patch_embed', 'input_proj'),
        ('input_embed', 'input_proj'),
        ('encoder_depths', 'encoder_depths'),
        ('decoder_depths', 'decoder_depths'),
        ('seg_head', 'head.head'),
        ('segmentation_head', 'head.head'),
        ('classifier', 'head.head'),
        ('head', 'head.head'),
    ]



def _v3_rules():
    return _shared_stage_rules() + [
        ('encoders', 'encoder_stages'),
        ('encoder', 'encoder_stages'),
        ('downs', 'downsamples'),
        ('downsamples', 'downsamples'),
        ('decoders', 'decoder_stages'),
        ('decoder', 'decoder_stages'),
        ('decoder_blocks', 'decoder_stages'),
        ('patch_embed', 'input_proj'),
        ('input_embed', 'input_proj'),
        ('encoder_depths', 'encoder_depths'),
        ('decoder_depths', 'decoder_depths'),
        ('seg_head', 'head.head'),
        ('segmentation_head', 'head.head'),
        ('classifier', 'head.head'),
        ('head', 'head.head'),
    ]



def _filter_head_keys(state_dict):
    filtered = {}
    dropped = []
    for key, value in state_dict.items():
        if key.startswith('head.'):
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
        'attention_related': [],
        'transition_related': [],
        'encoder_decoder_related': [],
        'other': [],
    }
    for key in unmatched_input_keys:
        lowered = key.lower()
        if 'head' in lowered or 'classifier' in lowered or 'seg' in lowered:
            unmatched_categories['head_keys'].append(key)
        elif any(token in lowered for token in ['query', 'key', 'value', 'attn', 'attention', 'q_proj', 'k_proj', 'v_proj']):
            unmatched_categories['attention_related'].append(key)
        elif any(token in lowered for token in ['down', 'up', 'skip', 'fuse', 'fusion', 'transition']):
            unmatched_categories['transition_related'].append(key)
        elif any(token in lowered for token in ['encoder', 'decoder', 'stage', 'block']):
            unmatched_categories['encoder_decoder_related'].append(key)
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



def _map_pointtransformer_state_dict(state_dict, pretrained_source, reset_classifier, rules, version_tag):
    original_state_dict = dict(state_dict)
    state_dict = _strip_prefixes(state_dict)
    renamed_pairs: List[Tuple[str, str]] = []

    if pretrained_source == 'official':
        state_dict, renamed_pairs = _rename_keys(state_dict, rules)

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
        'version_tag': version_tag,
        'state_dict': state_dict,
        'renamed_pairs': renamed_pairs,
        'dropped_classifier_keys': dropped_classifier_keys,
        'mapping_stats': mapping_stats,
    }



def map_pointtransformer_v1_state_dict(state_dict, pretrained_source='official', reset_classifier=False):
    return _map_pointtransformer_state_dict(
        state_dict, pretrained_source, reset_classifier, _v1_rules(), 'v1')



def map_pointtransformer_v2_state_dict(state_dict, pretrained_source='official', reset_classifier=False):
    return _map_pointtransformer_state_dict(
        state_dict, pretrained_source, reset_classifier, _v2_rules(), 'v2')



def map_pointtransformer_v3_state_dict(state_dict, pretrained_source='official', reset_classifier=False):
    return _map_pointtransformer_state_dict(
        state_dict, pretrained_source, reset_classifier, _v3_rules(), 'v3')



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



def _load_mapped_state_dict(model, config, mapper):
    pt_cfg = config.model.pointtransformer
    if not pt_cfg.pretrained:
        return

    checkpoint_path = pathlib.Path(pt_cfg.pretrained_path).expanduser()
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    raw_state_dict = _extract_state_dict(checkpoint)
    mapped = mapper(
        raw_state_dict,
        pretrained_source=pt_cfg.pretrained_source,
        reset_classifier=pt_cfg.reset_classifier,
    )
    model_state_dict = model.state_dict()
    compatible_state_dict, shape_mismatches = _collect_shape_mismatches(
        model_state_dict,
        mapped['state_dict'],
    )

    missing, unexpected = model.load_state_dict(
        compatible_state_dict,
        strict=pt_cfg.strict_pretrained_load and not pt_cfg.reset_classifier and not shape_mismatches,
    )
    return {
        'version_tag': mapped['version_tag'],
        'missing_keys': list(missing),
        'unexpected_keys': list(unexpected),
        'shape_mismatches': shape_mismatches,
        'renamed_pairs': mapped['renamed_pairs'],
        'dropped_classifier_keys': mapped['dropped_classifier_keys'],
        'loaded_key_count': len(compatible_state_dict),
        'mapping_stats': mapped['mapping_stats'],
    }



def load_pointtransformer_v1_pretrained(model, config):
    return _load_mapped_state_dict(model, config, map_pointtransformer_v1_state_dict)



def load_pointtransformer_v2_pretrained(model, config):
    return _load_mapped_state_dict(model, config, map_pointtransformer_v2_state_dict)



def load_pointtransformer_v3_pretrained(model, config):
    return _load_mapped_state_dict(model, config, map_pointtransformer_v3_state_dict)





def diagnose_pointtransformer_state_dict_mapping(version, state_dict, reset_classifier=False,
                                                 pretrained_source='official'):
    if version == 'v1':
        mapped = map_pointtransformer_v1_state_dict(
            state_dict,
            pretrained_source=pretrained_source,
            reset_classifier=reset_classifier,
        )
    elif version == 'v2':
        mapped = map_pointtransformer_v2_state_dict(
            state_dict,
            pretrained_source=pretrained_source,
            reset_classifier=reset_classifier,
        )
    elif version == 'v3':
        mapped = map_pointtransformer_v3_state_dict(
            state_dict,
            pretrained_source=pretrained_source,
            reset_classifier=reset_classifier,
        )
    else:
        raise ValueError(f'Unsupported version for mapping diagnosis: {version}')

    return {
        'version_tag': mapped['version_tag'],
        'mapped_keys': sorted(mapped['state_dict'].keys()),
        'renamed_pairs': mapped['renamed_pairs'],
        'dropped_classifier_keys': mapped['dropped_classifier_keys'],
        'mapped_key_count': len(mapped['state_dict']),
        'mapping_stats': mapped['mapping_stats'],
    }
