#!/usr/bin/env python

import argparse
import importlib
import json
import pathlib

import torch
import torch.nn as nn

from pytorch_point_cloud.config import get_default_config, update_config
from pytorch_point_cloud.models.pointbert.utils import diagnose_pointbert_state_dict_mapping
from pytorch_point_cloud.models.dgcnn.utils import diagnose_dgcnn_state_dict_mapping
from pytorch_point_cloud.models.kpconv.common import KPConvEncoder
from pytorch_point_cloud.models.kpconv.utils import diagnose_kpconv_state_dict_mapping
from pytorch_point_cloud.models.pointbert.utils import diagnose_pointbert_state_dict_mapping
from pytorch_point_cloud.models.pointmae.utils import diagnose_pointmae_state_dict_mapping
from pytorch_point_cloud.models.pointtransformer.pointtransformer_utils import (
    _extract_state_dict,
    diagnose_pointtransformer_state_dict_mapping,
)
from pytorch_point_cloud.models.pointvoxel.common import PointVoxelBackbone
from pytorch_point_cloud.models.pointvoxel.utils import diagnose_pointvoxel_state_dict_mapping
from pytorch_point_cloud.models.pvcnn.common import PVCNNBackbone
from pytorch_point_cloud.models.pvcnn.utils import diagnose_pvcnn_state_dict_mapping



def load_args():
    parser = argparse.ArgumentParser(
        description='Diagnose point-cloud checkpoint key mapping.')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config yaml file.')
    parser.add_argument('--checkpoint', type=str, default='',
                        help='Path to checkpoint file to inspect.')
    parser.add_argument('--dry-run', action='store_true',
                        help='Use structure-derived synthetic state_dict and run shape-aware diagnosis.')
    parser.add_argument('--version', type=str, default='', choices=['', 'v1', 'v2', 'v3'],
                        help='Override PointTransformer version from config.')
    parser.add_argument('--task', type=str, default='', choices=['', 'classification', 'segmentation'],
                        help='Override task for models that need it, such as DGCNN.')
    parser.add_argument('--pretrained-source', type=str, default='',
                        help='Override pretrained source, e.g. official or native.')
    parser.add_argument('--reset-classifier', action='store_true',
                        help='Force dropping task head keys during diagnosis.')
    parser.add_argument('--no-reset-classifier', action='store_true',
                        help='Force keeping task head keys during diagnosis.')
    parser.add_argument('--show-keys-limit', type=int, default=20,
                        help='Maximum number of keys to print per detailed section.')
    parser.add_argument('--json', action='store_true',
                        help='Print full diagnosis result as JSON.')
    parser.add_argument('options', default=None, nargs=argparse.REMAINDER,
                        help='Additional config overrides in KEY VALUE form.')
    return parser.parse_args()



def load_config(args):
    config = get_default_config()
    config.merge_from_file(args.config)
    config.merge_from_list(args.options)
    config = update_config(config)
    config.freeze()
    return config



def resolve_common_options(config, args):
    if args.reset_classifier and args.no_reset_classifier:
        raise ValueError('Cannot set both --reset-classifier and --no-reset-classifier.')

    model_type = config.model.type
    task = args.task or config.task

    if model_type == 'pointtransformer':
        family_cfg = config.model.pointtransformer
        version = args.version or family_cfg.version
    else:
        family_cfg = getattr(config.model, model_type)
        version = args.version or ''

    pretrained_source = args.pretrained_source or family_cfg.pretrained_source
    if args.reset_classifier:
        reset_classifier = True
    elif args.no_reset_classifier:
        reset_classifier = False
    else:
        reset_classifier = family_cfg.reset_classifier

    return {
        'model_type': model_type,
        'model_name': config.model.name,
        'task': task,
        'version': version,
        'pretrained_source': pretrained_source,
        'reset_classifier': reset_classifier,
        'dry_run': args.dry_run,
    }



def instantiate_model_from_config(config):
    module_name = f'pytorch_point_cloud.models.{config.model.type}.{config.model.name}'
    module = importlib.import_module(module_name)
    model = module.Network(config)
    model.eval()
    return model



def _build_kpconv_structure_model(config):
    kpconv_cfg = config.model.kpconv
    encoder_dims = list(kpconv_cfg.encoder_dims)
    encoder = KPConvEncoder(
        config.dataset.n_channels,
        encoder_dims,
        kpconv_cfg.k,
        kpconv_cfg.kernel_points,
        kpconv_cfg.sigma,
    )
    seg_input_dim = sum(encoder_dims) + encoder_dims[-1]
    seg_head = nn.Sequential(
        nn.Conv1d(seg_input_dim, kpconv_cfg.decoder_dim, kernel_size=1, bias=False),
        nn.BatchNorm1d(kpconv_cfg.decoder_dim),
        nn.ReLU(inplace=True),
        nn.Dropout(kpconv_cfg.dropout),
        nn.Conv1d(kpconv_cfg.decoder_dim, kpconv_cfg.decoder_dim, kernel_size=1, bias=False),
        nn.BatchNorm1d(kpconv_cfg.decoder_dim),
        nn.ReLU(inplace=True),
        nn.Dropout(kpconv_cfg.dropout),
        nn.Conv1d(kpconv_cfg.decoder_dim, config.dataset.n_seg_classes, kernel_size=1),
    )
    return nn.ModuleDict({'encoder': encoder, 'seg_head': seg_head})



def _build_pvcnn_structure_model(config):
    pvcnn_cfg = config.model.pvcnn
    block_channels = list(pvcnn_cfg.block_channels)
    backbone = PVCNNBackbone(
        config.dataset.n_channels,
        block_channels,
        pvcnn_cfg.voxel_resolution,
    )
    seg_input_dim = sum(block_channels) + block_channels[-1]
    seg_head = nn.Sequential(
        nn.Conv1d(seg_input_dim, pvcnn_cfg.decoder_dim, kernel_size=1, bias=False),
        nn.BatchNorm1d(pvcnn_cfg.decoder_dim),
        nn.ReLU(inplace=True),
        nn.Dropout(pvcnn_cfg.dropout),
        nn.Conv1d(pvcnn_cfg.decoder_dim, pvcnn_cfg.decoder_dim, kernel_size=1, bias=False),
        nn.BatchNorm1d(pvcnn_cfg.decoder_dim),
        nn.ReLU(inplace=True),
        nn.Dropout(pvcnn_cfg.dropout),
        nn.Conv1d(pvcnn_cfg.decoder_dim, config.dataset.n_seg_classes, kernel_size=1),
    )
    return nn.ModuleDict({'backbone': backbone, 'seg_head': seg_head})



def _build_pointvoxel_structure_model(config):
    pv_cfg = config.model.pointvoxel
    block_channels = list(pv_cfg.block_channels)
    backbone = PointVoxelBackbone(
        config.dataset.n_channels,
        block_channels,
        pv_cfg.voxel_resolution,
    )
    seg_input_dim = sum(block_channels) + block_channels[-1]
    seg_head = nn.Sequential(
        nn.Conv1d(seg_input_dim, pv_cfg.decoder_dim, kernel_size=1, bias=False),
        nn.BatchNorm1d(pv_cfg.decoder_dim),
        nn.ReLU(inplace=True),
        nn.Dropout(pv_cfg.dropout),
        nn.Conv1d(pv_cfg.decoder_dim, pv_cfg.decoder_dim, kernel_size=1, bias=False),
        nn.BatchNorm1d(pv_cfg.decoder_dim),
        nn.ReLU(inplace=True),
        nn.Dropout(pv_cfg.dropout),
        nn.Conv1d(pv_cfg.decoder_dim, config.dataset.n_seg_classes, kernel_size=1),
    )
    return nn.ModuleDict({'backbone': backbone, 'seg_head': seg_head})



def build_structure_model(config, options):
    if options['model_type'] == 'kpconv':
        return _build_kpconv_structure_model(config)
    if options['model_type'] == 'pvcnn':
        return _build_pvcnn_structure_model(config)
    if options['model_type'] == 'pointvoxel':
        return _build_pointvoxel_structure_model(config)
    return instantiate_model_from_config(config)



def synthesize_state_dict(config, options):
    try:
        model = build_structure_model(config, options)
        return model.state_dict(), model, 'structure'
    except Exception:
        model = instantiate_model_from_config(config)
        return model.state_dict(), model, 'initialized'



def load_or_synthesize_state_dict(args, config, options):
    if args.dry_run:
        state_dict, model, dry_run_source = synthesize_state_dict(config, options)
        return state_dict, None, model, dry_run_source

    if not args.checkpoint:
        raise ValueError('--checkpoint is required unless --dry-run is set.')

    checkpoint_path = pathlib.Path(args.checkpoint).expanduser()
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = _extract_state_dict(checkpoint)
    return state_dict, checkpoint_path, None, ''



def dispatch_diagnosis(config, state_dict, options):
    model_type = options['model_type']
    if model_type == 'pointtransformer':
        diagnosis = diagnose_pointtransformer_state_dict_mapping(
            options['version'],
            state_dict,
            reset_classifier=options['reset_classifier'],
            pretrained_source=options['pretrained_source'],
        )
        diagnosis_task = config.task
    elif model_type == 'dgcnn':
        diagnosis = diagnose_dgcnn_state_dict_mapping(
            options['task'],
            state_dict,
            reset_classifier=options['reset_classifier'],
            pretrained_source=options['pretrained_source'],
        )
        diagnosis_task = diagnosis['task']
    elif model_type == 'pointbert':
        diagnosis = diagnose_pointbert_state_dict_mapping(
            state_dict,
            reset_classifier=options['reset_classifier'],
            pretrained_source=options['pretrained_source'],
        )
        diagnosis_task = diagnosis['task']
    elif model_type == 'pointmae':
        diagnosis = diagnose_pointmae_state_dict_mapping(
            state_dict,
            reset_classifier=options['reset_classifier'],
            pretrained_source=options['pretrained_source'],
        )
        diagnosis_task = diagnosis['task']
    elif model_type == 'kpconv':
        diagnosis = diagnose_kpconv_state_dict_mapping(
            state_dict,
            reset_classifier=options['reset_classifier'],
            pretrained_source=options['pretrained_source'],
        )
        diagnosis_task = diagnosis['task']
    elif model_type == 'pvcnn':
        diagnosis = diagnose_pvcnn_state_dict_mapping(
            state_dict,
            reset_classifier=options['reset_classifier'],
            pretrained_source=options['pretrained_source'],
        )
        diagnosis_task = diagnosis['task']
    elif model_type == 'pointvoxel':
        diagnosis = diagnose_pointvoxel_state_dict_mapping(
            state_dict,
            reset_classifier=options['reset_classifier'],
            pretrained_source=options['pretrained_source'],
        )
        diagnosis_task = diagnosis['task']
    else:
        raise ValueError(f'Unsupported model type for diagnosis: {model_type}')

    return diagnosis, diagnosis_task



def analyze_model_compatibility(model_state_dict, mapped_keys):
    mapped_key_set = set(mapped_keys)
    model_key_set = set(model_state_dict.keys())

    loadable_keys = sorted(model_key_set & mapped_key_set)
    missing_model_keys = sorted(model_key_set - mapped_key_set)
    unexpected_mapped_keys = sorted(mapped_key_set - model_key_set)

    return {
        'enabled': True,
        'model_key_count': len(model_key_set),
        'mapped_key_count': len(mapped_key_set),
        'loadable_key_count': len(loadable_keys),
        'missing_model_key_count': len(missing_model_keys),
        'unexpected_mapped_key_count': len(unexpected_mapped_keys),
        'shape_mismatch_count': 0,
        'missing_model_keys': missing_model_keys,
        'unexpected_mapped_keys': unexpected_mapped_keys,
        'shape_mismatches': [],
    }



def build_report(args, diagnosis, checkpoint_path, options, diagnosis_task,
                 dry_run_summary=None, dry_run_source=''):
    version = diagnosis.get('version_tag', options['version'])
    report = {
        'basic_info': {
            'config': pathlib.Path(args.config).expanduser().as_posix(),
            'checkpoint': checkpoint_path.as_posix() if checkpoint_path else '',
            'model_type': options['model_type'],
            'model_name': options['model_name'],
            'task': diagnosis_task,
            'version': version,
            'pretrained_source': options['pretrained_source'],
            'reset_classifier': options['reset_classifier'],
            'dry_run': options['dry_run'],
            'dry_run_source': dry_run_source,
            'show_keys_limit': args.show_keys_limit,
        },
        'mapping_summary': diagnosis['mapping_stats'],
        'renamed_pairs': [
            {'source': source, 'target': target}
            for source, target in diagnosis['renamed_pairs']
        ],
        'dropped_classifier_keys': diagnosis['dropped_classifier_keys'],
        'mapped_keys': diagnosis['mapped_keys'],
    }

    if dry_run_summary is not None:
        report['dry_run_summary'] = {
            key: value for key, value in dry_run_summary.items()
            if key not in ['missing_model_keys', 'unexpected_mapped_keys', 'shape_mismatches']
        }
        report['missing_model_keys'] = dry_run_summary['missing_model_keys']
        report['unexpected_mapped_keys'] = dry_run_summary['unexpected_mapped_keys']
        report['shape_mismatches'] = dry_run_summary['shape_mismatches']

    return report



def print_section(title):
    print(f'\n[{title}]')



def print_kv(key, value):
    print(f'{key}: {value}')



def print_list_section(title, values, limit):
    print_section(title)
    if not values:
        print('(none)')
        return

    shown = values[:limit]
    for item in shown:
        print(f'- {item}')
    remaining = len(values) - len(shown)
    if remaining > 0:
        print(f'... ({remaining} more)')



def print_renamed_pairs(renamed_pairs, limit):
    print_section('renamed_pairs')
    if not renamed_pairs:
        print('(none)')
        return

    shown = renamed_pairs[:limit]
    for source, target in shown:
        print(f'- {source} -> {target}')
    remaining = len(renamed_pairs) - len(shown)
    if remaining > 0:
        print(f'... ({remaining} more)')



def print_shape_mismatches(shape_mismatches, limit):
    print_section('shape_mismatches')
    if not shape_mismatches:
        print('(none)')
        return

    shown = shape_mismatches[:limit]
    for item in shown:
        print(f"- {item['key']}: model={item['model_shape']} checkpoint={item['checkpoint_shape']}")
    remaining = len(shape_mismatches) - len(shown)
    if remaining > 0:
        print(f'... ({remaining} more)')



def print_text_report(report):
    mapping_stats = report['mapping_summary']
    unmatched_categories = mapping_stats['unmatched_input_key_categories']
    basic_info = report['basic_info']

    print_section('basic_info')
    print_kv('config', basic_info['config'])
    print_kv('checkpoint', basic_info['checkpoint'])
    print_kv('model_type', basic_info['model_type'])
    print_kv('model_name', basic_info['model_name'])
    print_kv('task', basic_info['task'])
    print_kv('version', basic_info['version'])
    print_kv('pretrained_source', basic_info['pretrained_source'])
    print_kv('reset_classifier', basic_info['reset_classifier'])
    print_kv('dry_run', basic_info['dry_run'])
    print_kv('dry_run_source', basic_info['dry_run_source'])
    print_kv('show_keys_limit', basic_info['show_keys_limit'])

    print_section('mapping_summary')
    print_kv('input_key_count', mapping_stats['input_key_count'])
    print_kv('mapped_key_count', mapping_stats['mapped_key_count'])
    print_kv('exact_match_key_count', mapping_stats['exact_match_key_count'])
    print_kv('renamed_key_count', mapping_stats['renamed_key_count'])
    print_kv('dropped_key_count', mapping_stats['dropped_key_count'])
    print_kv('unmatched_input_key_count', mapping_stats['unmatched_input_key_count'])

    if 'dry_run_summary' in report:
        print_section('dry_run_summary')
        for key, value in report['dry_run_summary'].items():
            print_kv(key, value)

    print_section('unmatched_category_counts')
    for category_name, keys in unmatched_categories.items():
        print_kv(category_name, len(keys))

    renamed_pairs = [(item['source'], item['target']) for item in report['renamed_pairs']]
    print_renamed_pairs(renamed_pairs, basic_info['show_keys_limit'])
    print_list_section('dropped_classifier_keys', report['dropped_classifier_keys'], basic_info['show_keys_limit'])
    print_list_section('unmatched_input_keys', mapping_stats['unmatched_input_keys'], basic_info['show_keys_limit'])

    for category_name, keys in unmatched_categories.items():
        print_list_section(f'unmatched_{category_name}', keys, basic_info['show_keys_limit'])

    if 'dry_run_summary' in report:
        print_list_section('missing_model_keys', report['missing_model_keys'], basic_info['show_keys_limit'])
        print_list_section('unexpected_mapped_keys', report['unexpected_mapped_keys'], basic_info['show_keys_limit'])
        print_shape_mismatches(report['shape_mismatches'], basic_info['show_keys_limit'])



def main():
    args = load_args()
    config = load_config(args)
    options = resolve_common_options(config, args)

    state_dict, checkpoint_path, model, dry_run_source = load_or_synthesize_state_dict(args, config, options)
    diagnosis, diagnosis_task = dispatch_diagnosis(config, state_dict, options)

    dry_run_summary = None
    if args.dry_run:
        dry_run_summary = analyze_model_compatibility(model.state_dict(), diagnosis['mapped_keys'])

    report = build_report(
        args,
        diagnosis,
        checkpoint_path,
        options,
        diagnosis_task,
        dry_run_summary,
        dry_run_source,
    )

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return

    print_text_report(report)


if __name__ == '__main__':
    main()
