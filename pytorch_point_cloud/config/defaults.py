from pytorch_image_classification.config.config_node import ConfigNode

config = ConfigNode()
config.task = 'classification'
config.device = 'cuda'

config.cudnn = ConfigNode()
config.cudnn.benchmark = True
config.cudnn.deterministic = False

config.dataset = ConfigNode()
config.dataset.name = 'ModelNet40'
config.dataset.dataset_dir = '~/datasets'
config.dataset.num_points = 1024
config.dataset.n_channels = 3
config.dataset.n_classes = 40
config.dataset.n_seg_classes = 50
config.dataset.use_normals = False

config.model = ConfigNode()
config.model.type = 'pointnet'
config.model.name = 'pointnet_cls'
config.model.pretrained = False
config.model.pretrained_path = ''

config.model.pointnet = ConfigNode()
config.model.pointnet.stn_regularization_weight = 0.001
config.model.pointnet.feature_dim = 1024
config.model.pointnet.dropout = 0.3

config.model.pointnet2 = ConfigNode()
config.model.pointnet2.sa_channels = [64, 128, 256]
config.model.pointnet2.fp_channels = [256, 128, 128]
config.model.pointnet2.feature_dim = 1024
config.model.pointnet2.dropout = 0.4
config.model.pointnet2.use_msg = True

config.model.pointtransformer = ConfigNode()
config.model.pointtransformer.version = 'v1'
config.model.pointtransformer.embed_dim = 64
config.model.pointtransformer.encoder_dims = [64, 128, 256]
config.model.pointtransformer.decoder_dims = [256, 128, 64]
config.model.pointtransformer.depth = [2, 2, 2]
config.model.pointtransformer.num_heads = [4, 8, 16]
config.model.pointtransformer.k = 16
config.model.pointtransformer.kv_k = 16
config.model.pointtransformer.dropout = 0.1
config.model.pointtransformer.drop_path = 0.1
config.model.pointtransformer.mlp_ratio = 4.0
config.model.pointtransformer.qkv_bias = True
config.model.pointtransformer.use_relative_position = True
config.model.pointtransformer.use_partition_pooling = True
config.model.pointtransformer.sampling_ratio = [1.0, 0.5, 0.25]
config.model.pointtransformer.decoder_dim = 128
config.model.pointtransformer.pretrained = False
config.model.pointtransformer.pretrained_path = ''
config.model.pointtransformer.pretrained_source = 'official'
config.model.pointtransformer.strict_pretrained_load = False
config.model.pointtransformer.reset_classifier = True

config.model.dgcnn = ConfigNode()
config.model.dgcnn.k = 20
config.model.dgcnn.edge_dims = [64, 64, 128, 256]
config.model.dgcnn.emb_dims = 1024
config.model.dgcnn.dropout = 0.5
config.model.dgcnn.pretrained = False
config.model.dgcnn.pretrained_path = ''
config.model.dgcnn.pretrained_source = 'official'
config.model.dgcnn.strict_pretrained_load = False
config.model.dgcnn.reset_classifier = True

config.model.kpconv = ConfigNode()
config.model.kpconv.k = 16
config.model.kpconv.encoder_dims = [64, 128, 256]
config.model.kpconv.kernel_points = 15
config.model.kpconv.sigma = 0.1
config.model.kpconv.decoder_dim = 128
config.model.kpconv.dropout = 0.3
config.model.kpconv.pretrained = False
config.model.kpconv.pretrained_path = ''
config.model.kpconv.pretrained_source = 'official'
config.model.kpconv.strict_pretrained_load = False
config.model.kpconv.reset_classifier = True

config.model.pvcnn = ConfigNode()
config.model.pvcnn.block_channels = [64, 128, 256]
config.model.pvcnn.voxel_resolution = 16
config.model.pvcnn.decoder_dim = 128
config.model.pvcnn.dropout = 0.3
config.model.pvcnn.pretrained = False
config.model.pvcnn.pretrained_path = ''
config.model.pvcnn.pretrained_source = 'official'
config.model.pvcnn.strict_pretrained_load = False
config.model.pvcnn.reset_classifier = True

config.model.pointvoxel = ConfigNode()
config.model.pointvoxel.block_channels = [64, 128, 256]
config.model.pointvoxel.voxel_resolution = 16
config.model.pointvoxel.decoder_dim = 128
config.model.pointvoxel.dropout = 0.3
config.model.pointvoxel.pretrained = False
config.model.pointvoxel.pretrained_path = ''
config.model.pointvoxel.pretrained_source = 'official'
config.model.pointvoxel.strict_pretrained_load = False
config.model.pointvoxel.reset_classifier = True

config.model.pointbert = ConfigNode()
config.model.pointbert.num_groups = 128
config.model.pointbert.group_size = 32
config.model.pointbert.embed_dim = 256
config.model.pointbert.depth = 6
config.model.pointbert.num_heads = 8
config.model.pointbert.mlp_ratio = 4.0
config.model.pointbert.dropout = 0.1
config.model.pointbert.pretrained = False
config.model.pointbert.pretrained_path = ''
config.model.pointbert.pretrained_source = 'official'
config.model.pointbert.strict_pretrained_load = False
config.model.pointbert.reset_classifier = True

config.model.pointmae = ConfigNode()
config.model.pointmae.num_groups = 128
config.model.pointmae.group_size = 32
config.model.pointmae.embed_dim = 256
config.model.pointmae.depth = 6
config.model.pointmae.num_heads = 8
config.model.pointmae.mlp_ratio = 4.0
config.model.pointmae.dropout = 0.1
config.model.pointmae.pretrained = False
config.model.pointmae.pretrained_path = ''
config.model.pointmae.pretrained_source = 'official'
config.model.pointmae.strict_pretrained_load = False
config.model.pointmae.reset_classifier = True


config.train = ConfigNode()
config.train.checkpoint = ''
config.train.resume = False
config.train.batch_size = 16
config.train.subdivision = 1
config.train.optimizer = 'adam'
config.train.base_lr = 0.001
config.train.weight_decay = 1e-4
config.train.momentum = 0.9
config.train.nesterov = False
config.train.gradient_clip = 1.0
config.train.seed = 0
config.train.output_dir = 'experiments/point_cloud/exp00'
config.train.log_period = 10
config.train.checkpoint_period = 10
config.train.use_tensorboard = True
config.train.use_apex = False
config.train.precision = 'O0'
config.train.no_weight_decay_on_bn = False
config.train.val_first = False
config.train.val_period = 1

config.train.dist = ConfigNode()
config.train.distributed = False
config.train.dist.backend = 'nccl'
config.train.dist.init_method = 'env://'
config.train.dist.world_size = -1
config.train.dist.node_rank = -1
config.train.dist.local_rank = 0
config.train.dist.use_sync_bn = False

config.qat = ConfigNode()
config.qat.enabled = False
config.qat.backend = 'fbgemm'
config.qat.qconfig = 'default'
config.qat.freeze_bn_epoch = 2
config.qat.disable_observer_epoch = 3
config.qat.convert_before_export = True

config.optim = ConfigNode()
config.optim.adam = ConfigNode()
config.optim.adam.betas = (0.9, 0.999)
config.optim.adamw = ConfigNode()
config.optim.adamw.betas = (0.9, 0.999)

config.scheduler = ConfigNode()
config.scheduler.epochs = 200
config.scheduler.type = 'cosine'
config.scheduler.milestones = [120, 160]
config.scheduler.lr_decay = 0.1
config.scheduler.lr_min_factor = 1e-3
config.scheduler.warmup = ConfigNode()
config.scheduler.warmup.type = 'none'
config.scheduler.warmup.epochs = 0
config.scheduler.warmup.start_factor = 1e-3
config.scheduler.warmup.exponent = 4
config.scheduler.T0 = 10
config.scheduler.T_mul = 1.0

config.train.dataloader = ConfigNode()
config.train.dataloader.num_workers = 2
config.train.dataloader.drop_last = True
config.train.dataloader.pin_memory = False

config.validation = ConfigNode()
config.validation.batch_size = 16
config.validation.dataloader = ConfigNode()
config.validation.dataloader.num_workers = 2
config.validation.dataloader.drop_last = False
config.validation.dataloader.pin_memory = False

config.test = ConfigNode()
config.test.checkpoint = ''
config.test.output_dir = ''
config.test.batch_size = 16
config.test.dataloader = ConfigNode()
config.test.dataloader.num_workers = 2
config.test.dataloader.pin_memory = False

config.export = ConfigNode()
config.export.format = 'onnx'
config.export.opset = 12
config.export.dynamic_axes = True
config.export.checkpoint = ''
config.export.output_file = ''
config.export.quantized_onnx = False
config.export.quantized_onnx_backend = 'onnxruntime_dynamic'

config.prune = ConfigNode()
config.prune.enabled = False
config.prune.backend = 'builtin'
config.prune.method = 'global_unstructured'
config.prune.amount = 0.3
config.prune.modules = ['conv', 'linear']
config.prune.norm = 2
config.prune.n = 1
config.prune.dim = 0
config.prune.remove_reparam = True
config.prune.checkpoint = ''
config.prune.output_dir = ''
config.prune.save_name = 'model_pruned.pth'
config.prune.example_batch_size = 1
config.prune.example_num_points = 1024
config.prune.target = 'model'

config.augmentation = ConfigNode()
config.augmentation.point_cloud = ConfigNode()
config.augmentation.point_cloud.dropout_ratio = 0.875
config.augmentation.point_cloud.scale_low = 0.8
config.augmentation.point_cloud.scale_high = 1.25
config.augmentation.point_cloud.shift_range = 0.1
config.augmentation.point_cloud.jitter_sigma = 0.01
config.augmentation.point_cloud.jitter_clip = 0.05



def update_config(config):
    if config.dataset.name == 'ModelNet40':
        config.task = 'classification'
        config.dataset.n_classes = 40
        config.dataset.n_seg_classes = 0
    elif config.dataset.name == 'ShapeNetPart':
        config.task = 'segmentation'
        config.dataset.n_classes = 16
        config.dataset.n_seg_classes = 50
    else:
        raise ValueError(f'Unsupported point cloud dataset: {config.dataset.name}')

    if not config.dataset.use_normals:
        config.dataset.n_channels = 3
    else:
        config.dataset.n_channels = 6

    return config



def get_default_config():
    return config.clone()
