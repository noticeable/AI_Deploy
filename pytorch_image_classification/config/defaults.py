from .config_node import ConfigNode

config = ConfigNode()

config.device = 'cuda'
# cuDNN
config.cudnn = ConfigNode()
config.cudnn.benchmark = True
config.cudnn.deterministic = False

config.dataset = ConfigNode()
config.dataset.name = 'CIFAR10'
config.dataset.dataset_dir = ''
config.dataset.image_size = 32
config.dataset.n_channels = 3
config.dataset.n_classes = 10

config.model = ConfigNode()
# options: 'cifar', 'imagenet'
# Use 'cifar' for small input images
config.model.type = 'cifar'
config.model.name = 'resnet_preact'
config.model.init_mode = 'kaiming_fan_out'

config.model.transformer = ConfigNode()
config.model.transformer.patch_size = 16
config.model.transformer.emb_dim = 768
config.model.transformer.mlp_dim = 3072
config.model.transformer.num_heads = 12
config.model.transformer.num_layers = 12
config.model.transformer.dropout = 0.0
config.model.transformer.attention_dropout = 0.0
config.model.transformer.drop_path_rate = 0.0
config.model.transformer.classifier = 'token'
config.model.transformer.representation_size = 0
config.model.transformer.qkv_bias = True
config.model.transformer.pretrained = False
config.model.transformer.pretrained_path = ''
config.model.transformer.pretrained_source = 'checkpoint'
config.model.transformer.strict_pretrained_load = True
config.model.transformer.reset_classifier = True
config.model.transformer.interpolate_position_embedding = True

config.model.vit = ConfigNode()
config.model.vit.patch_size = 16
config.model.vit.emb_dim = 768
config.model.vit.mlp_dim = 3072
config.model.vit.num_heads = 12
config.model.vit.num_layers = 12
config.model.vit.dropout = 0.0
config.model.vit.attention_dropout = 0.0
config.model.vit.classifier = 'token'
config.model.vit.representation_size = 0
config.model.vit.pretrained = False
config.model.vit.pretrained_path = ''
config.model.vit.pretrained_source = 'checkpoint'
config.model.vit.strict_pretrained_load = True
config.model.vit.reset_classifier = True
config.model.vit.interpolate_position_embedding = True

config.model.deit = ConfigNode()
config.model.deit.patch_size = 16
config.model.deit.emb_dim = 768
config.model.deit.mlp_dim = 3072
config.model.deit.num_heads = 12
config.model.deit.num_layers = 12
config.model.deit.dropout = 0.0
config.model.deit.attention_dropout = 0.0
config.model.deit.drop_path_rate = 0.0
config.model.deit.classifier = 'token'
config.model.deit.representation_size = 0
config.model.deit.qkv_bias = True
config.model.deit.distilled = False
config.model.deit.pretrained = False
config.model.deit.pretrained_path = ''
config.model.deit.pretrained_source = 'checkpoint'
config.model.deit.strict_pretrained_load = True
config.model.deit.reset_classifier = True
config.model.deit.interpolate_position_embedding = True

config.model.cvt = ConfigNode()
config.model.cvt.patch_size = 7
config.model.cvt.emb_dim = 64
config.model.cvt.mlp_dim = 256
config.model.cvt.num_heads = [1, 3, 6]
config.model.cvt.num_layers = [1, 2, 10]
config.model.cvt.dropout = 0.0
config.model.cvt.attention_dropout = 0.0
config.model.cvt.drop_path_rate = 0.0
config.model.cvt.classifier = 'gap'
config.model.cvt.representation_size = 0
config.model.cvt.qkv_bias = True
config.model.cvt.stage_dims = [64, 192, 384]
config.model.cvt.stage_patch_sizes = [7, 3, 3]
config.model.cvt.stage_strides = [4, 2, 2]
config.model.cvt.mlp_ratios = [4.0, 4.0, 4.0]
config.model.cvt.pretrained = False
config.model.cvt.pretrained_path = ''
config.model.cvt.pretrained_source = 'checkpoint'
config.model.cvt.strict_pretrained_load = True
config.model.cvt.reset_classifier = True
config.model.cvt.interpolate_position_embedding = True

config.model.convit = ConfigNode()
config.model.convit.patch_size = 16
config.model.convit.emb_dim = 768
config.model.convit.mlp_dim = 3072
config.model.convit.num_heads = 12
config.model.convit.num_layers = 12
config.model.convit.dropout = 0.0
config.model.convit.attention_dropout = 0.0
config.model.convit.drop_path_rate = 0.0
config.model.convit.classifier = 'token'
config.model.convit.representation_size = 0
config.model.convit.qkv_bias = True
config.model.convit.local_layers = 10
config.model.convit.locality_strength = 1.0
config.model.convit.pretrained = False
config.model.convit.pretrained_path = ''
config.model.convit.pretrained_source = 'checkpoint'
config.model.convit.strict_pretrained_load = True
config.model.convit.reset_classifier = True
config.model.convit.interpolate_position_embedding = True

config.model.t2t_vit = ConfigNode()
config.model.t2t_vit.patch_size = 16
config.model.t2t_vit.emb_dim = 768
config.model.t2t_vit.mlp_dim = 3072
config.model.t2t_vit.num_heads = 12
config.model.t2t_vit.num_layers = 12
config.model.t2t_vit.dropout = 0.0
config.model.t2t_vit.attention_dropout = 0.0
config.model.t2t_vit.drop_path_rate = 0.0
config.model.t2t_vit.classifier = 'token'
config.model.t2t_vit.representation_size = 0
config.model.t2t_vit.qkv_bias = True
config.model.t2t_vit.pretrained = False
config.model.t2t_vit.pretrained_path = ''
config.model.t2t_vit.pretrained_source = 'checkpoint'
config.model.t2t_vit.strict_pretrained_load = True
config.model.t2t_vit.reset_classifier = True
config.model.t2t_vit.interpolate_position_embedding = True

config.model.cct = ConfigNode()
config.model.cct.patch_size = 16
config.model.cct.emb_dim = 768
config.model.cct.mlp_dim = 3072
config.model.cct.num_heads = 12
config.model.cct.num_layers = 12
config.model.cct.dropout = 0.0
config.model.cct.attention_dropout = 0.0
config.model.cct.drop_path_rate = 0.0
config.model.cct.classifier = 'token'
config.model.cct.representation_size = 0
config.model.cct.qkv_bias = True
config.model.cct.pretrained = False
config.model.cct.pretrained_path = ''
config.model.cct.pretrained_source = 'checkpoint'
config.model.cct.strict_pretrained_load = True
config.model.cct.reset_classifier = True
config.model.cct.interpolate_position_embedding = True

config.model.pvt = ConfigNode()
config.model.pvt.patch_size = 4
config.model.pvt.emb_dim = 64
config.model.pvt.mlp_dim = 256
config.model.pvt.num_heads = [1, 2, 5, 8]
config.model.pvt.num_layers = [2, 2, 2, 2]
config.model.pvt.dropout = 0.0
config.model.pvt.attention_dropout = 0.0
config.model.pvt.drop_path_rate = 0.0
config.model.pvt.classifier = 'gap'
config.model.pvt.representation_size = 0
config.model.pvt.qkv_bias = True
config.model.pvt.stage_dims = [64, 128, 320, 512]
config.model.pvt.mlp_ratios = [8.0, 8.0, 4.0, 4.0]
config.model.pvt.sr_ratios = [8, 4, 2, 1]
config.model.pvt.pretrained = False
config.model.pvt.pretrained_path = ''
config.model.pvt.pretrained_source = 'checkpoint'
config.model.pvt.strict_pretrained_load = True
config.model.pvt.reset_classifier = True
config.model.pvt.interpolate_position_embedding = True

config.model.swin_transformer = ConfigNode()
config.model.swin_transformer.patch_size = 4
config.model.swin_transformer.emb_dim = 96
config.model.swin_transformer.mlp_dim = 384
config.model.swin_transformer.num_heads = [3, 6, 12, 24]
config.model.swin_transformer.num_layers = [2, 2, 6, 2]
config.model.swin_transformer.dropout = 0.0
config.model.swin_transformer.attention_dropout = 0.0
config.model.swin_transformer.drop_path_rate = 0.0
config.model.swin_transformer.classifier = 'gap'
config.model.swin_transformer.representation_size = 0
config.model.swin_transformer.qkv_bias = True
config.model.swin_transformer.window_size = 7
config.model.swin_transformer.mlp_ratios = [4.0, 4.0, 4.0, 4.0]
config.model.swin_transformer.pretrained = False
config.model.swin_transformer.pretrained_path = ''
config.model.swin_transformer.pretrained_source = 'checkpoint'
config.model.swin_transformer.strict_pretrained_load = True
config.model.swin_transformer.reset_classifier = True
config.model.swin_transformer.interpolate_position_embedding = True

config.model.mvit = ConfigNode()
config.model.mvit.patch_size = 16
config.model.mvit.emb_dim = 768
config.model.mvit.mlp_dim = 3072
config.model.mvit.num_heads = 12
config.model.mvit.num_layers = 12
config.model.mvit.dropout = 0.0
config.model.mvit.attention_dropout = 0.0
config.model.mvit.drop_path_rate = 0.0
config.model.mvit.classifier = 'token'
config.model.mvit.representation_size = 0
config.model.mvit.qkv_bias = True
config.model.mvit.pretrained = False
config.model.mvit.pretrained_path = ''
config.model.mvit.pretrained_source = 'checkpoint'
config.model.mvit.strict_pretrained_load = True
config.model.mvit.reset_classifier = True
config.model.mvit.interpolate_position_embedding = True

config.model.focal_transformer = ConfigNode()
config.model.focal_transformer.patch_size = 16
config.model.focal_transformer.emb_dim = 768
config.model.focal_transformer.mlp_dim = 3072
config.model.focal_transformer.num_heads = 12
config.model.focal_transformer.num_layers = 12
config.model.focal_transformer.dropout = 0.0
config.model.focal_transformer.attention_dropout = 0.0
config.model.focal_transformer.drop_path_rate = 0.0
config.model.focal_transformer.classifier = 'token'
config.model.focal_transformer.representation_size = 0
config.model.focal_transformer.qkv_bias = True
config.model.focal_transformer.pretrained = False
config.model.focal_transformer.pretrained_path = ''
config.model.focal_transformer.pretrained_source = 'checkpoint'
config.model.focal_transformer.strict_pretrained_load = True
config.model.focal_transformer.reset_classifier = True
config.model.focal_transformer.interpolate_position_embedding = True

config.model.performer = ConfigNode()
config.model.performer.patch_size = 16
config.model.performer.emb_dim = 768
config.model.performer.mlp_dim = 3072
config.model.performer.num_heads = 12
config.model.performer.num_layers = 12
config.model.performer.dropout = 0.0
config.model.performer.attention_dropout = 0.0
config.model.performer.drop_path_rate = 0.0
config.model.performer.classifier = 'token'
config.model.performer.representation_size = 0
config.model.performer.qkv_bias = True
config.model.performer.pretrained = False
config.model.performer.pretrained_path = ''
config.model.performer.pretrained_source = 'checkpoint'
config.model.performer.strict_pretrained_load = True
config.model.performer.reset_classifier = True
config.model.performer.interpolate_position_embedding = True

config.model.maxvit = ConfigNode()
config.model.maxvit.patch_size = 16
config.model.maxvit.emb_dim = 768
config.model.maxvit.mlp_dim = 3072
config.model.maxvit.num_heads = 12
config.model.maxvit.num_layers = 12
config.model.maxvit.dropout = 0.0
config.model.maxvit.attention_dropout = 0.0
config.model.maxvit.drop_path_rate = 0.0
config.model.maxvit.classifier = 'token'
config.model.maxvit.representation_size = 0
config.model.maxvit.qkv_bias = True
config.model.maxvit.pretrained = False
config.model.maxvit.pretrained_path = ''
config.model.maxvit.pretrained_source = 'checkpoint'
config.model.maxvit.strict_pretrained_load = True
config.model.maxvit.reset_classifier = True
config.model.maxvit.interpolate_position_embedding = True

config.model.efficientvit = ConfigNode()
config.model.efficientvit.patch_size = 16
config.model.efficientvit.emb_dim = 768
config.model.efficientvit.mlp_dim = 3072
config.model.efficientvit.num_heads = 12
config.model.efficientvit.num_layers = 12
config.model.efficientvit.dropout = 0.0
config.model.efficientvit.attention_dropout = 0.0
config.model.efficientvit.drop_path_rate = 0.0
config.model.efficientvit.classifier = 'token'
config.model.efficientvit.representation_size = 0
config.model.efficientvit.qkv_bias = True
config.model.efficientvit.pretrained = False
config.model.efficientvit.pretrained_path = ''
config.model.efficientvit.pretrained_source = 'checkpoint'
config.model.efficientvit.strict_pretrained_load = True
config.model.efficientvit.reset_classifier = True
config.model.efficientvit.interpolate_position_embedding = True

config.model.edgevit = ConfigNode()
config.model.edgevit.patch_size = 16
config.model.edgevit.emb_dim = 768
config.model.edgevit.mlp_dim = 3072
config.model.edgevit.num_heads = 12
config.model.edgevit.num_layers = 12
config.model.edgevit.dropout = 0.0
config.model.edgevit.attention_dropout = 0.0
config.model.edgevit.drop_path_rate = 0.0
config.model.edgevit.classifier = 'token'
config.model.edgevit.representation_size = 0
config.model.edgevit.qkv_bias = True
config.model.edgevit.pretrained = False
config.model.edgevit.pretrained_path = ''
config.model.edgevit.pretrained_source = 'checkpoint'
config.model.edgevit.strict_pretrained_load = True
config.model.edgevit.reset_classifier = True
config.model.edgevit.interpolate_position_embedding = True

config.model.retnet = ConfigNode()
config.model.retnet.patch_size = 16
config.model.retnet.emb_dim = 768
config.model.retnet.mlp_dim = 3072
config.model.retnet.num_heads = 12
config.model.retnet.num_layers = 12
config.model.retnet.dropout = 0.0
config.model.retnet.attention_dropout = 0.0
config.model.retnet.drop_path_rate = 0.0
config.model.retnet.classifier = 'token'
config.model.retnet.representation_size = 0
config.model.retnet.qkv_bias = True
config.model.retnet.pretrained = False
config.model.retnet.pretrained_path = ''
config.model.retnet.pretrained_source = 'checkpoint'
config.model.retnet.strict_pretrained_load = True
config.model.retnet.reset_classifier = True
config.model.retnet.interpolate_position_embedding = True

config.model.vgg = ConfigNode()
config.model.vgg.n_channels = [64, 128, 256, 512, 512]
config.model.vgg.n_layers = [2, 2, 3, 3, 3]
config.model.vgg.use_bn = True

config.model.resnet = ConfigNode()
config.model.resnet.depth = 110  # for cifar type model
config.model.resnet.n_blocks = [2, 2, 2, 2]  # for imagenet type model
config.model.resnet.block_type = 'basic'
config.model.resnet.initial_channels = 16

config.model.resnet_preact = ConfigNode()
config.model.resnet_preact.depth = 110  # for cifar type model
config.model.resnet_preact.n_blocks = [2, 2, 2, 2]  # for imagenet type model
config.model.resnet_preact.block_type = 'basic'
config.model.resnet_preact.initial_channels = 16
config.model.resnet_preact.remove_first_relu = False
config.model.resnet_preact.add_last_bn = False
config.model.resnet_preact.preact_stage = [True, True, True]

config.model.wrn = ConfigNode()
config.model.wrn.depth = 28  # for cifar type model
config.model.wrn.initial_channels = 16
config.model.wrn.widening_factor = 10
config.model.wrn.drop_rate = 0.0

config.model.densenet = ConfigNode()
config.model.densenet.depth = 100  # for cifar type model
config.model.densenet.n_blocks = [6, 12, 24, 16]  # for imagenet type model
config.model.densenet.block_type = 'bottleneck'
config.model.densenet.growth_rate = 12
config.model.densenet.drop_rate = 0.0
config.model.densenet.compression_rate = 0.5

config.model.pyramidnet = ConfigNode()
config.model.pyramidnet.depth = 272  # for cifar type model
config.model.pyramidnet.n_blocks = [3, 24, 36, 3]  # for imagenet type model
config.model.pyramidnet.initial_channels = 16
config.model.pyramidnet.block_type = 'bottleneck'
config.model.pyramidnet.alpha = 200

config.model.resnext = ConfigNode()
config.model.resnext.depth = 29  # for cifar type model
config.model.resnext.n_blocks = [3, 4, 6, 3]  # for imagenet type model
config.model.resnext.initial_channels = 64
config.model.resnext.cardinality = 8
config.model.resnext.base_channels = 4

config.model.shake_shake = ConfigNode()
config.model.shake_shake.depth = 26  # for cifar type model
config.model.shake_shake.initial_channels = 96
config.model.shake_shake.shake_forward = True
config.model.shake_shake.shake_backward = True
config.model.shake_shake.shake_image = True

config.model.se_resnet_preact = ConfigNode()
config.model.se_resnet_preact.depth = 110  # for cifar type model
config.model.se_resnet_preact.initial_channels = 16
config.model.se_resnet_preact.se_reduction = 16
config.model.se_resnet_preact.block_type = 'basic'
config.model.se_resnet_preact.initial_channels = 16
config.model.se_resnet_preact.remove_first_relu = False
config.model.se_resnet_preact.add_last_bn = False
config.model.se_resnet_preact.preact_stage = [True, True, True]

config.train = ConfigNode()
config.train.checkpoint = ''
config.train.resume = False
config.train.use_apex = True
# optimization level for NVIDIA apex
# O0 = fp32
# O1 = mixed precision
# O2 = almost fp16
# O3 = fp16
config.train.precision = 'O0'
config.train.batch_size = 128
config.train.subdivision = 1
# optimizer (options: sgd, adam, adamw, lars, adabound, adaboundw)
config.train.optimizer = 'sgd'
config.train.base_lr = 0.1
config.train.momentum = 0.9
config.train.nesterov = True
config.train.weight_decay = 1e-4
config.train.no_weight_decay_on_bn = False
config.train.gradient_clip = 0.0
config.train.start_epoch = 0
config.train.seed = 0
config.train.val_first = True
config.train.val_period = 1
config.train.val_ratio = 0.0
config.train.use_test_as_val = True

config.train.output_dir = 'experiments/exp00'
config.train.log_period = 100
config.train.checkpoint_period = 10

config.train.use_tensorboard = True
config.qat = ConfigNode()
config.qat.enabled = False
config.qat.backend = 'fbgemm'
config.qat.qconfig = 'default'
config.qat.start_epoch = 0
config.qat.freeze_bn_epoch = 2
config.qat.disable_observer_epoch = 3
config.qat.convert_before_export = True
config.qat.checkpoint = ''
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
config.prune.example_image_size = 32
config.prune.target = 'model'
config.tensorboard = ConfigNode()
config.tensorboard.train_images = False
config.tensorboard.val_images = False
config.tensorboard.model_params = False

# optimizer
config.optim = ConfigNode()
# Adam
config.optim.adam = ConfigNode()
config.optim.adam.betas = (0.9, 0.999)
# LARS
config.optim.lars = ConfigNode()
config.optim.lars.eps = 1e-9
config.optim.lars.threshold = 1e-2
# AdaBound
config.optim.adabound = ConfigNode()
config.optim.adabound.betas = (0.9, 0.999)
config.optim.adabound.final_lr = 0.1
config.optim.adabound.gamma = 1e-3

# scheduler
config.scheduler = ConfigNode()
config.scheduler.epochs = 160
# warm up (options: none, linear, exponential)
config.scheduler.warmup = ConfigNode()
config.scheduler.warmup.type = 'none'
config.scheduler.warmup.epochs = 0
config.scheduler.warmup.start_factor = 1e-3
config.scheduler.warmup.exponent = 4
# main scheduler (options: constant, linear, multistep, cosine, sgdr)
config.scheduler.type = 'multistep'
config.scheduler.milestones = [80, 120]
config.scheduler.lr_decay = 0.1
config.scheduler.lr_min_factor = 0.001
config.scheduler.T0 = 10
config.scheduler.T_mul = 1.

# train data loader
config.train.dataloader = ConfigNode()
config.train.dataloader.num_workers = 2
config.train.dataloader.drop_last = True
config.train.dataloader.pin_memory = False
config.train.dataloader.non_blocking = False

# validation data loader
config.validation = ConfigNode()
config.validation.batch_size = 256
config.validation.dataloader = ConfigNode()
config.validation.dataloader.num_workers = 2
config.validation.dataloader.drop_last = False
config.validation.dataloader.pin_memory = False
config.validation.dataloader.non_blocking = False

# distributed
config.train.distributed = False
config.train.dist = ConfigNode()
config.train.dist.backend = 'nccl'
config.train.dist.init_method = 'env://'
config.train.dist.world_size = -1
config.train.dist.node_rank = -1
config.train.dist.local_rank = 0
config.train.dist.use_sync_bn = False

config.augmentation = ConfigNode()
config.augmentation.use_random_crop = True
config.augmentation.use_random_horizontal_flip = True
config.augmentation.use_cutout = False
config.augmentation.use_random_erasing = False
config.augmentation.use_dual_cutout = False
config.augmentation.use_mixup = False
config.augmentation.use_ricap = False
config.augmentation.use_cutmix = False
config.augmentation.use_label_smoothing = False

config.augmentation.random_crop = ConfigNode()
config.augmentation.random_crop.padding = 4
config.augmentation.random_crop.fill = 0
config.augmentation.random_crop.padding_mode = 'constant'

config.augmentation.random_horizontal_flip = ConfigNode()
config.augmentation.random_horizontal_flip.prob = 0.5

config.augmentation.cutout = ConfigNode()
config.augmentation.cutout.prob = 1.0
config.augmentation.cutout.mask_size = 16
config.augmentation.cutout.cut_inside = False
config.augmentation.cutout.mask_color = 0
config.augmentation.cutout.dual_cutout_alpha = 0.1

config.augmentation.random_erasing = ConfigNode()
config.augmentation.random_erasing.prob = 0.5
config.augmentation.random_erasing.area_ratio_range = [0.02, 0.4]
config.augmentation.random_erasing.min_aspect_ratio = 0.3
config.augmentation.random_erasing.max_attempt = 20

config.augmentation.mixup = ConfigNode()
config.augmentation.mixup.alpha = 1.0

config.augmentation.ricap = ConfigNode()
config.augmentation.ricap.beta = 0.3

config.augmentation.cutmix = ConfigNode()
config.augmentation.cutmix.alpha = 1.0

config.augmentation.label_smoothing = ConfigNode()
config.augmentation.label_smoothing.epsilon = 0.1

config.tta = ConfigNode()
config.tta.use_resize = False
config.tta.use_center_crop = False
config.tta.resize = 256

config.export = ConfigNode()
config.export.format = 'onnx'
config.export.opset = 12
config.export.dynamic_axes = True
config.export.checkpoint = ''
config.export.output_file = ''
config.export.quantized_onnx = False
config.export.quantized_onnx_backend = 'onnxruntime_dynamic'

# test config
config.test = ConfigNode()
config.test.checkpoint = ''
config.test.output_dir = ''
config.test.batch_size = 256
# test data loader
config.test.dataloader = ConfigNode()
config.test.dataloader.num_workers = 2
config.test.dataloader.pin_memory = False


def get_default_config():
    return config.clone()
