from pytorch_image_classification.config.config_node import ConfigNode

config = ConfigNode()
config.task = 'det_seg'
config.device = 'cuda'

config.cudnn = ConfigNode()
config.cudnn.benchmark = True
config.cudnn.deterministic = False

config.dataset = ConfigNode()
config.dataset.name = 'DetSegYOLO'
config.dataset.format = 'det_seg_yolo'
config.dataset.dataset_dir = ''
config.dataset.image_size = 640
config.dataset.n_channels = 3
config.dataset.n_classes = 1
config.dataset.class_names = []
config.dataset.ignore_index = 255

config.model = ConfigNode()
config.model.meta_architecture = 'det_seg'
config.model.name = 'shared_backbone_tiny'
config.model.pretrained = False
config.model.pretrained_path = ''

config.model.backbone = ConfigNode()
config.model.backbone.width_mult = 0.25
config.model.backbone.channels = []

config.model.det_neck = ConfigNode()
config.model.det_neck.hidden_channels = 128

config.model.seg_neck = ConfigNode()
config.model.seg_neck.hidden_channels = 64

config.model.det_head = ConfigNode()
config.model.det_head.score_threshold = 0.25

config.model.seg_head = ConfigNode()
config.model.seg_head.out_channels = 0

config.train = ConfigNode()
config.train.checkpoint = ''
config.train.resume = False
config.train.batch_size = 8
config.train.optimizer = 'adamw'
config.train.base_lr = 1e-4
config.train.weight_decay = 1e-4
config.train.gradient_clip = 0.1
config.train.seed = 0
config.train.output_dir = 'experiments/det_seg/exp00'
config.train.log_period = 20
config.train.checkpoint_period = 1

config.qat = ConfigNode()
config.qat.enabled = False
config.qat.backend = 'fbgemm'
config.qat.qconfig = 'default'
config.qat.freeze_bn_epoch = 2
config.qat.disable_observer_epoch = 3
config.qat.convert_before_export = True

config.optim = ConfigNode()
config.optim.adamw = ConfigNode()
config.optim.adamw.betas = (0.9, 0.999)

config.scheduler = ConfigNode()
config.scheduler.epochs = 12
config.scheduler.type = 'multistep'
config.scheduler.milestones = [8, 11]
config.scheduler.lr_decay = 0.1
config.scheduler.lr_min_factor = 0.0
config.scheduler.warmup = ConfigNode()
config.scheduler.warmup.type = 'none'
config.scheduler.warmup.epochs = 0
config.scheduler.warmup.start_factor = 1e-3
config.scheduler.warmup.exponent = 4
config.scheduler.T0 = 10
config.scheduler.T_mul = 1.0

config.train.dataloader = ConfigNode()
config.train.dataloader.num_workers = 2
config.train.dataloader.drop_last = False
config.train.dataloader.pin_memory = False

config.validation = ConfigNode()
config.validation.batch_size = 8
config.validation.dataloader = ConfigNode()
config.validation.dataloader.num_workers = 2
config.validation.dataloader.drop_last = False
config.validation.dataloader.pin_memory = False

config.test = ConfigNode()
config.test.checkpoint = ''
config.test.output_dir = ''
config.test.batch_size = 1
config.test.dataloader = ConfigNode()
config.test.dataloader.num_workers = 2
config.test.dataloader.pin_memory = False

config.augmentation = ConfigNode()
config.augmentation.use_random_horizontal_flip = True
config.augmentation.normalize = True

config.eval = ConfigNode()
config.eval.iou_threshold = 0.5

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
config.prune.modules = ['conv']
config.prune.norm = 2
config.prune.n = 1
config.prune.dim = 0
config.prune.remove_reparam = True
config.prune.checkpoint = ''
config.prune.output_dir = ''
config.prune.save_name = 'model_pruned.pth'
config.prune.example_batch_size = 1
config.prune.example_image_size = 640
config.prune.target = 'backbone'

config.loss = ConfigNode()
config.loss.det_weight = 1.0
config.loss.seg_weight = 1.0

config.limtl = ConfigNode()
config.limtl.enabled = False
config.limtl.strategy = 'EW'
config.limtl.graddrop = ConfigNode()
config.limtl.graddrop.leak = 0.0



def get_default_config():
    return config.clone()
