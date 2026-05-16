from pytorch_image_classification.config.config_node import ConfigNode

config = ConfigNode()
config.task = 'detection'
config.device = 'cuda'

config.cudnn = ConfigNode()
config.cudnn.benchmark = True
config.cudnn.deterministic = False

config.dataset = ConfigNode()
config.dataset.name = 'COCO'
config.dataset.format = 'coco'
config.dataset.dataset_dir = ''
config.dataset.train_ann = ''
config.dataset.val_ann = ''
config.dataset.test_ann = ''
config.dataset.image_size = 640
config.dataset.n_channels = 3
config.dataset.n_classes = 80
config.dataset.class_names = []

config.model = ConfigNode()
config.model.meta_architecture = 'yolo'
config.model.name = 'yolov8_n'
config.model.pretrained = False
config.model.pretrained_path = ''

config.model.yolo = ConfigNode()
config.model.yolo.width_mult = 0.25
config.model.yolo.depth_mult = 0.34
config.model.yolo.channels = []
config.model.yolo.strides = [8, 16, 32]
config.model.yolo.conf_threshold = 0.25
config.model.yolo.nms_threshold = 0.45
config.model.yolo.max_detections = 300
config.model.yolo.num_candidates = 84
config.model.yolo.min_box_size = 0.02
config.model.yolo.dense_head = ConfigNode()
config.model.yolo.dense_head.enabled = True
config.model.yolo.dense_head.type = 'fpn_multi'
config.model.yolo.dense_head.per_cell_boxes = 2
config.model.yolo.dense_head.use_objectness = True
config.model.yolo.dense_head.neck_channels = 128
config.model.yolo.block = 'conv'
config.model.yolo.block_kernel_size = 3

config.model.detr = ConfigNode()
config.model.detr.hidden_dim = 256
config.model.detr.num_queries = 100
config.model.detr.nheads = 8
config.model.detr.num_encoder_layers = 6
config.model.detr.num_decoder_layers = 6
config.model.detr.dim_feedforward = 2048
config.model.detr.dropout = 0.1
config.model.detr.aux_loss = True
config.model.detr.num_feature_levels = 3
config.model.detr.score_threshold = 0.05

config.train = ConfigNode()
config.train.checkpoint = ''
config.train.resume = False
config.train.batch_size = 8
config.train.subdivision = 1
config.train.optimizer = 'adamw'
config.train.base_lr = 1e-4
config.train.backbone_lr = 1e-5
config.train.weight_decay = 1e-4
config.train.momentum = 0.9
config.train.nesterov = True
config.train.gradient_clip = 0.1
config.train.amp = True
config.train.accumulate_steps = 1
config.train.ema = False
config.train.start_epoch = 0
config.train.seed = 0
config.train.output_dir = 'experiments/detection/exp00'
config.train.log_period = 20
config.train.checkpoint_period = 1
config.train.val_period = 1
config.train.use_tensorboard = True

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
config.train.dataloader.non_blocking = False

config.validation = ConfigNode()
config.validation.batch_size = 8
config.validation.dataloader = ConfigNode()
config.validation.dataloader.num_workers = 2
config.validation.dataloader.drop_last = False
config.validation.dataloader.pin_memory = False
config.validation.dataloader.non_blocking = False

config.test = ConfigNode()
config.test.checkpoint = ''
config.test.output_dir = ''
config.test.batch_size = 1
config.test.dataloader = ConfigNode()
config.test.dataloader.num_workers = 2
config.test.dataloader.pin_memory = False

config.assignment = ConfigNode()
config.assignment.name = 'dynamic_k'
config.assignment.background_label = 'auto'
config.assignment.max_matches_per_target = 1
config.assignment.iou_threshold = 0.5
config.assignment.topk = 10
config.assignment.dynamic_k_topk = 10
config.assignment.cls_cost = 1.0
config.assignment.iou_cost = 3.0
config.assignment.eps = 1e-8
config.assignment.box_loss = 'l1'
config.assignment.box_loss_weight = 1.0
config.assignment.iou_variant = 'iou'
config.assignment.yolo = ConfigNode()
config.assignment.yolo.name = 'dynamic_k'
config.assignment.detr = ConfigNode()
config.assignment.detr.name = 'hungarian'

config.augmentation = ConfigNode()
config.augmentation.use_random_horizontal_flip = True
config.augmentation.use_mosaic = False
config.augmentation.use_mixup = False
config.augmentation.use_random_affine = False
config.augmentation.use_label_smoothing = False
config.augmentation.normalize = True
config.augmentation.hflip_prob = 0.5
config.augmentation.mosaic_prob = 1.0
config.augmentation.mixup_prob = 1.0
config.augmentation.affine_prob = 1.0
config.augmentation.mixup_alpha = 0.5
config.augmentation.mosaic_center_ratio_range = [0.5, 1.5]
config.augmentation.affine_degrees = 10.0
config.augmentation.affine_translate = 0.1
config.augmentation.affine_scale_range = [0.8, 1.2]
config.augmentation.affine_shear_degrees = 2.0
config.augmentation.affine_perspective = 0.0
config.augmentation.label_smoothing = ConfigNode()
config.augmentation.label_smoothing.epsilon = 0.0

config.eval = ConfigNode()
config.eval.iou_types = ['bbox']
config.eval.conf_threshold = 0.25
config.eval.nms_threshold = 0.45
config.eval.max_detections = 300
config.eval.nms_type = 'soft'
config.eval.soft_nms_sigma = 0.5
config.eval.soft_nms_score_threshold = 0.01

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
config.prune.example_image_size = 640
config.prune.target = 'backbone'

config.distill = ConfigNode()
config.distill.enabled = False
config.distill.teacher_checkpoint = ''
config.distill.temperature = 1.0
config.distill.cls_weight = 1.0
config.distill.box_weight = 1.0
config.distill.hard_loss_weight = 1.0
config.distill.soft_loss_weight = 1.0


def get_default_config():
    return config.clone()
