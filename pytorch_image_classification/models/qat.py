import copy

import torch
import torch.nn as nn
import torch.ao.quantization as quantization

from pytorch_image_classification.models.qat_common import (
    QATStateController,
    apply_qat_epoch_controls,
    create_default_qat_qconfig,
    is_qat_enabled,
)


SUPPORTED_CLASSIFICATION_QAT_MODELS = {
    ('cifar', 'resnet'),
    ('imagenet', 'resnet'),
    ('cifar', 'vit'),
}


class QuantizableBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels,
                               out_channels,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(out_channels,
                               out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=False)
        self.add = nn.quantized.FloatFunctional()

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module(
                'conv',
                nn.Conv2d(in_channels,
                          out_channels,
                          kernel_size=1,
                          stride=stride,
                          padding=0,
                          bias=False))
            self.shortcut.add_module('bn', nn.BatchNorm2d(out_channels))

    def forward(self, x):
        y = self.relu1(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        y = self.add.add(y, self.shortcut(x))
        y = self.relu2(y)
        return y

    def fuse_model(self):
        quantization.fuse_modules(self, [['conv1', 'bn1', 'relu1'], ['conv2', 'bn2']], inplace=True)
        if len(self.shortcut) == 2:
            quantization.fuse_modules(self.shortcut, [['conv', 'bn']], inplace=True)


class QuantizableBottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        bottleneck_channels = out_channels // self.expansion
        self.conv1 = nn.Conv2d(in_channels,
                               bottleneck_channels,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(bottleneck_channels,
                               bottleneck_channels,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv3 = nn.Conv2d(bottleneck_channels,
                               out_channels,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu3 = nn.ReLU(inplace=False)
        self.add = nn.quantized.FloatFunctional()

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module(
                'conv',
                nn.Conv2d(in_channels,
                          out_channels,
                          kernel_size=1,
                          stride=stride,
                          padding=0,
                          bias=False))
            self.shortcut.add_module('bn', nn.BatchNorm2d(out_channels))

    def forward(self, x):
        y = self.relu1(self.bn1(self.conv1(x)))
        y = self.relu2(self.bn2(self.conv2(y)))
        y = self.bn3(self.conv3(y))
        y = self.add.add(y, self.shortcut(x))
        y = self.relu3(y)
        return y

    def fuse_model(self):
        quantization.fuse_modules(self, [['conv1', 'bn1', 'relu1'], ['conv2', 'bn2', 'relu2'], ['conv3', 'bn3']], inplace=True)
        if len(self.shortcut) == 2:
            quantization.fuse_modules(self.shortcut, [['conv', 'bn']], inplace=True)




class QuantizableResNet(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.quant = quantization.QuantStub()
        self.conv = base_model.conv
        self.bn = base_model.bn
        self.relu = nn.ReLU(inplace=False)
        self.stage1 = base_model.stage1
        self.stage2 = base_model.stage2
        self.stage3 = base_model.stage3
        self.stage4 = getattr(base_model, 'stage4', None)
        self.fc = base_model.fc
        self.dequant = quantization.DeQuantStub()

    def _forward_conv(self, x):
        x = self.quant(x)
        x = self.relu(self.bn(self.conv(x)))
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        if self.stage4 is not None:
            x = self.stage4(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, output_size=1)
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.dequant(x)
        return x

    def fuse_model(self):
        quantization.fuse_modules(self, [['conv', 'bn', 'relu']], inplace=True)
        for stage_name in ('stage1', 'stage2', 'stage3', 'stage4'):
            stage = getattr(self, stage_name, None)
            if stage is None:
                continue
            for module in stage.modules():
                if hasattr(module, 'fuse_model'):
                    module.fuse_model()


class QuantizableVisionTransformer(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.model_name = getattr(base_model, 'model_name', 'vit')
        self.classifier = base_model.classifier
        self.representation_size = base_model.representation_size
        self.emb_dim = base_model.emb_dim
        self.patch_size = base_model.patch_size
        self.image_size = base_model.image_size
        self.grid_size = base_model.grid_size
        self.seq_length = base_model.seq_length
        self.conv_proj = base_model.conv_proj
        self.conv_proj.qconfig = None
        self.class_token = base_model.class_token
        self.encoder = base_model.encoder
        self.encoder.qconfig = None
        self.pre_logits = base_model.pre_logits
        self.act = base_model.act
        self.head = nn.Linear(base_model.head.in_features,
                              base_model.head.out_features,
                              bias=base_model.head.bias is not None)
        self.head.load_state_dict(base_model.head.state_dict(), strict=True)
        self.head.qconfig = None
        self.quant = quantization.QuantStub()
        self.dequant = quantization.DeQuantStub()

    def _process_input(self, x):
        n, _, h, w = x.shape
        if h != self.image_size or w != self.image_size:
            raise ValueError(f'Expected input size {(self.image_size, self.image_size)}, '
                             f'got {(h, w)}')
        x = self.conv_proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

    def forward_features(self, x):
        x = self._process_input(x)
        if self.class_token is not None:
            class_token = self.class_token.expand(x.size(0), -1, -1)
            x = torch.cat((class_token, x), dim=1)
        x = self.encoder(x)
        if self.classifier == 'token':
            x = x[:, 0]
        else:
            x = x.mean(dim=1)
        x = self.pre_logits(x)
        x = self.act(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.quant(x)
        x = self.dequant(x)
        x = self.head(x)
        return x

    def fuse_model(self):
        return None


def is_qat_supported_model(config):
    return (config.model.type, config.model.name) in SUPPORTED_CLASSIFICATION_QAT_MODELS


def _convert_stage_to_quantizable(stage, block_map):
    converted = nn.Sequential()
    for name, module in stage.named_children():
        block_type = type(module)
        if block_type not in block_map:
            raise ValueError(f'Unsupported ResNet block for QAT: {block_type.__name__}')
        quant_block = block_map[block_type](module)
        converted.add_module(name, quant_block)
    return converted


def _convert_basic_block(module):
    block = QuantizableBasicBlock(module.conv1.in_channels,
                                  module.conv2.out_channels,
                                  module.conv1.stride[0])
    block.load_state_dict(module.state_dict(), strict=True)
    return block




def _convert_bottleneck_block(module):
    block = QuantizableBottleneckBlock(module.conv1.in_channels,
                                       module.conv3.out_channels,
                                       module.conv2.stride[0])
    block.load_state_dict(module.state_dict(), strict=True)
    return block




def _convert_vit_model(module):
    vit_model = QuantizableVisionTransformer(module)
    vit_model.load_state_dict(module.state_dict(), strict=False)
    return vit_model


def convert_model_to_qat_compatible(config, model):
    if not is_qat_supported_model(config):
        raise ValueError(f'QAT is not supported for model family {(config.model.type, config.model.name)}')
    if getattr(model, '_is_qat_compatible', False):
        return model

    model_copy = copy.deepcopy(model)
    if config.model.name == 'vit':
        qat_model = _convert_vit_model(model_copy)
        qat_model._is_qat_compatible = True
        return qat_model

    block_type_name = type(model_copy.stage1[0]).__name__
    if block_type_name == 'BasicBlock':
        block_converter = _convert_basic_block
    elif block_type_name == 'BottleneckBlock':
        block_converter = _convert_bottleneck_block
    else:
        raise ValueError(f'Unsupported ResNet block type for QAT: {block_type_name}')

    model_copy.stage1 = _convert_stage_to_quantizable(model_copy.stage1, {type(model_copy.stage1[0]): block_converter})
    model_copy.stage2 = _convert_stage_to_quantizable(model_copy.stage2, {type(model_copy.stage2[0]): block_converter})
    model_copy.stage3 = _convert_stage_to_quantizable(model_copy.stage3, {type(model_copy.stage3[0]): block_converter})
    if hasattr(model_copy, 'stage4'):
        model_copy.stage4 = _convert_stage_to_quantizable(model_copy.stage4, {type(model_copy.stage4[0]): block_converter})
    qat_model = QuantizableResNet(model_copy)
    qat_model._is_qat_compatible = True
    return qat_model


def prepare_model_for_qat(config, model):
    model = convert_model_to_qat_compatible(config, model)
    model.eval()
    model.fuse_model()
    model.train()
    model.qconfig = create_default_qat_qconfig(config)
    if isinstance(model, QuantizableVisionTransformer):
        model.head.qconfig = None
    quantization.prepare_qat(model, inplace=True)
    return model, QATStateController()


def convert_qat_model(config, model):
    if not is_qat_enabled(config):
        return model
    model_to_convert = copy.deepcopy(model).cpu().eval()
    return quantization.convert(model_to_convert, inplace=True)
