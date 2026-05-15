import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.ao.nn.quantized as nnq
import torch.ao.quantization as quantization

from pytorch_image_classification.models.qat_common import (
    QATStateController,
    apply_qat_epoch_controls,
    create_default_qat_qconfig,
    is_qat_enabled,
)
from pytorch_point_cloud.models.pointnet.common import feature_transform_regularizer

SUPPORTED_POINT_CLOUD_QAT_MODELS = {
    ('pointnet', 'pointnet_cls'),
    ('pointnet', 'pointnet_seg'),
    ('pointnet2', 'pointnet2_cls'),
    ('pointnet2', 'pointnet2_seg'),
    ('dgcnn', 'dgcnn_cls'),
    ('dgcnn', 'dgcnn_seg'),
}


class QuantizablePointNetClassifier(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.config = copy.deepcopy(getattr(base_model, 'config', None))
        self.reg_weight = base_model.reg_weight
        self.encoder = base_model.encoder
        self.fc1 = nn.Linear(base_model.fc1.in_features,
                             base_model.fc1.out_features,
                             bias=base_model.fc1.bias is not None)
        self.fc1.load_state_dict(base_model.fc1.state_dict(), strict=True)
        self.bn1 = nn.BatchNorm1d(base_model.bn1.num_features,
                                  eps=base_model.bn1.eps,
                                  momentum=base_model.bn1.momentum,
                                  affine=base_model.bn1.affine,
                                  track_running_stats=base_model.bn1.track_running_stats)
        self.bn1.load_state_dict(base_model.bn1.state_dict(), strict=True)
        self.fc2 = nn.Linear(base_model.fc2.in_features,
                             base_model.fc2.out_features,
                             bias=base_model.fc2.bias is not None)
        self.fc2.load_state_dict(base_model.fc2.state_dict(), strict=True)
        self.bn2 = nn.BatchNorm1d(base_model.bn2.num_features,
                                  eps=base_model.bn2.eps,
                                  momentum=base_model.bn2.momentum,
                                  affine=base_model.bn2.affine,
                                  track_running_stats=base_model.bn2.track_running_stats)
        self.bn2.load_state_dict(base_model.bn2.state_dict(), strict=True)
        self.fc3 = nn.Linear(base_model.fc3.in_features,
                             base_model.fc3.out_features,
                             bias=base_model.fc3.bias is not None)
        self.fc3.load_state_dict(base_model.fc3.state_dict(), strict=True)
        self.dropout = nn.Dropout(base_model.dropout.p)
        self.relu1 = nn.ReLU(inplace=False)
        self.relu2 = nn.ReLU(inplace=False)
        self.quant = quantization.QuantStub()
        self.dequant = quantization.DeQuantStub()

    def forward(self, points):
        x = points.transpose(1, 2)
        global_feature, _, trans_feat = self.encoder(x)
        x = self.fc1(global_feature)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.quant(x)
        x = self.fc2(x)
        x = self.dequant(x)
        x = self.dropout(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        regularization_loss = feature_transform_regularizer(trans_feat) * self.reg_weight
        return {
            'logits': x,
            'regularization_loss': regularization_loss,
        }


class QuantizablePointNetSegmentation(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.reg_weight = base_model.reg_weight
        self.num_seg_classes = base_model.num_seg_classes
        self.num_cls = base_model.num_cls
        self.encoder = base_model.encoder
        self.conv1 = nn.Conv1d(base_model.conv1.in_channels,
                               base_model.conv1.out_channels,
                               base_model.conv1.kernel_size,
                               bias=base_model.conv1.bias is not None)
        self.conv1.load_state_dict(base_model.conv1.state_dict(), strict=True)
        self.conv2 = nn.Conv1d(base_model.conv2.in_channels,
                               base_model.conv2.out_channels,
                               base_model.conv2.kernel_size,
                               bias=base_model.conv2.bias is not None)
        self.conv2.load_state_dict(base_model.conv2.state_dict(), strict=True)
        self.conv3 = nn.Conv1d(base_model.conv3.in_channels,
                               base_model.conv3.out_channels,
                               base_model.conv3.kernel_size,
                               bias=base_model.conv3.bias is not None)
        self.conv3.load_state_dict(base_model.conv3.state_dict(), strict=True)
        self.conv4 = nn.Conv1d(base_model.conv4.in_channels,
                               base_model.conv4.out_channels,
                               base_model.conv4.kernel_size,
                               bias=base_model.conv4.bias is not None)
        self.conv4.load_state_dict(base_model.conv4.state_dict(), strict=True)
        self.bn1 = nn.BatchNorm1d(base_model.bn1.num_features,
                                  eps=base_model.bn1.eps,
                                  momentum=base_model.bn1.momentum,
                                  affine=base_model.bn1.affine,
                                  track_running_stats=base_model.bn1.track_running_stats)
        self.bn1.load_state_dict(base_model.bn1.state_dict(), strict=True)
        self.bn2 = nn.BatchNorm1d(base_model.bn2.num_features,
                                  eps=base_model.bn2.eps,
                                  momentum=base_model.bn2.momentum,
                                  affine=base_model.bn2.affine,
                                  track_running_stats=base_model.bn2.track_running_stats)
        self.bn2.load_state_dict(base_model.bn2.state_dict(), strict=True)
        self.bn3 = nn.BatchNorm1d(base_model.bn3.num_features,
                                  eps=base_model.bn3.eps,
                                  momentum=base_model.bn3.momentum,
                                  affine=base_model.bn3.affine,
                                  track_running_stats=base_model.bn3.track_running_stats)
        self.bn3.load_state_dict(base_model.bn3.state_dict(), strict=True)
        self.relu1 = nn.ReLU(inplace=False)
        self.relu2 = nn.ReLU(inplace=False)
        self.relu3 = nn.ReLU(inplace=False)
        self.quant = quantization.QuantStub()
        self.dequant = quantization.DeQuantStub()

    def forward(self, points):
        x = points.transpose(1, 2)
        global_feature, point_feature, trans_feat = self.encoder(x)
        num_points = points.size(1)
        cls_one_hot = torch.zeros(points.size(0), self.num_cls, device=points.device)
        cls_feature = cls_one_hot.unsqueeze(-1).repeat(1, 1, num_points)
        global_feature = global_feature.unsqueeze(-1).repeat(1, 1, num_points)
        x = torch.cat([point_feature, global_feature, cls_feature], dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.quant(x)
        x = self.conv2(x)
        x = self.dequant(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        seg_logits = self.conv4(x).transpose(1, 2)
        regularization_loss = feature_transform_regularizer(trans_feat) * self.reg_weight
        return {
            'seg_logits': seg_logits,
            'regularization_loss': regularization_loss,
        }


class QuantizableDGCNNClassifier(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.backbone = base_model.backbone
        self.classifier = copy.deepcopy(base_model.classifier)
        self.quant = quantization.QuantStub()
        self.dequant = quantization.DeQuantStub()

    def forward(self, points):
        features = self.backbone(points)
        x = features['global_feature']
        x = self.classifier[0](x)
        x = self.classifier[1](x)
        x = self.classifier[2](x)
        x = self.classifier[3](x)
        x = self.classifier[4](x)
        x = self.quant(x)
        x = self.classifier[5](x)
        x = self.dequant(x)
        x = self.classifier[6](x)
        x = self.classifier[7](x)
        return {'logits': x}


class QuantizableDGCNNSegmentation(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.num_cls = base_model.num_cls
        self.backbone = base_model.backbone
        self.seg_head = copy.deepcopy(base_model.seg_head)
        self.quant = quantization.QuantStub()
        self.dequant = quantization.DeQuantStub()

    def forward(self, points):
        features = self.backbone(points)
        num_points = points.size(1)
        local_features = torch.cat(features['stage_features'], dim=1)
        cls_one_hot = torch.zeros(points.size(0), self.num_cls, num_points, device=points.device)
        seg_features = torch.cat([local_features, features['fused_features'], cls_one_hot], dim=1)
        x = self.seg_head[0](seg_features)
        x = self.seg_head[1](x)
        x = self.seg_head[2](x)
        x = self.seg_head[3](x)
        x = self.quant(x)
        x = self.seg_head[4](x)
        x = self.dequant(x)
        x = self.seg_head[5](x)
        x = self.seg_head[6](x)
        x = self.seg_head[7](x)
        seg_logits = self.seg_head[8](x).transpose(1, 2)
        return {'seg_logits': seg_logits}


class QuantizablePointNet2Classifier(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.sa1 = base_model.sa1
        self.sa2 = base_model.sa2
        self.sa3 = base_model.sa3
        self.head = copy.deepcopy(base_model.head)
        self.quant = quantization.QuantStub()
        self.dequant = quantization.DeQuantStub()

    def forward(self, points):
        x = points.transpose(1, 2)
        x, _ = self.sa1(x)
        x, _ = self.sa2(x)
        x, _ = self.sa3(x)
        x = x[:, :, 0]
        x = self.head[0](x)
        x = self.head[1](x)
        x = self.head[2](x)
        x = self.quant(x)
        x = self.head[3](x)
        x = self.dequant(x)
        x = self.head[4](x)
        return {'logits': x}


class QuantizablePointNet2Segmentation(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.num_cls = base_model.num_cls
        self.num_seg_classes = base_model.num_seg_classes
        self.sa1 = base_model.sa1
        self.sa2 = base_model.sa2
        self.sa3 = base_model.sa3
        self.fp3 = base_model.fp3
        self.fp2 = base_model.fp2
        self.fp1 = base_model.fp1
        self.classifier = copy.deepcopy(base_model.classifier)
        self.quant = quantization.QuantStub()
        self.dequant = quantization.DeQuantStub()

    def forward(self, points):
        x = points.transpose(1, 2)
        l1_global, l1_skip = self.sa1(x)
        l2_global, l2_skip = self.sa2(l1_skip)
        l3_global, _ = self.sa3(l2_skip)
        x = self.fp3(l3_global, l2_skip)
        x = self.fp2(x, l1_skip)
        cls_one_hot = torch.zeros(points.size(0), self.num_cls, points.size(1), device=points.device)
        x = self.fp1(x, torch.cat([points.transpose(1, 2), cls_one_hot], dim=1))
        x = self.classifier[0](x)
        x = self.classifier[1](x)
        x = self.classifier[2](x)
        x = self.quant(x)
        x = self.classifier[3](x)
        x = self.dequant(x)
        seg_logits = x.transpose(1, 2)
        return {'seg_logits': seg_logits}


class QuantizedPointNetClassifier(nn.Module):
    def __init__(self, encoder, fc1, bn1, fc2, bn2, fc3, reg_weight):
        super().__init__()
        self.encoder = encoder
        self.fc1 = fc1
        self.bn1 = bn1
        self.fc2 = fc2
        self.bn2 = bn2
        self.fc3 = fc3
        self.reg_weight = reg_weight
        self.relu1 = nn.ReLU(inplace=False)
        self.relu2 = nn.ReLU(inplace=False)
        self.quant = quantization.QuantStub()
        self.dequant = quantization.DeQuantStub()
        self.manual_quantized_inference = False

    def forward(self, points):
        x = points.transpose(1, 2)
        global_feature, _, trans_feat = self.encoder(x)
        x = self.fc1(global_feature)
        x = self.bn1(x)
        x = self.relu1(x)
        if self.manual_quantized_inference:
            x = x.to(dtype=torch.float32, device='cpu')
            x_min = float(x.min().item()) if x.numel() > 0 else 0.0
            x_max = float(x.max().item()) if x.numel() > 0 else 0.0
            value_range = max(x_max - x_min, 1e-8)
            scale = value_range / 255.0
            zero_point = int(round(-x_min / scale))
            zero_point = max(0, min(255, zero_point))
            x = torch.quantize_per_tensor(x, scale=scale, zero_point=zero_point, dtype=torch.quint8)
            x = self.fc2(x)
            x = x.dequantize()
        else:
            x = self.quant(x)
            x = self.fc2(x)
            x = self.dequant(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        regularization_loss = feature_transform_regularizer(trans_feat) * self.reg_weight
        return {
            'logits': x,
            'regularization_loss': regularization_loss,
        }


class PointCloudQATConvertedModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, points):
        outputs = self.model(points)
        if isinstance(outputs, dict):
            if 'logits' in outputs:
                return outputs['logits']
            if 'seg_logits' in outputs:
                return outputs['seg_logits']
        return outputs



def is_qat_supported_model(config):
    return (config.model.type, config.model.name) in SUPPORTED_POINT_CLOUD_QAT_MODELS



def convert_model_to_qat_compatible(config, model):
    if not is_qat_supported_model(config):
        raise ValueError(f'QAT is not supported for point-cloud model {(config.model.type, config.model.name)}')
    if config.model.type == 'pointnet' and config.model.name == 'pointnet_cls':
        return QuantizablePointNetClassifier(model)
    if config.model.type == 'pointnet' and config.model.name == 'pointnet_seg':
        return QuantizablePointNetSegmentation(model)
    if config.model.type == 'pointnet2' and config.model.name == 'pointnet2_cls':
        return QuantizablePointNet2Classifier(model)
    if config.model.type == 'pointnet2' and config.model.name == 'pointnet2_seg':
        return QuantizablePointNet2Segmentation(model)
    if config.model.type == 'dgcnn' and config.model.name == 'dgcnn_cls':
        return QuantizableDGCNNClassifier(model)
    if config.model.type == 'dgcnn' and config.model.name == 'dgcnn_seg':
        return QuantizableDGCNNSegmentation(model)
    raise ValueError(f'Unhandled point-cloud QAT model {(config.model.type, config.model.name)}')



def prepare_model_for_qat(config, model):
    model = convert_model_to_qat_compatible(config, model)
    model.train()
    qconfig = create_default_qat_qconfig(config)
    model.qconfig = None
    if hasattr(model, 'encoder'):
        model.encoder.qconfig = None
    if hasattr(model, 'backbone'):
        model.backbone.qconfig = None

    if isinstance(model, QuantizablePointNetClassifier):
        model.fc1.qconfig = None
        model.bn1.qconfig = None
        model.fc2.qconfig = qconfig
        model.dropout.qconfig = None
        model.bn2.qconfig = None
        model.relu1.qconfig = None
        model.relu2.qconfig = None
        model.fc3.qconfig = None
        model.quant.qconfig = qconfig
        model.dequant.qconfig = None
    elif isinstance(model, QuantizablePointNetSegmentation):
        model.conv1.qconfig = None
        model.bn1.qconfig = None
        model.conv2.qconfig = qconfig
        model.bn2.qconfig = None
        model.conv3.qconfig = None
        model.bn3.qconfig = None
        model.conv4.qconfig = None
        model.relu1.qconfig = None
        model.relu2.qconfig = None
        model.relu3.qconfig = None
        model.quant.qconfig = qconfig
        model.dequant.qconfig = None
    elif isinstance(model, QuantizablePointNet2Classifier):
        model.sa1.qconfig = None
        model.sa2.qconfig = None
        model.sa3.qconfig = None
        for module in model.head:
            module.qconfig = None
        model.head[3].qconfig = qconfig
        model.quant.qconfig = qconfig
        model.dequant.qconfig = None
    elif isinstance(model, QuantizablePointNet2Segmentation):
        model.sa1.qconfig = None
        model.sa2.qconfig = None
        model.sa3.qconfig = None
        model.fp3.qconfig = None
        model.fp2.qconfig = None
        model.fp1.qconfig = None
        for module in model.classifier:
            module.qconfig = None
        model.classifier[3].qconfig = qconfig
        model.quant.qconfig = qconfig
        model.dequant.qconfig = None
    elif isinstance(model, QuantizableDGCNNClassifier):
        for module in model.classifier:
            module.qconfig = None
        model.classifier[5].qconfig = qconfig
        model.quant.qconfig = qconfig
        model.dequant.qconfig = None
    elif isinstance(model, QuantizableDGCNNSegmentation):
        for module in model.seg_head:
            module.qconfig = None
        model.seg_head[4].qconfig = qconfig
        model.quant.qconfig = qconfig
        model.dequant.qconfig = None
    else:
        raise ValueError(f'Unhandled point-cloud QAT model type: {type(model)}')

    quantization.prepare_qat(model, inplace=True)
    return model, QATStateController()



def export_qat_model(config, model):
    if hasattr(model, 'module'):
        model = model.module
    exported = copy.deepcopy(model).cpu().eval()
    return PointCloudQATConvertedModel(exported)



def convert_qat_model(config, model):
    if hasattr(model, 'module'):
        model = model.module
    converted = copy.deepcopy(model).cpu().eval()
    converted.manual_quantized_inference = True
    return PointCloudQATConvertedModel(converted)
