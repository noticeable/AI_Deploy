import pathlib

import torch
import torch.nn as nn


class DetectionExportWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images):
        detections = self.model(images)
        batch_detections = []
        max_detections = 0
        for detection in detections:
            boxes = detection['boxes']
            scores = detection['scores'].to(boxes.dtype).unsqueeze(-1)
            labels = detection['labels'].to(boxes.dtype).unsqueeze(-1)
            packed = torch.cat([boxes, scores, labels], dim=-1)
            batch_detections.append(packed)
            max_detections = max(max_detections, packed.size(0))

        padded_detections = []
        for packed in batch_detections:
            if packed.size(0) < max_detections:
                padding = packed.new_zeros((max_detections - packed.size(0), packed.size(1)))
                packed = torch.cat([packed, padding], dim=0)
            padded_detections.append(packed)
        return torch.stack(padded_detections, dim=0)


class DetectionExporter:
    def __init__(self, config):
        self.config = config

    def export(self, model):
        if str(getattr(self.config.eval, 'nms_type', 'hard')).lower() == 'soft':
            raise ValueError('Soft-NMS is not supported during ONNX export. Set eval.nms_type to hard before exporting.')
        output_file = self.config.export.output_file or 'detection_model.onnx'
        output_path = pathlib.Path(output_file)
        output_path.parent.mkdir(exist_ok=True, parents=True)
        model.eval()
        export_model = DetectionExportWrapper(model)
        dummy = torch.randn(1,
                            self.config.dataset.n_channels,
                            self.config.dataset.image_size,
                            self.config.dataset.image_size,
                            device=next(model.parameters()).device)
        dynamic_axes = None
        if self.config.export.dynamic_axes:
            dynamic_axes = {
                'images': {0: 'batch_size'},
                'detections': {0: 'batch_size', 1: 'num_detections'},
            }
        torch.onnx.export(export_model,
                          dummy,
                          output_path.as_posix(),
                          opset_version=self.config.export.opset,
                          input_names=['images'],
                          output_names=['detections'],
                          dynamic_axes=dynamic_axes)
        return output_path


def create_exporter(config):
    return DetectionExporter(config)
