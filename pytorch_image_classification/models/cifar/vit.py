from ..vit_modules import VisionTransformer
from ..vit_utils import load_vit_pretrained


class Network(VisionTransformer):
    """CIFAR ViT wrapper / CIFAR 场景下的 ViT 包装器。"""

    def __init__(self, config):
        """Build the shared ViT backbone and optionally load pretrained weights / 构建共享 ViT 主干并按配置加载预训练权重。"""
        super().__init__(config)
        load_vit_pretrained(self, config)
