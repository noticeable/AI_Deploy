from ..vit_modules import VisionTransformer
from ..vit_utils import load_vit_pretrained


class Network(VisionTransformer):
    """ImageNet DeiT wrapper / ImageNet 场景下的 DeiT 包装器。"""

    def __init__(self, config):
        """Build the shared DeiT backbone and optionally load pretrained weights / 构建共享 DeiT 主干并按配置加载预训练权重。"""
        super().__init__(config, model_name='deit')
        load_vit_pretrained(self, config, model_name='deit')
