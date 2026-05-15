import torch
import torch.ao.nn.intrinsic.qat as intrinsic_qat
import torch.ao.quantization as quantization
from torch.ao.quantization import QConfig
from torch.ao.quantization.fake_quantize import FusedMovingAvgObsFakeQuantize
from torch.ao.quantization.observer import MovingAverageMinMaxObserver, MovingAveragePerChannelMinMaxObserver


class QATStateController:
    def __init__(self):
        self.bn_frozen = False
        self.observer_disabled = False


def is_qat_enabled(config):
    return bool(getattr(config, 'qat', None) and config.qat.enabled)


def _create_fbgemm_qat_qconfig():
    activation = FusedMovingAvgObsFakeQuantize.with_args(
        observer=MovingAverageMinMaxObserver,
        quant_min=0,
        quant_max=255,
        reduce_range=True,
    )
    weight = FusedMovingAvgObsFakeQuantize.with_args(
        observer=MovingAveragePerChannelMinMaxObserver,
        quant_min=-128,
        quant_max=127,
        dtype=torch.qint8,
        qscheme=torch.per_channel_symmetric,
    )
    return QConfig(activation=activation, weight=weight)


def create_default_qat_qconfig(config):
    backend = getattr(config.qat, 'backend', 'fbgemm')
    torch.backends.quantized.engine = backend
    qconfig_name = getattr(config.qat, 'qconfig', 'default')
    if qconfig_name == 'default':
        if backend == 'fbgemm':
            return _create_fbgemm_qat_qconfig()
        return quantization.get_default_qat_qconfig(backend)
    raise ValueError(f'Unsupported qat.qconfig: {qconfig_name}')


def freeze_bn_stats(module):
    if hasattr(module, 'freeze_bn_stats'):
        module.freeze_bn_stats()
    elif isinstance(module, intrinsic_qat.ConvBn2d):
        module.freeze_bn_stats()
    elif isinstance(module, intrinsic_qat.ConvBnReLU2d):
        module.freeze_bn_stats()


def apply_qat_epoch_controls(config, model, epoch, state):
    if state is None:
        return
    if (not state.bn_frozen) and epoch >= int(config.qat.freeze_bn_epoch):
        model.apply(freeze_bn_stats)
        state.bn_frozen = True
    if (not state.observer_disabled) and epoch >= int(config.qat.disable_observer_epoch):
        model.apply(quantization.disable_observer)
        state.observer_disabled = True


def quantize_input_tensor(x: torch.Tensor) -> torch.Tensor:
    x = x.detach().to(dtype=torch.float32, device='cpu')
    x_min = float(x.min().item()) if x.numel() > 0 else 0.0
    x_max = float(x.max().item()) if x.numel() > 0 else 0.0
    value_range = max(x_max - x_min, 1e-8)
    scale = value_range / 255.0
    zero_point = int(round(-x_min / scale))
    zero_point = max(0, min(255, zero_point))
    return torch.quantize_per_tensor(x, scale=scale, zero_point=zero_point, dtype=torch.quint8)


def ensure_cpu_float_tensor(x):
    if isinstance(x, torch.Tensor) and x.is_quantized:
        x = x.dequantize()
    if isinstance(x, torch.Tensor):
        x = x.to(dtype=torch.float32, device='cpu')
    return x


def unwrap_model(model):
    return model.module if hasattr(model, 'module') else model
