import torch
import torch.nn as nn
import torch.nn.functional as F

from ..initializer import create_initializer


class BasicBlock(nn.Module):
    """Basic DenseNet growth block / DenseNet 基础增长块。"""

    def __init__(self, in_channels, out_channels, drop_rate):
        """Build a single 3x3 dense block unit / 构建单个 3x3 dense block 单元。"""
        super().__init__()

        self.drop_rate = drop_rate

        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=3,
                              stride=1,
                              padding=1,
                              bias=False)

    def forward(self, x):
        """Append newly generated features to the input tensor / 将新生成特征与输入特征拼接。"""
        y = self.conv(F.relu(self.bn(x), inplace=True))
        if self.drop_rate > 0:
            y = F.dropout(y,
                          p=self.drop_rate,
                          training=self.training,
                          inplace=False)
        return torch.cat([x, y], dim=1)


class BottleneckBlock(nn.Module):
    """Bottleneck DenseNet growth block / DenseNet 瓶颈增长块。"""

    def __init__(self, in_channels, out_channels, drop_rate):
        """Build a 1x1 + 3x3 bottleneck dense unit / 构建 1x1 + 3x3 的瓶颈 dense 单元。"""
        super().__init__()

        self.drop_rate = drop_rate

        bottleneck_channels = out_channels * 4

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels,
                               bottleneck_channels,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)

        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        self.conv2 = nn.Conv2d(bottleneck_channels,
                               out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)

    def forward(self, x):
        """Expand then generate features before concatenation / 先扩展通道再生成特征并拼接回输入。"""
        y = self.conv1(F.relu(self.bn1(x), inplace=True))
        if self.drop_rate > 0:
            y = F.dropout(y,
                          p=self.drop_rate,
                          training=self.training,
                          inplace=False)
        y = self.conv2(F.relu(self.bn2(y), inplace=True))
        if self.drop_rate > 0:
            y = F.dropout(y,
                          p=self.drop_rate,
                          training=self.training,
                          inplace=False)
        return torch.cat([x, y], dim=1)


class TransitionBlock(nn.Module):
    """Transition block between DenseNet stages / DenseNet stage 之间的过渡块。"""

    def __init__(self, in_channels, out_channels, drop_rate):
        """Build a channel compression and downsampling block / 构建通道压缩与下采样模块。"""
        super().__init__()

        self.drop_rate = drop_rate

        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=1,
                              stride=1,
                              padding=0,
                              bias=False)

    def forward(self, x):
        """Compress channels and halve spatial resolution / 压缩通道并将空间分辨率减半。"""
        x = self.conv(F.relu(self.bn(x), inplace=True))
        if self.drop_rate > 0:
            x = F.dropout(x,
                          p=self.drop_rate,
                          training=self.training,
                          inplace=False)
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class Network(nn.Module):
    """DenseNet image classifier / DenseNet 图像分类网络。"""

    def __init__(self, config):
        """Build DenseNet stages and classifier head / 构建 DenseNet 各 stage 与分类头。"""
        super().__init__()

        model_config = config.model.densenet
        block_type = model_config.block_type
        n_blocks = model_config.n_blocks
        self.growth_rate = model_config.growth_rate
        self.drop_rate = model_config.drop_rate
        self.compression_rate = model_config.compression_rate

        assert block_type in ['basic', 'bottleneck']
        if block_type == 'basic':
            block = BasicBlock
        else:
            block = BottleneckBlock

        in_channels = [2 * self.growth_rate]
        for index in range(4):
            denseblock_out_channels = int(in_channels[-1] +
                                          n_blocks[index] * self.growth_rate)
            if index < 3:
                transitionblock_out_channels = int(denseblock_out_channels *
                                                   self.compression_rate)
            else:
                transitionblock_out_channels = denseblock_out_channels
            in_channels.append(transitionblock_out_channels)

        self.conv = nn.Conv2d(config.dataset.n_channels,
                              in_channels[0],
                              kernel_size=7,
                              stride=2,
                              padding=3,
                              bias=False)
        self.bn = nn.BatchNorm2d(in_channels[0])
        self.stage1 = self._make_stage(in_channels[0], n_blocks[0], block,
                                       True)
        self.stage2 = self._make_stage(in_channels[1], n_blocks[1], block,
                                       True)
        self.stage3 = self._make_stage(in_channels[2], n_blocks[2], block,
                                       True)
        self.stage4 = self._make_stage(in_channels[3], n_blocks[3], block,
                                       False)
        self.bn_last = nn.BatchNorm2d(in_channels[4])

        # compute conv feature size
        with torch.no_grad():
            dummy_data = torch.zeros(
                (1, config.dataset.n_channels, config.dataset.image_size,
                 config.dataset.image_size),
                dtype=torch.float32)
            self.feature_size = self._forward_conv(dummy_data).view(
                -1).shape[0]

        self.fc = nn.Linear(self.feature_size, config.dataset.n_classes)

        # initialize weights
        initializer = create_initializer(config.model.init_mode)
        self.apply(initializer)

    def _make_stage(self, in_channels, n_blocks, block, add_transition_block):
        """Assemble one DenseNet stage with optional transition / 组装单个 DenseNet stage，并按需追加 transition。"""
        stage = nn.Sequential()
        for index in range(n_blocks):
            stage.add_module(
                f'block{index + 1}',
                block(in_channels + index * self.growth_rate, self.growth_rate,
                      self.drop_rate))
        if add_transition_block:
            in_channels = int(in_channels + n_blocks * self.growth_rate)
            out_channels = int(in_channels * self.compression_rate)
            stage.add_module(
                'transition',
                TransitionBlock(in_channels, out_channels, self.drop_rate))
        return stage

    def _forward_conv(self, x):
        """Run the convolutional DenseNet backbone / 执行 DenseNet 卷积主干前向。"""
        x = F.relu(self.bn(self.conv(x)), inplace=True)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.bn_last(x)
        x = F.adaptive_avg_pool2d(x, output_size=1)
        return x

    def forward(self, x):
        """Run end-to-end classification / 执行端到端分类前向。"""
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
