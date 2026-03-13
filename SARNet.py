"""
SARNet Model Definition Module

This module implements the core network architecture of SARNet
(Segment-Amplify-Refine Network) for Camouflaged Object Detection (COD).

Main Components:
- SARNet: Main network based on PVTv2 backbone, integrating OAA and FGC modules
- CBR: Convolution-BatchNorm-ReLU basic building block
- OAA (Object Area Amplification): Multi-level feature fusion module for amplifying target regions
- FGC (Figure-Ground Conversion): Progressive refinement module for fine-grained detection

Reference:
    Go Closer to See Better: Camouflaged Object Detection via
    Object Area Amplification and Figure-Ground Conversion
    https://ieeexplore.ieee.org/document/10065514
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import math
import sys
from mmcv.cnn import build_norm_layer
sys.path.insert(0, '../../')
from mmcv.ops.carafe import CARAFEPack
from pvtv2 import pvt_v2_b0, pvt_v2_b1, pvt_v2_b2, pvt_v2_b2_li, pvt_v2_b3, pvt_v2_b4, pvt_v2_b5


def np2th(weights, conv=False):
    """Convert NumPy weight arrays to PyTorch tensors.

    Args:
        weights: NumPy weight array.
        conv: If True, transpose from HWIO to OIHW format for convolution layers.

    Returns:
        Converted PyTorch tensor.
    """
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


class SARNet(nn.Module):
    """SARNet: Camouflaged Object Detection Network.

    The network adopts an encoder-decoder architecture:
    - Encoder: PVTv2 backbone extracting 4-level multi-scale features
    - Decoder: OAA modules fuse adjacent-level features, then FGC modules
      progressively refine predictions from deep to shallow layers

    Network Pipeline:
        1. PVTv2 extracts 4-level features [layer0, layer1, layer2, layer3]
        2. OAA modules fuse adjacent-level features to produce s1-s5
        3. CBR reduces channels of the deepest features to get s5, generating initial prediction (predict4)
        4. FGC modules progressively refine from deep to shallow: fgc3 -> fgc2 -> fgc1 -> fgc0
        5. All prediction maps are upsampled to the original input resolution

    Args:
        fun_str: PVTv2 backbone variant name, default is 'pvt_v2_b0'.
    """

    def __init__(self, fun_str='pvt_v2_b0'):
        super().__init__()
        # Initialize PVTv2 backbone and obtain embedding dimensions for each stage
        self.backbone, embedding_dims = eval(fun_str)()

        # FGC modules: Figure-Ground Conversion, progressive refinement from deep to shallow
        # fgc3: Processes the deepest two fused features, focuses on background regions
        self.fgc3 = FGC(embedding_dims[3] // 8, embedding_dims[3] // 8, focus_background=True,
                        opr_kernel_size=7, iterations=1)
        # fgc2: Processes intermediate fused features, focuses on background regions
        self.fgc2 = FGC(embedding_dims[1] // 2, embedding_dims[3] // 8, focus_background=True,
                        opr_kernel_size=7, iterations=1)
        # fgc1: Processes shallow fused features, focuses on background regions
        self.fgc1 = FGC(embedding_dims[0] // 2, embedding_dims[1] // 2, focus_background=True,
                        opr_kernel_size=7, iterations=1)
        # fgc0: Final refinement, focuses on foreground regions (focus_background=False)
        self.fgc0 = FGC(embedding_dims[0] // 4, embedding_dims[0] // 2, focus_background=False,
                        opr_kernel_size=7, iterations=1)

        # OAA modules: Object Area Amplification, fusing adjacent-level features
        # oaa0: Fuses features from layer0 and layer1
        self.oaa0 = OAA(cur_in_channels=embedding_dims[0], low_in_channels=embedding_dims[1],
                        out_channels=embedding_dims[0] // 2, cur_scale=1, low_scale=2)
        # oaa1: Fuses features from layer1 and layer2
        self.oaa1 = OAA(cur_in_channels=embedding_dims[1], low_in_channels=embedding_dims[2],
                        out_channels=embedding_dims[1] // 2, cur_scale=1, low_scale=2)
        # oaa2: Fuses features from layer2 and layer3
        self.oaa2 = OAA(cur_in_channels=embedding_dims[2], low_in_channels=embedding_dims[3],
                        out_channels=embedding_dims[3] // 8, cur_scale=1, low_scale=2)

        # CBR for channel reduction on the deepest layer features
        self.cbr = CBR(in_channels=embedding_dims[3], out_channels=embedding_dims[3] // 8,
                       kernel_size=3, stride=1, dilation=1, padding=1)

        # Prediction head: maps features to single-channel prediction maps
        self.predict_conv = nn.Sequential(
            nn.Conv2d(in_channels=embedding_dims[3] // 8, out_channels=1, kernel_size=3, padding=1, stride=1))

        # oaa3: Fuses shallow refined features with deep reduced features for final fine prediction
        self.oaa3 = OAA(cur_in_channels=embedding_dims[0] // 2, low_in_channels=embedding_dims[3] // 8,
                        out_channels=embedding_dims[0] // 4, cur_scale=2, low_scale=16)

    def forward(self, x):
        """Forward pass.

        Args:
            x: Input image tensor of shape (B, 3, H, W).

        Returns:
            tuple: 5 prediction maps at different levels (predict4, predict3, predict2, predict1, predict0),
                   ordered from coarse to fine, all upsampled to the input resolution.
        """
        # Extract multi-scale features via backbone
        layer = self.backbone(x)

        # OAA modules fuse adjacent-level features
        s2 = self.oaa0(layer[0], layer[1])  # Fuse shallow and mid-shallow layers
        s3 = self.oaa1(layer[1], layer[2])  # Fuse mid-shallow and mid-deep layers
        s4 = self.oaa2(layer[2], layer[3])  # Fuse mid-deep and deepest layers
        s5 = self.cbr(layer[3])             # Channel reduction on deepest layer
        s1 = self.oaa3(s2, s5)             # Fuse shallow refined features with deep features

        # Generate initial prediction from the deepest layer
        predict4 = self.predict_conv(s5)

        # FGC modules progressively refine predictions
        fgc3, predict3 = self.fgc3(s4, s5, predict4)
        fgc2, predict2 = self.fgc2(s3, fgc3, predict3)
        fgc1, predict1 = self.fgc1(s2, fgc2, predict2)
        fgc0, predict0 = self.fgc0(s1, fgc1, predict1)

        # Bilinear interpolation to upsample all predictions to the original input size
        predict4 = F.interpolate(predict4, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict3 = F.interpolate(predict3, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict2 = F.interpolate(predict2, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict1 = F.interpolate(predict1, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict0 = F.interpolate(predict0, size=x.size()[2:], mode='bilinear', align_corners=True)

        return predict4, predict3, predict2, predict1, predict0


class CBR(nn.Module):
    """Convolution-BatchNorm-ReLU basic building block.

    Performs the standard Conv2d -> BatchNorm -> ReLU operation sequence.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Size of the convolution kernel, default 3.
        stride: Stride of the convolution, default 1.
        padding: Padding size, default 1.
        dilation: Dilation rate, default 1.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1):
        super(CBR, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation)
        self.norm_cfg = {'type': 'BN', 'requires_grad': True}
        _, self.bn = build_norm_layer(self.norm_cfg, out_channels)

    def forward(self, x):
        """Forward pass: Conv -> BatchNorm -> ReLU."""
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x, inplace=True)
        return x


class OAA(nn.Module):
    """Object Area Amplification (OAA) Module.

    Fuses current-level features with lower-resolution (deeper) features to amplify
    the representation of target regions, enabling the network to better capture
    spatial information of camouflaged objects.

    Fusion Pipeline:
        1. Apply convolution transforms to current-level and low-resolution features separately
        2. Upsample both feature maps to the target resolution
        3. Concatenate along the channel dimension and fuse

    Args:
        cur_in_channels: Number of input channels for the current level.
        low_in_channels: Number of input channels for the lower-resolution level.
        out_channels: Number of output channels.
        cur_scale: Upsampling factor for the current-level features.
        low_scale: Upsampling factor for the lower-resolution features.
    """

    def __init__(self, cur_in_channels=64, low_in_channels=32, out_channels=16, cur_scale=2, low_scale=1):
        super(OAA, self).__init__()
        self.cur_in_channels = cur_in_channels

        # Current-level feature transform: Conv -> BN -> GELU
        self.cur_conv = nn.Sequential(
            nn.Conv2d(in_channels=cur_in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.GELU()
        )
        # Low-resolution feature transform: Conv -> BN -> GELU
        self.low_conv = nn.Sequential(
            nn.Conv2d(in_channels=low_in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.GELU()
        )

        self.cur_scale = cur_scale
        self.low_scale = low_scale

        # Fusion convolution: reduces concatenated double-channel features
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels=2 * out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.GELU()
        )

    def forward(self, x_cur, x_low):
        """Forward pass.

        Args:
            x_cur: Current-level features.
            x_low: Lower-resolution (deeper layer) features.

        Returns:
            Fused feature map.
        """
        # Transform and upsample separately (using bicubic interpolation)
        x_cur = self.cur_conv(x_cur)
        x_cur = F.interpolate(x_cur, scale_factor=self.cur_scale, mode='bicubic', align_corners=False)

        x_low = self.low_conv(x_low)
        x_low = F.interpolate(x_low, scale_factor=self.low_scale, mode='bicubic', align_corners=False)

        # Concatenate along channel dimension and fuse
        x = torch.cat((x_cur, x_low), dim=1)
        x = self.out_conv(x)
        return x


import numpy as np
import cv2


def get_open_map(input, kernel_size, iterations):
    """Generate an open region map via morphological dilation.

    Applies dilation to the input prediction map to expand predicted regions,
    used in the FGC module to identify background regions that the deeper layers
    missed but shallower layers should attend to.

    Args:
        input: Input tensor of shape (B, 1, H, W).
        kernel_size: Kernel size for the dilation operation.
        iterations: Number of dilation iterations.

    Returns:
        Dilated tensor with the same shape as the input.
    """
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    open_map_list = map(
        lambda i: cv2.dilate(i.permute(1, 2, 0).detach().numpy(), kernel=kernel, iterations=iterations),
        input.cpu()
    )
    open_map_tensor = torch.from_numpy(np.array(list(open_map_list)))
    return open_map_tensor.unsqueeze(1).cuda()


class Basic_Conv(nn.Module):
    """Basic convolution block: Conv -> BN -> ReLU.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Size of the convolution kernel.
        stride: Stride of the convolution.
        padding: Padding size.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Basic_Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, padding=padding, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        """Forward pass: Conv -> BatchNorm -> ReLU."""
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class FGC(nn.Module):
    """Figure-Ground Conversion (FGC) Module.

    Leverages deep-layer prediction maps to guide selective attention on shallow-layer
    features for foreground-background region differentiation, achieving coarse-to-fine
    prediction refinement.

    Operating Modes:
    - focus_background=True: Attends to background regions missed by deeper layers
      (obtained by subtracting the prediction map from the dilated map)
    - focus_background=False: Directly attends to foreground regions detected by deeper layers
      (uses the prediction map as attention weights)

    Args:
        channel1: Number of channels for current-level features.
        channel2: Number of channels for deeper-level features.
        focus_background: Whether to focus on background regions, default True.
        opr_kernel_size: Kernel size for morphological dilation, default 3.
        iterations: Number of dilation iterations, default 1.
    """

    def __init__(self, channel1, channel2, focus_background=True, opr_kernel_size=3, iterations=1):
        super(FGC, self).__init__()
        self.channel1 = channel1
        self.channel2 = channel2
        self.focus_background = focus_background

        # Deep feature upsampling path: Conv channel reduction -> BN -> ReLU -> 2x upsample
        self.up = nn.Sequential(
            nn.Conv2d(self.channel2, self.channel1, 7, 1, 3),
            nn.BatchNorm2d(self.channel1),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )
        # Prediction map upsampling with Sigmoid normalization
        self.input_map = nn.Sequential(nn.UpsamplingBilinear2d(scale_factor=2), nn.Sigmoid())
        # Auxiliary module for computing dilated region difference map (placeholder, no size change)
        self.increase_input_map = nn.Sequential(nn.UpsamplingBilinear2d(scale_factor=1))
        # Output prediction convolution
        self.output_map = nn.Conv2d(self.channel1, 1, 7, 1, 3)
        # Learnable fusion weight parameter beta
        self.beta = nn.Parameter(torch.ones(1))

        # Attention-weighted feature convolution for current level
        self.conv2 = nn.Conv2d(in_channels=self.channel1, out_channels=self.channel1, kernel_size=3, padding=1,
                               stride=1)
        # Feature fusion convolution sequence: concatenate -> reduce -> refine twice
        self.conv_cur_dep1 = Basic_Conv(2 * self.channel1, self.channel1, 3, 1, 1)
        self.conv_cur_dep2 = Basic_Conv(in_channels=self.channel1, out_channels=self.channel1, kernel_size=3,
                                        padding=1, stride=1)
        self.conv_cur_dep3 = Basic_Conv(in_channels=self.channel1, out_channels=self.channel1, kernel_size=3,
                                        padding=1, stride=1)

        self.opr_kernel_size = opr_kernel_size
        self.iterations = iterations

    def forward(self, cur_x, dep_x, in_map):
        """Forward pass.

        Args:
            cur_x: Current-level (shallow) features.
            dep_x: Deeper-level features.
            in_map: Deep-layer prediction map (before Sigmoid).

        Returns:
            tuple: (refine2, output_map)
                - refine2: Refined feature map.
                - output_map: Current-level prediction map.
        """
        # Upsample deep features to current-level resolution
        dep_x = self.up(dep_x)
        # Upsample and normalize deep prediction map
        input_map = self.input_map(in_map)

        if self.focus_background:
            # Background attention mode: compute difference between dilated and predicted regions
            # to focus on areas missed by deeper layers
            self.increase_map = self.increase_input_map(
                get_open_map(input_map, self.opr_kernel_size, self.iterations) - input_map
            )
            # Attend to regions in current level that deeper layers did not focus on
            b_feature = cur_x * self.increase_map
        else:
            # Foreground attention mode: enhance attention on regions detected by deeper layers
            b_feature = cur_x * input_map

        # Apply convolution to attention-weighted features
        fn = self.conv2(b_feature)

        # Fuse upsampled deep features with attention features (beta is a learnable weight)
        refine2 = self.conv_cur_dep1(torch.cat((dep_x, self.beta * fn), dim=1))
        # Two consecutive convolution refinements
        refine2 = self.conv_cur_dep2(refine2)
        refine2 = self.conv_cur_dep3(refine2)

        # Generate current-level prediction map
        output_map = self.output_map(refine2)

        return refine2, output_map


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    from thop import profile

    # Compute model FLOPs and parameter count
    net = SARNet('pvt_v2_b3').cuda()
    data = torch.randn(1, 3, 672, 672).cuda()
    flops, params = profile(net, (data,))
    print('flops: %.2f G, params: %.2f M' % (flops / (1024 * 1024 * 1024), params / (1024 * 1024)))
    y = net(data)
    for i in y:
        print(i.shape)