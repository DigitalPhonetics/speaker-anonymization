# Copyright 2020 Johns Hopkins University (Shinji Watanabe)
#                Northwestern Polytechnical University (Pengcheng Guo)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
# Adapted by Florian Lux 2021


from torch import nn


class ConvolutionModule(nn.Module):
    """
    ConvolutionModule in Conformer model.

    Args:
        channels (int): The number of channels of conv layers.
        kernel_size (int): Kernel size of conv layers.

    """

    def __init__(self, channels, kernel_size, activation=nn.ReLU(), bias=True):
        super(ConvolutionModule, self).__init__()
        # kernel_size should be an odd number for 'SAME' padding
        assert (kernel_size - 1) % 2 == 0

        self.pointwise_conv1 = nn.Conv1d(channels, 2 * channels, kernel_size=1, stride=1, padding=0, bias=bias, )
        self.depthwise_conv = nn.Conv1d(channels, channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2, groups=channels, bias=bias, )
        self.norm = nn.GroupNorm(num_groups=32, num_channels=channels)
        self.pointwise_conv2 = nn.Conv1d(channels, channels, kernel_size=1, stride=1, padding=0, bias=bias, )
        self.activation = activation

    def forward(self, x):
        """
        Compute convolution module.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, channels).

        Returns:
            torch.Tensor: Output tensor (#batch, time, channels).

        """
        # exchange the temporal dimension and the feature dimension
        x = x.transpose(1, 2)

        # GLU mechanism
        x = self.pointwise_conv1(x)  # (batch, 2*channel, dim)
        x = nn.functional.glu(x, dim=1)  # (batch, channel, dim)

        # 1D Depthwise Conv
        x = self.depthwise_conv(x)
        x = self.activation(self.norm(x))

        x = self.pointwise_conv2(x)

        return x.transpose(1, 2)
