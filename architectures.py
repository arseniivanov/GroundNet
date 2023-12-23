import torch.nn as nn
# Standard Convolutional Layer
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

# Depthwise Convolutional Layer
class DepthwiseConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super(DepthwiseConvLayer, self).__init__()
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=bias)
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.relu(x)
        x = self.pointwise_conv(x)
        x = self.bn(x)
        return x
    
class MCUnetBackbone(nn.Module):
    def __init__(self):
        super(MCUnetBackbone, self).__init__()
        self.layers = nn.ModuleList()

        # First Convolutional Layer with BatchNorm and ReLU6
        self.layers.append(ConvLayer(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1, bias=True))

        # Depthwise Separable Convolution Layer 1
        self.layers.append(DepthwiseConvLayer(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=True))

        # Second Convolutional Layer with BatchNorm and ReLU6
        self.layers.append(ConvLayer(in_channels=16, out_channels=32, kernel_size=1, stride=1, padding=0, bias=True))

        # Padding Layer
        # We need to implement a custom padding layer because PyTorch does not support asymmetric padding directly
        #self.layers.append(nn.ZeroPad2d((1, 1, 1, 1)))

        # Depthwise Separable Convolution Layer 2
        self.layers.append(DepthwiseConvLayer(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=0, bias=True))

        # Third Convolutional Layer with BatchNorm and ReLU6
        self.layers.append(ConvLayer(in_channels=32, out_channels=16, kernel_size=1, stride=1, padding=0, bias=True))

        # Fourth Convolutional Layer with BatchNorm and ReLU6
        self.layers.append(ConvLayer(in_channels=16, out_channels=64, kernel_size=1, stride=1, padding=0, bias=True))

         # Depthwise Separable Convolution Layer 3
        self.skip_conv = DepthwiseConvLayer(in_channels=64, out_channels=16, kernel_size=3, stride=2, padding=0, bias=True)

        # Fifth Convolutional Layer with BatchNorm and ReLU6 for the skip connection
        self.layers.append(ConvLayer(in_channels=16, out_channels=48, kernel_size=1, stride=1, padding=0, bias=True))

        # Depthwise Separable Convolution Layer 4
        self.layers.append(DepthwiseConvLayer(in_channels=48, out_channels=16, kernel_size=3, stride=1, padding=1, bias=True))
        
        self.max_pool = nn.MaxPool2d(2)

        self.final_conv = nn.Conv2d(in_channels=16, out_channels=3, kernel_size=1)

    def forward(self, x):
        # Process the initial layers

        for layer in self.layers[:-2]:  # This will exclude the last three layers intended for skip connection
            x = layer(x)

        # Store the output for the skip connection
        skip_conn = self.skip_conv(x)

        # Process the final layer (Seventh Convolutional Layer)

        x = self.layers[-2](skip_conn)
        x = self.layers[-1](x)

        # Apply the skip connection
        x = x + skip_conn

        #Add more convs to reduce dimensionality      

        x = self.max_pool(x)
        x = self.final_conv(x)

        return x
    