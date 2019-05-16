import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):

    def __init__(self, input_channels, output_channels, arch = [64, 64, 64]):
        super(ConvNet, self).__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.num_channels = input_channels

        in_channels = self.input_channels
        out_channels = self.output_channels

        self.layers = []
        for i, c in enumerate(arch):
            self.layers += [nn.Sequential(nn.Conv2d(in_channels = in_channels,
                                                    out_channels = c,
                                                    kernel_size = 3,
                                                    padding = 1),
                                          nn.BatchNorm2d(c))]
            setattr(self, "layer_" + str(i), self.layers[-1])
            in_channels = c


        self.layers += [nn.Sequential(nn.Conv2d(in_channels = in_channels,
                                                out_channels = self.output_channels,
                                                kernel_size = 3,
                                                padding = 1))]
        setattr(self, "layer_" + str(i), self.layers[-1])


    def forward(self, input):

        x = input
        for l in self.layers[:-1]:
            x = F.relu(l(x))

        l = self.layers[-1]
        return l(x)
