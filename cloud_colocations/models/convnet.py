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

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, depth, kernel_width = 3):
        super(Encoder, self).__init__()

        self.modules = []

        self.modules += [nn.Conv2d(in_channels, out_channels, kernel_width, padding = 1)]
        setattr(self, "conv2d_" + str(0), self.modules[-1])
        self.modules += [nn.BatchNorm2d(out_channels)]
        setattr(self, "batch_norm_" + str(0), self.modules[-1])
        self.modules += [nn.ReLU()]

        for i in range(1, depth):
            self.modules += [nn.Conv2d(out_channels, out_channels, kernel_width, padding = 1)]
            setattr(self, "conv2d_" + str(i), self.modules[-1])
            self.modules += [nn.BatchNorm2d(out_channels)]
            setattr(self, "batch_norm_" + str(i), self.modules[-1])
            self.modules += [nn.ReLU()]

        self.modules += [nn.Conv2d(out_channels, out_channels, kernel_width, padding = 1, stride = 2)]

    def forward(self, x):
        for m in self.modules:
            x = m(x)
        return x

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, depth, kernel_width = 3):
        super(Decoder, self).__init__()

        self.modules = []

        self.modules += [nn.ConvTranspose2d(in_channels, in_channels, kernel_width, padding = 1,
                                            output_padding = 1, stride = 2)]
        setattr(self, "conv2d_" + str(0), self.modules[-1])
        self.modules += [nn.BatchNorm2d(out_channels)]
        setattr(self, "batch_norm_" + str(0), self.modules[-1])
        self.modules += [nn.ReLU()]

        for i in range(1, depth):
            self.modules += [nn.Conv2d(out_channels, out_channels, kernel_width, padding = 1)]
            setattr(self, "conv2d_" + str(i), self.modules[-1])
            self.modules += [nn.BatchNorm2d(out_channels)]
            setattr(self, "batch_norm_" + str(i), self.modules[-1])
            self.modules += [nn.ReLU()]

    def forward(self, x, x2):
        x = self.modules[0](x)
        m, n = x2.size()[-2:]

        slices = [slice(s) for s in x.size()[:-2]]
        slices += [slice(0, m)]
        slices += [slice(0, n)]
        x = torch.cat([x[tuple(slices)], x2], dim = -3)
        for m in self.modules[1:]:
            x = m(x)
        return x

class CNet(nn.Module):

    def _make_center(self, c_center, l_center):
        modules = [nn.Conv2d(in_channels  = c_center, out_channels = c_center, kernel_size = self.ks, padding = 1),
                   nn.BatchNorm2d(c_center),
                   nn.ReLU()] * l_center
        self.center = nn.Sequential(*modules)

    def __init__(self,
                 channels_in,
                 channels_out,
                 arch = [(64, 2), (128, 2)],
                 l_center = 2,
                 c_center = 128,
                 ks = 3):
        super(CNet, self).__init__()
        self.ks = ks

        #
        # Encoder
        #
        cs = []

        self.encoders = []
        in_channels = channels_in
        for i, (c, d) in enumerate(arch):
            cs += [in_channels]
            self.encoders += [Encoder(in_channels, c, d)]
            setattr(self, "encoder_" + str(i), self.encoders[-1])
            in_channels = c

        #
        # Some convolutions in center
        #

        self._make_center(in_channels, l_center)

        #
        # Decoder
        #

        self.decoders = []
        for i, ((c, d), c2) in enumerate(zip(arch[::-1], cs[::-1])):
            self.decoders += [Decoder(in_channels, in_channels + c2, d)]
            setattr(self, "decoder_" + str(i), self.decoders[-1])
            in_channels = in_channels + c2

        self.last = nn.Conv2d(in_channels = in_channels, out_channels = channels_out,
                              kernel_size = self.ks, padding = 1)


    def forward(self, x):
        xs = []
        for e in self.encoders:
            xs += [x]
            x = e(x)

        x = self.center(x)

        for d, x2 in zip(self.decoders, xs[::-1]):
            x = d(x, x2)

        x = self.last(x)
        return x
