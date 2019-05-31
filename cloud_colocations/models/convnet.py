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

class ConvBlock(nn.Module):
    def __init__(self, c_in, c, c_out, w = 3, kw = 3, last = False):
        super().__init__()
        self.modules = []
        if w <= 1:
            self.modules += [nn.Sequential(nn.Conv2d(c_in, c_out, kw, padding = kw // 2),
                                          nn.BatchNorm2d(c_out),
                                          nn.ReLU())]
        else:
            self.modules += [nn.Sequential(nn.Conv2d(c_in, c, kw, padding = kw // 2),
                                          nn.BatchNorm2d(c_out),
                                          nn.ReLU())]

            for i in range(1, w - 1):
                self.modules += [nn.Sequential(nn.Conv2d(c, c, kw, padding = kw // 2),
                                            nn.BatchNorm2d(c_out),
                                            nn.ReLU())]


            if not last:
                self.modules += [nn.Sequential(nn.Conv2d(c, c_out, kw, padding = kw // 2),
                                            nn.BatchNorm2d(c_out),
                                            nn.ReLU())]
            else:
                self.modules += [nn.Conv2d(c, c_out, kw, padding = kw // 2)]

        for i, m in enumerate(self.modules):
            setattr(self, "module_{0}".format(i), m)

    def forward(self, x):
        y = x
        for m in self.modules:
            y = m(y)
        y = torch.cat([y, x], -3)

        self.__output__ = y
        return y

class Downsampling(nn.Module):
    def __init__(self, c_in, c_out, kw):
        super().__init__()
        self.ds = nn.Conv2d(c_in, c_out, kw, padding = kw // 2, stride = 2)

    def forward(self, x):
        x = self.ds(x)
        return x

class Upsampling(nn.Module):
    def __init__(self, c_in, c_out, kw, skip_connection = None):
        super().__init__()
        self.us = nn.ConvTranspose2d(c_in, c_out, kw, padding = kw // 2, output_padding = kw // 2, stride = 2)
        self.skip_connection = skip_connection

    def forward(self, x):
        x = self.us(x)
        if self.skip_connection:
            x2 = self.skip_connection.__output__
            x = torch.cat([x, x2], -3)
        return x

class CNet(nn.Module):

    def __init__(self,
                 c_in, c_out,
                 arch = [(64, 3), (64, 3)],
                 ks = 3):

        super(CNet, self).__init__()
        self.ks = ks

        self.modules = []
        cs = []
        bs = []
        channels_in = c_in

        # First block

        w, l = arch[0]
        self.modules += [ConvBlock(c_in, w, w, self.ks)]
        c_in = w + c_in

        # Remaining blocks
        for (w, l) in arch[1:]:
            cs += [c_in]
            bs += [self.modules[-1]]
            self.modules += [Downsampling(c_in, w, self.ks)]
            self.modules += [ConvBlock(w, w, w, self.ks)]
            c_in = w + w

        # Upsampling
        for c_s, b_s, (w, l) in zip(cs[::-1], bs[::-1], arch[-2::-1]):
            self.modules += [Upsampling(c_in, w, self.ks, b_s)]
            self.modules += [ConvBlock(w + c_s, w, w, self.ks)]
            c_in = w + w + c_s

        self.modules += [nn.Conv2d(c_in, c_out, self.ks, padding = self.ks // 2)]

        for i, m in enumerate(self.modules):
            setattr(self, "module_{0}".format(i), m)

    def forward(self, x):
        for m in self.modules:
            x = m(x)
        return x
