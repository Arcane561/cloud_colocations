import torch
import torch.nn as nn
import torch.nn.functional as F

class SegNet(nn.Module):

    def __init__(self, input_channels, output_channels, arch = [[64], [128], [256]]):
        super(SegNet, self).__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.num_channels = input_channels


        def make_encoder(in_channels, out_channels):
            return nn.Sequential(nn.Conv2d(in_channels = in_channels,
                                           out_channels = out_channels,
                                           kernel_size = 3,
                                           padding = 1),
                                 nn.BatchNorm2d(out_channels))

        def make_decoder(in_channels, out_channels, final = False):
            layers = [nn.ConvTranspose2d(in_channels = in_channels,
                                         out_channels = out_channels,
                                         kernel_size = 3,
                                         padding = 1)]
            if not final:
                layers += [nn.BatchNorm2d(out_channels)]

            return nn.Sequential(*layers)




        in_channels = 0
        out_channels = self.input_channels
        final = True

        self.encoders = []
        self.decoders = []
        for i, dims in enumerate(arch):
            self.encoders += [[]]
            self.decoders = [[]] + self.decoders
            for j, d in enumerate(dims):
                in_channels = out_channels
                out_channels = d
                self.encoders[-1] += [make_encoder(in_channels, out_channels)]
                setattr(self, "encoder_{}_{}".format(i, j), self.encoders[-1][-1])

                if final:
                    self.decoders[0] += [make_decoder(out_channels, self.output_channels, True)]
                    final = False
                else:
                    self.decoders[0] += [make_decoder(out_channels, in_channels, False)]
                setattr(self, "decoder_{}_{}".format(i, j), self.encoders[-1][-1])
            self.decoders[0] = self.decoders[0][::-1]


    def forward(self, input):

        x = input
        indices = []
        dims = [x.size()]
        for es in self.encoders:
            for e in es:
                x = F.relu(e(x))
            x, inds = F.max_pool2d(x, kernel_size = 2, stride = 2, return_indices = True)
            dims += [x.size()]
            indices += [inds]

        for ds, inds, dim in zip(self.decoders, indices[::-1], dims[-2::-1]):
            x = F.max_unpool2d(x, inds, kernel_size = 2, stride = 2, output_size = dim)
            for d in ds:
                x = d(F.relu(x))

        return x
