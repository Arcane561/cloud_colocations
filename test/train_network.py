import os
os.environ["GPM_COLOCATION_PATH"] = "/home/simonpf/Dendrite/UserAreas/Simon/cloud_colocations/gpm"

from IPython import get_ipython
ip = get_ipython()
if not ip is None:
    ip.magic("%load_ext autoreload")
    ip.magic("%autoreload 2")

import torch
import torch.optim
from cloud_colocations.models.convnet import ConvNet
from cloud_colocations.data import GpmColocations
from cloud_colocations.models.training import masked_loss, train_network, load_most_recent

data = GpmColocations()
optimizer = torch.optim.Adam(model.parameters())
criterion = masked_loss
output_path = "conv_128_6"

model = ConvNet(13, 1, arch = [128] * 6)
load_most_recent(model, output_path)

train_network(data, model, optimizer, criterion, output_path, 10, lambda ds: ds._load_file())
