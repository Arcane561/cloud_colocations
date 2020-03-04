import os
import glob
import numpy as np
import torch
import torch.utils
import torch.utils.data
import tempfile
import socket
from tempfile import NamedTemporaryFile
import subprocess
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


y_mean = np.array([227.60539246, 227.60655212, 227.59770203, 227.59559631,
                   227.60879517, 227.64025879, 227.67428589, 227.70542908,
                   227.72660828, 227.7388916 , 227.7660675 , 227.79171753,
                   227.81913757])
y_std = np.array([55.27218628, 55.25743103, 55.24773026, 55.23535538, 55.22212219,
                  55.22195435, 55.23134232, 55.23345947, 55.22345734, 55.20721436,
                  55.19950485, 55.18951416, 55.17945862])

tempdir = tempfile.mkdtemp()

def list_files():
    try:
        assert(socket.gethostname() == "titanite")
        gpm_path = os.environ["GPM_COLOCATION_PATH"]
        return glob.glob(os.path.join(gpm_path, "**", "**", "cloud_colocations.nc"))
    except:
        gpm_path = "/home/simonpf/Dendrite/UserAreas/Simon/cloud_colocations/gpm"
        days = ["0" * (3 - len(str(i))) + str(i) for i in range(1, 365)]
        return [os.path.join(gpm_path, str(2016), d, "cloud_colocations.nc") for d in days]

def copy_file(f, syncd = True):
    if socket.gethostname() == "titanite":
        return f
    else:
        src_path = "simonpf@titanite.rss.chalmers.se:" + f
        f = NamedTemporaryFile(delete = False, dir = ".")
        out_path = f.name
        p = subprocess.Popen(["scp", src_path, out_path])
        if syncd:
            p.wait()
            return out_path
        else:
            return p, out_path

class GpmColocations(torch.utils.data.Dataset):
    """GPM precipitation dataset"""

    def preprocess(file_handle):
        g = file_handle["S1"]
        m, n = g["Tc"].shape[:2]
        mm = 16 * ((m // 16) + 1)
        ml = (mm - m) // 2 
        mr = ml + m
        nn = 16 * ((n // 16) + 1)
        nl = (nn - n) // 2
        nr = nl + n
        y = np.zeros((1, mm, nn, 13))

        y[0, ml:mr, nl:nr, :9] = g['Tc'][:]

        g = file_handle['S2']
        y[0, ml:mr, nl:nr, 9:] = g['Tc'][:]
        x = np.transpose(y, axes = (0, 3, 1, 2))
        x = np.maximum(x, 0.0)
        x = np.minimum(x, 500.0)
        x = torch.tensor((x - y_mean.reshape(-1, 1, 1)) / y_std.reshape(-1, 1, 1)).float()
        return x

    def _load_file(self):
        files = list_files()
        if not self.file_handle is None:
            fp = self.file_handle.filepath()
            self.file_handle.close()
            if self._temp_file:
                os.unlink(fp)

        if not self._next_file is None:
            if type(self._next_file) is tuple:
                p = self._next_file[0].wait()
                f = self._next_file[1]
            else:
                f = self._next_file
            self.file_handle = Dataset(f, "r")
        else:
            f = copy_file(np.random.choice(files))
            self.file_handle = Dataset(f, "r")

        self._next_file = copy_file(np.random.choice(files), syncd = False)
        if type(self._next_file) is tuple:
            self._temp_file = True
        else:
            self._temp_file = False

    def __init__(self, transform = None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.file_handle = None
        self._next_file = None
        self._load_file()
        self.transform = transform

    def __len__(self):
        return self.file_handle.dimensions["colocation_index"].size

    def __getitem__(self, idx):
        try:
            x = np.transpose(self.file_handle["gmi"]["y"][idx, :, :, :], axes = (2, 0, 1))
            x = np.maximum(x, 0.0)
            x = (x - y_mean.reshape(-1, 1, 1)) / y_std.reshape(-1, 1, 1)
            y = self.file_handle["dpr"]["rr"][idx, :, :]
            y[:27, :] = -1
            y[-27:, :] = -1
            y[:, -27:] = -1
            y[:, :27] = -1

            p = np.random.uniform()
            if p < 0.5:
                x = np.array(x[:, ::-1, :])
                y = np.array(y[::-1, :])
            p = np.random.uniform()
            if p < 0.5:
                x = np.array(x[:, :, ::-1])
                y = np.array(y[:, ::-1])
        except Exception as e:
            raise e

        return torch.tensor(x[:, 2:-1, 2:-1]).float(), torch.tensor(y[2:-1, 2:-1]).float()


class GpmData:
    def __init__(self, filename, n = -1):
        self.file = Dataset(filename)

        x = self.file["x"][:n, :, :, :]

        valid = []
        x_normed = np.zeros(x.shape, dtype=np.float32)

        for i in range(x.shape[-1]):

            valid = np.logical_and(x[:, :, :, i] > 0.0,
                                   x[:, :, :, i] < 500.0)
            x_min = x[valid, i].min()
            x_max = x[valid, i].max()
            x_normed[:, :, :, i] = -0.9 + 1.9 * (x[:, :, :, i] - x_min) / (x_max - x_min)
            x_normed[~valid, i] = -1.0
            print(x_min, x[:, :, :, i].min(), x_max, x[:, :, :, i].max(), x_normed[:, :, i].min())

        self.x = np.transpose(x_normed, (0, 3, 1, 2))

        self.y = self.file["y"][:n, :, :]
        valid = self.y >= 0.0
        self.y[~valid] = -1.0

        n = self.y.shape[1] // 2
        self.y[:, :, :n - 10] = -1
        self.y[:, :, (n + 10) :] = -1

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        x = torch.tensor(self.x[index, :, 3:-2, 3:-2])
        y = torch.tensor(self.y[index, 3:-2, 3:-2])
        y[:10, :] = -1
        y[-10:, :] = -1
        return (x, y)

    def plot_colocation(self):

        ind = np.random.randint(self.x.shape[0])

        f = plt.figure(figsize = (16, 4))
        gs = GridSpec(1, 4)

        for i, j in enumerate([5, 9, 12]):
            ax = plt.subplot(gs[i])
            ax.pcolormesh(self.x[ind, :, :, j])

        ax = plt.subplot(gs[-1])
        ax.pcolormesh(self.y[ind, :, :])
        return f





