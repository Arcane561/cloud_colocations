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


y_mean = np.array([212.89453, 131.55109, 222.22789, 160.21983, 227.815,
                   227.78896, 206.75392, 255.43271, 231.61769, 276.89465,
                   263.819  , 255.67427, 281.76434])
y_std = np.array([69.32374572753906, 88.8903579711914, 37.72039031982422,
                  60.515342712402344, 28.749357223510742, 24.532129287719727,
                  50.60862350463867, 23.05337905883789, 31.068477630615234,
                   24.02410888671875, 23.94635581970215, 9.11214542388916,
                  22.18482208251953])

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

def copy_file(f, syncd = False):
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
        y = np.zeros((1, m, n, 13))
        y[0, :, :, :9] = g['Tc'][:]

        g = file_handle['S2']
        y[0, :, :, 9:] = g['Tc'][:]
        x = np.transpose(y, axes = (0, 3, 1, 2))
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
            if self._next_file is tuple:
                p = self._next_file[0].wait()
                f = self._next_file[1]
            else:
                f = self._next_file
            self.file_handle = Dataset(f, "r")
        else:
            f = copy_file(np.random.choice(files))
            self.file_handle = Dataset(f, "r")

        self._next_file = copy_file(np.random.choice(files), syncd = False)
        if self._next_file is tuple:
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
