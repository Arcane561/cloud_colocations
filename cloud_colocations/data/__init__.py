import os
import glob
import numpy as np
import torch
import torch.utils
import torch.utils.data
import tempfile
import socket
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
    gpm_path = os.environ["GPM_COLOCATION_PATH"]
    #return glob.glob(os.path.join(gpm_path, "**", "**", "cloud_colocations.nc"))
    days = ["0" * (3 - len(str(i))) + str(i) for i in range(100)]
    return [os.path.join(gpm_path, str(2016), d, "cloud_colocations.nc") for d in days]

def copy_file(f):
    if socket.gethostname() == "titanite":
        return f
    else:
        src_path = "simonpf@titanite.rss.chalmers.se:" + f
        out_path = os.path.join(tempdir, "data.nc")
        subprocess.run(["scp", src_path, out_path])
        return out_path

class GpmColocations(torch.utils.data.Dataset):
    """GPM precipitation dataset"""

    def _load_file(self):
        files = list_files()
        f = np.random.choice(files)

        if not self.file_handle is None:
            try:
                self.file_handle.close()
            except:
                pass

        self.file_handle = Dataset(f, "r")


    def __init__(self, transform = None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.file_handle = None
        self._load_file()
        self.transform = transform

    def __len__(self):
        return self.file_handle.dimensions["colocation_index"].size

    def __getitem__(self, idx):
        x = np.transpose(self.file_handle["gmi"]["y"][idx, :, :, :], axes = (2, 0, 1))
        x = torch.tensor((x - y_mean.reshape(-1, 1, 1)) / y_std.reshape(-1, 1, 1)).float()
        y = torch.tensor(self.file_handle["dpr"]["rr"][idx, :, :]).float()
        y[:27, :] = -1
        y[-27:, :] = -1
        y[:, -27:] = -1
        y[:, :27] = -1

        return x, y
