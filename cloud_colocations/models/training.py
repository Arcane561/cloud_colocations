import torch
import torch.cuda
from torch.utils.data import DataLoader
import os
import glob
import numpy as np

def masked_loss(y_pred, y, tau = 0.5):
    dy = torch.abs(y_pred.view(y.size()) - y)
    dy = torch.where(y >= 0.0, dy, torch.zeros(dy.size()))
    dy = torch.where(y_pred > y, (1 - tau) * dy, tau * dy)
    return dy.mean()

def quantile_loss(y_pred, y, quantiles):
    l = torch.tensor(0.0)
    for i, q in enumerate(quantiles):
        l += masked_loss(y_pred[:, i, :, :], y, q)
    return l

def load_most_recent(model, output_path):
    model_files = glob.glob(os.path.join(output_path, "model_*.pt"))
    if len(model_files) == 0:
        print("No model found in {}.".format(output_path))
        return None
    indices = np.array([int(os.path.basename(m).split("_")[1].split(".")[0]) for m in model_files])
    print(indices)
    i = np.argmax(indices)

    print("Loading most recent model {}.".format(model_files[i]))
    cuda = torch.cuda.is_available()
    if cuda:
        model.load_state_dict(torch.load(model_files[i]))
    else:
        model.load_state_dict(torch.load(model_files[i], map_location = "cpu"))

    model.eval()


def train_network(data_set,
                  model,
                  optimizer,
                  criterion,
                  output_path,
                  n_epochs ,
                  dataset_callback = None):


    if not os.path.exists(output_path):
        os.makedirs(output_path)

    cuda = torch.cuda.is_available()
    if cuda:
        print("Running on GPU.")
        model.cuda()
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        print("Running on CPU.")

    log_file = os.path.join(output_path, "training_log.txt")
    if os.path.isfile(log_file):
        log_file = open(log_file, mode = "r+", buffering = 1)
    else:
        log_file = open(log_file, mode = "w", buffering = 1)


    for i in range(n_epochs):

        dataset_callback(data_set)
        data_loader = DataLoader(data_set, batch_size = 128, shuffle = True)


        epoch_loss = 0.0

        for j, (x, y) in enumerate(data_loader):

            if cuda:
                x = x.cuda()
                y = y.cuda()

            y_pred = model(x)
            optimizer.zero_grad()
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.float()

            print("Epoch {0}, batch {1}: {2}".format(i, j, loss.float()))

        log_file.write("{0}\n".format(epoch_loss))
        model_files = glob.glob(os.path.join(output_path, "model_*.pt"))
        i_m = len(model_files)
        filename = os.path.join(output_path, "model_{0}.pt".format(i_m))
        torch.save(model.state_dict(), filename)
