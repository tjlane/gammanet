
import torch
from torch import nn

import numpy as np


class ConvNet(nn.Module):

    def __init__(self, max_photons=128):
        super(ConvNet, self).__init__()

        dropout_p = 0.5

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.drop_out = nn.Dropout(dropout_p)
        self.fc1 = nn.Linear(7 * 7 * 64, max_photons * 3 * 3)
        self.fc2 = nn.Linear(max_photons * 3 * 3, max_photons*3)

        return


    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        out = out.reshape(-1, 3) # is_photon, x, y
        out[:,0] = torch.sigmoid(out[:,0])
        return out



def photon_loss(labels, preds, confidence=0.9, class_weight=1.0):
    """
    preds is a (batches x N x 3) tensor
      last index:
        dim 0 : [0,1)    is it a photon?
        dim 1 : [0,nx]   x-location
        dim 2 : [0,ny]   y-location

    labels only has dim 1 & 2, and may be a
    different length!
    """

    # TODO : think about the right functional form for the loss
    # right now it's just temporary

    pos_idx   = (preds[:,:,0] > confidence)
    pos_preds = preds[:,:,1:] # could maybe slice out here

    # assign photons to their closest label
    #   >> all-to-other distance

    n  = pos_preds.size(1)
    m  = labels.size(1)
    d1 = pos_preds.size(0)
    d2 = pos_preds.size(2)

    x = pos_preds.unsqueeze(1).expand(d1, n, m, d2)
    y = labels.unsqueeze(2).expand(d1, n, m, d2)

    dist = torch.pow(x - y, 2).sum(3) # CHANGE LAST

    assert dist.shape == (d1, n, m)

    # now take min(dist) to match
    val, idx = dist.min(2)

    # ^^^

    distance_error = torch.sum(val[pos_idx])

    return distance_error




if __name__ == '__main__':

    from gammanet.utils import create_loader

    nx, ny   = 16, 16    # image size
    n_train  = 16

    k_bar    = 0.1       # photons/px
    contrast = 0.9       # 
    corr_len = 0.2       # px
    sigma    = 2.0       # px
    epsilon_gain = 0.03  # adu
    epsilon_ped  = 0.05  # adu

    max_n_photons = 128
    batch_size    = 4

    params = (nx, ny, k_bar, contrast, corr_len,
              sigma, epsilon_gain, epsilon_ped,
              max_n_photons, batch_size)

    l = create_loader(*(n_train,) + params)

    for i, (_, labels) in enumerate(l):

        pi = np.random.permutation(labels.shape[0])
        test_pred = labels[pi,:]
        o = torch.ones(test_pred.shape[:2]+(1,), dtype=float)
        test_pred = torch.cat([ o, test_pred ], 2)
        print(photon_loss(labels, test_pred))


