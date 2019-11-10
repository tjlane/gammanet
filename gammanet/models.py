
import torch
from torch import nn

import numpy as np


class ConvNet(nn.Module):

    def __init__(self, image_dims=(32,32), max_photons=128):
        super(ConvNet, self).__init__()

        final_shape = (image_dims[0]-24, image_dims[1]-24) # dbl chk
        final_size  = final_shape[0] * final_shape[1]

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
        self.fc1 = nn.Linear(final_size * 64, max_photons * 3 * 3)
        self.fc2 = nn.Linear(max_photons * 3 * 3, max_photons*3)

        return


    def forward(self, x):
        xp = x.unsqueeze(1)
        out = self.layer1(xp)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        out = out.reshape(out.size(0), -1, 3)  # is_photon, x, y
        out[:,:,0] = torch.sigmoid(out[:,:,0]) # sigmoid classifier
        return out



def photon_loss(labels, preds, confidence=0.8, weight=10.0):
    """
    preds is a (data_pts x n_centers x 3) tensor
      last index:
        dim 0 : [0,1)    is it a photon?
        dim 1 : [0,nx]   x-location
        dim 2 : [0,ny]   y-location

    labels only has dim 1 & 2, and may be a
    different length!
    """

    pos_idx = (preds[:,:,0] > confidence)
    pred_xy = preds[:,:,1:] 
    what = torch.sum(preds[:,:,0])

    # any -1 in simulation indicates that was not a photon
    label_idx = (labels[:,:,0] > 0)    

    # assign photons to their closest label
    #   >> all-to-other distance

    n  = pred_xy.size(1)
    m  = labels.size(1)
    d1 = pred_xy.size(0)
    d2 = pred_xy.size(2)

    x = pred_xy.unsqueeze(1).expand(d1, n, m, d2)
    y = labels.unsqueeze(2).expand(d1, n, m, d2)

    dist = torch.pow(x - y, 2).sum(3)

    # dimensions: datapoints x predictions x labels
    assert dist.shape == (d1, n, m)

    # this is an array of the same size indicating where
    # BOTH the label and prediction are positive
    pos = pos_idx.unsqueeze(2).expand(d1, n, m) * \
          label_idx.unsqueeze(1).expand(d1, n, m)

    # penalize mis-matches
    rw_dist = dist + (1-pos.float())*weight # penalize mismatches

    # here we take the smallest distance for each label
    # if a label is mis-assigned, it has the large penalty
    val, idx = rw_dist.min(2)
    distance_error = torch.sum(val)

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


