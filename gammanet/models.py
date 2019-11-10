
import torch
from torch import nn


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



def photon_loss(ref, pred, class_weight=1.0):
    """
    data are an N x 3 tensor
      dim 0 : [0,1]    is it a photon?
      dim 1 : [0,nx]   x-location
      dim 2 : [0,ny]   y-location
    """

    classification_loss = nn.functional.binary_cross_entropy(ref[:,0], pred[:,0], 
                                                             reduction='mean')
    position_loss       = nn.functional.mse_loss(ref[:,1:], pred[:,1:],
                                                 reduction='mean')

    return classification_loss * class_weight + position_loss



