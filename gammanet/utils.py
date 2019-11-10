
import numpy as np
import torch

from gammanet.simulate import sim_detector_image

def create_loader(n_images, nx, ny,
                  k_bar, contrast, corr_len,
                  sigma, epsilon_gain, epsilon_ped,
                  max_n_photons, batch_size):
    """
    Generate a batch of torch ready simulations
    """

    images  = []
    centers = []

    for i in range(n_images):
        img, cs = sim_detector_image(k_bar, contrast, corr_len,
                                     sigma, epsilon_gain, epsilon_ped,
                                     nx=nx, ny=ny, return_centers=True)

        images.append(torch.tensor(img))
        padded_cs = np.zeros((max_n_photons, 2))
        padded_cs[:cs.shape[0],:] = cs
        centers.append(torch.tensor(padded_cs))

    # turn them into torch datasets
    loader = torch.utils.data.DataLoader(
               torch.utils.data.TensorDataset( torch.stack(images),
                                               torch.stack(centers) ),
             batch_size=batch_size, shuffle=True)

    return loader




if __name__ == '__main__':
    nx, ny   = 16, 16    # image size
    n_train  = 16

    k_bar    = 0.1       # photons/px
    contrast = 0.9       # 
    corr_len = 0.2       # px
    sigma    = 2.0       # px
    epsilon_gain = 0.03  # adu
    epsilon_ped  = 0.05  # adu

    max_n_photons = 128
    batch_size    = 16

    params = (nx, ny, k_bar, contrast, corr_len,
              sigma, epsilon_gain, epsilon_ped, 
              max_n_photons, batch_size)

    l = create_loader(*(n_train,) + params)

    for i, (images, labels) in enumerate(l):
        print(images.shape, labels.shape)



