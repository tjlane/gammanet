

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.special import gamma, gammaln
from scipy.special import psi as digamma


def generate_random_centers(n_photons, nx, ny):
    cx1 = np.random.uniform(0, nx, n_photons)
    cy1 = np.random.uniform(0, ny, n_photons)
    return np.vstack([cx1, cy1]).T


def gaussian_2d(center, width, amplitude=1, nx=10, ny=10):
    x = np.arange(0., nx, 1.)
    y = np.arange(0., ny, 1.)
    xx, yy = np.meshgrid(x,y, sparse=True)
    gx = (xx-center[0])**2
    gy = (yy-center[1])**2
    g = np.exp(-4*np.log(2)*(gx+gy)/width**2)
    g /= np.max(g.flatten()) # or sum
    return g*amplitude


def charge_sharing(pn_list, corr_len, sigma, nx, ny, return_centers=False):
    """
    pn_list : number of photons to "throw"
        each item in the list is the number to sample for [one, two, ...] photons/region
    corr_len : size of gaussian perturbation for multi-photon clusters
    sigma : size of charge sharing region
    nx / ny : size of the final image
    """

    img = np.zeros([nx, ny])

    if return_centers:
        all_centers = []

    # loop over one, two, three, photon speckles    
    for pn,n in enumerate(pn_list):
        centers = generate_random_centers(n, nx, ny)

        # for each speckle
        for i in range(n):

            # for each photon in that speckle
            for p in range(pn+1):
                center = centers[i] + np.random.normal(0.0, corr_len, 2)
                if return_centers:
                    all_centers.append(center)
                img += gaussian_2d(center, sigma, amplitude=1.0, nx=nx, ny=ny) 
    
    if return_centers:
        if len(all_centers) > 0:
            return img, np.vstack(all_centers)
        else:
            return img, np.zeros((0,2))
    else:
        return img


def negative_binomial_pmf(k_range, k_bar, contrast):
    """
    Evaluate the negative binomial probablility mass function.
    
    Parameters
    ----------
    k_range : ndarray, int
        The integer values in the domain of the PMF to evaluate.
        
    k_bar : float
        The mean count density. Greater than zero.
        
    contrast : float
        The contrast parameter, in [0.0, 1.0).
        
    Returns
    -------
    pmf : np.ndarray, float
        The normalized pmf for each point in `k_range`.
    """

    if type(k_range) == int:
        k_range = np.arange(k_range)
    elif type(k_range) == np.ndarray:
        pass
    else:
        raise ValueError('invalid type for k_range: %s' % type(k_range))

    M = 1.0 / contrast
    norm = np.exp(gammaln(k_range + M) - gammaln(M) - gammaln(k_range+1))
    f1 = np.power(1.0 + M/k_bar, -k_range)
    f2 = np.power(1.0 + k_bar/M, -M)
    
    return norm * f1 * f2


def negative_binomial_samples(k_bar, contrast, size=1):
    M = 1.0 / contrast
    p = 1.0 / (k_bar/M + 1.0)
    samples = np.random.negative_binomial(M, p, size=size)
    return samples


def sim_detector_image(k_bar, contrast, corr_len, sigma, epsilon_gain, epsilon_ped, 
                       nx=528, ny=528, k_range=10, return_centers=False):

    p_bar = int(nx * ny * k_bar) # expected photons
    pn_list = np.bincount(negative_binomial_samples(k_bar, contrast, size=p_bar))

    if return_centers:
        img, cs = charge_sharing(pn_list, corr_len, sigma, nx, ny,
                                 return_centers=return_centers)
    else:
        img = charge_sharing(pn_list, corr_len, sigma, nx, ny, 
                             return_centers=return_centers)

    peds  = np.random.normal(0.0, epsilon_ped,  [nx,ny])
    gain  = np.random.normal(1.0, epsilon_gain, [nx,ny])
    img_n = img*gain + peds

    if return_centers:
        return img_n, cs
    else:
        return img_n


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    k_bar    = 0.01      # photons/px
    contrast = 0.9       # 
    corr_len = 0.2       # px
    sigma    = 2.0       # px
    epsilon_gain = 0.03  # adu
    epsilon_ped  = 0.05  # adu

    img, cs = sim_detector_image(k_bar, contrast, corr_len, 
                                 sigma, epsilon_gain, epsilon_ped,
                                 nx=32, ny=32, return_centers=True)

    print(cs)

    plt.figure()
    plt.imshow(img)
    plt.show()


