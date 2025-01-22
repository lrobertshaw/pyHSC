import uproot
import matplotlib.pyplot as plt
import numpy as np
import math
from utils import *
import mplhep
from histogrammer import setup_bins, create_histogram, find_local_maxima_with_secondary_exclude_adjacent
plt.style.use(mplhep.style.CMS)

def getImages(coords, pts, maskSize):
    imgs = []
    for seed in coords:
        eta, phi = seed
        img = []
        for e in np.arange(eta-math.floor(maskSize/2), eta+math.ceil(maskSize/2), 1):
            for p in np.arange(phi-math.floor(maskSize/2), phi+math.ceil(maskSize/2), 1):
                pt = pts[e][p]
                img.append((int(e), int(p), pt))
        imgs.append(img)

    return np.array(imgs)


def plotImage(imgs, jetIdx):
    e, p, pt = zip(*imgs[jetIdx])
    
    plt.figure(figsize=(maskSize, maskSize), facecolor="white")
    plt.imshow(np.array(pt).reshape((maskSize, maskSize)).T, origin='lower', aspect='auto',
               extent=[e[0], e[-1]+1, p[0], p[-1]+1],
               cmap='viridis')
    plt.show()
    
    
def main():
    read = lambda path: uproot.open(path)["Events"]
    
    event = 0
    nBins = 72
    N = 9
    seedThresh = 5
    noisy = 0.005
    log = False
    
    print("Reading in data...")
    data = read("input_data/lightH.root")
    print("Getting PUPPI candidates...")
    puppi = getBranch(data, "PUPPICands_")

    # Define bins
    eta_bins, phi_bins, bin_width = setup_bins( nBins )
    # Histogram PUPPI candidates according to above bins
    pts, eta_edges, phi_edges = create_histogram( puppi.eta, puppi.phi, puppi.pt, eta_bins, phi_bins, event, noise_scale=noisy, log=log )
    # Find seeds, subseeds and sums
    coords, seccoords, sums = find_local_maxima_with_secondary_exclude_adjacent( pts, N, seedThresh, exclusionRegion=1, sort=True )
    # Get images of masks which find seeds
    imgs = getImages(coords, pts, N)
    
    return imgs

if __name__ == "__main__":
    main()