import uproot
import matplotlib.pyplot as plt
import numpy as np
import math
from utils import *
import mplhep
from histogrammer import setup_bins, create_histogram, find_local_maxima_with_secondary_exclude_adjacent
plt.style.use(mplhep.style.CMS)

def getImages(coords, pts, maskSize, rel=True, bw=1):
    imgs = []
    eta_lim, phi_lim = len(pts[0]), len(pts[1])    # 36 for nBins=36
    for seed in coords:
        eta, phi = seed
        img = []
        etas = np.arange(eta - math.floor(maskSize/2), eta + math.ceil(maskSize/2), 1)
        phis = np.arange(phi - math.floor(maskSize/2), phi + math.ceil(maskSize/2), 1)
        if rel: rel_e = 0
        for e in etas:
            if rel: rel_p = 0
            for p in phis:
                # Phi wrapping
                p_orig = p
                p = p - phi_lim if p >= phi_lim else p
                p = p + phi_lim if p < 0 else p
             
                if e >= eta_lim: pt = 0
                elif e < 0: pt = 0
                else: pt = pts[e][p]
                    
                if not rel:
                    img.append((int(e)*bw, int(p_orig)*bw, pt))
#                     img.append((int(e), int(p_orig), pt))
                if rel:
                    img.append((int(rel_e)*bw, int(rel_p)*bw, pt))
#                     img.append((int(rel_e), int(rel_p), pt))
                    rel_p += 1
            
            if rel: rel_e += 1
                
        imgs.append(img)

    return np.array(imgs)


def plotImage(img):
    e, p, pt = zip(*img)
    maskSize=int(np.sqrt(len(pt)))
    plt.figure(figsize=(maskSize, maskSize), facecolor="white")
    
    bw = p[-1] - p[-2]
    plt.imshow(np.array(pt).reshape((maskSize, maskSize)).T, origin='lower', aspect='auto',
               extent=[e[0], e[-1]+bw, p[0], p[-1]+bw],
               cmap='viridis')
    
    plt.xticks( np.round(np.linspace(e[0], e[-1]+bw, maskSize+1), 2) )
    plt.yticks( np.round(np.linspace(p[0], p[-1]+bw, maskSize+1), 2) )
    
    plt.xlabel("$\eta$"); plt.ylabel("$\phi$")
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