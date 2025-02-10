import uproot
import matplotlib.pyplot as plt
import numpy as np
import awkward as ak
import vector
from utils import *
from scipy.ndimage import maximum_filter
vector.register_awkward()
import mplhep
plt.style.use(mplhep.style.CMS)


def setup_bins(nBins):
    nBins+=1
    eta_bins = np.linspace(-3, 3, nBins)
    phi_bins = np.linspace(-np.pi, np.pi, nBins)
    bin_width = abs(eta_bins[-1] - eta_bins[-2])
    return eta_bins, phi_bins, bin_width


def create_histogram(eta, phi, pt, eta_bins, phi_bins, ev=0, noise_scale=0.05, log=False):
    # Ensure inputs are flattened and converted to NumPy arrays
    eta = ak.to_numpy(eta[ev])
    phi = ak.to_numpy(phi[ev])
    pt = ak.to_numpy(pt[ev])

    # Create the histogram without noise
    if log: pts, eta_edges, phi_edges = np.histogram2d(eta, phi, bins=[eta_bins, phi_bins], weights=np.log(pt))
    else: pts, eta_edges, phi_edges = np.histogram2d(eta, phi, bins=[eta_bins, phi_bins], weights=pt)
    
    # Add small random noise to each bin to break degeneracy and avoid mutual exclusion
    noise = np.random.uniform(-noise_scale, noise_scale, size=pts.shape)
    pts += noise
    
    return pts, eta_edges, phi_edges


def get_decay_quarks(genParts, bosonPDG, ev):
    fst, lst = getQuarks(genParts)
    qs, qbs, vs, qls, qlbs = boostedVs(quarks=fst, lst_quarks=lst, genParts=genParts, bosonPDG=bosonPDG, r=3.0)
    return qs[ev], qbs[ev], vs[ev], qls[ev], qlbs[ev]


def find_local_maxima_with_secondary_exclude_adjacent(pts, N, minPt, exclusionRegion=3, sort=False):
    # Step 1: Identify primary local maxima
    pts_max = maximum_filter(pts, size=N, mode=('constant', 'wrap'))
    local_maxima = (pts == pts_max) & (pts > minPt)
    maxima_coords = np.argwhere(local_maxima)
    
    # Step 2: For each primary maximum, identify the second-highest pT in its region, excluding adjacent bins
    secondary_coords = []
    sums_in_filter = []
    
    for (i, j) in maxima_coords:
        # Define the N x N region around the current local maximum
        eta_min, eta_max = max(0, i - N // 2), min(pts.shape[0], i + N // 2 + 1)
        phi_min, phi_max = max(0, j - N // 2), min(pts.shape[1], j + N // 2 + 1)
        
        # Extract pT values within this region
        region = pts[eta_min:eta_max, phi_min:phi_max].copy()
        
        # Calculate the sum of pT values within the N x N region
        sum_in_region = np.sum(region)
        sums_in_filter.append(sum_in_region)
        
        # Define exclusion region centered at the local maximum
        excl_eta_min, excl_eta_max = max(0, i - (exclusionRegion//2)) - eta_min, min(pts.shape[0], i + ((exclusionRegion//2) + 1)) - eta_min
        excl_phi_min, excl_phi_max = max(0, j - (exclusionRegion//2)) - phi_min, min(pts.shape[1], j + ((exclusionRegion//2) + 1)) - phi_min
        
        # Set values in the 5x5 adjacent region to a very low number so they are not chosen
        region[excl_eta_min:excl_eta_max, excl_phi_min:excl_phi_max] = -np.inf
        
        # Find the second-highest pT in the modified region
        second_highest_pt = np.max(region)
        if second_highest_pt > -np.inf:
            # Get the coordinates of this second-highest point within the full pts array
            second_highest_idx = np.unravel_index(
                np.argmax(region == second_highest_pt), region.shape
            )
            secondary_coords.append([eta_min + second_highest_idx[0], phi_min + second_highest_idx[1]])
    
    sums = np.array(sums_in_filter)
    secondary_coords = np.array(secondary_coords)
    if sort:
        sorting = np.argsort(sums)[::-1]
        sums, maxima_coords, secondary_coords = sums[sorting], maxima_coords[sorting], secondary_coords[sorting]
        
    return maxima_coords, secondary_coords, sums


def plot_histogram(pts, eta_edges, phi_edges, maxima_coords, eta_bins, phi_bins, ev, nBins, N, *qs):
    plt.figure(figsize=(12, 12), facecolor="white")
    plt.imshow(pts.T, origin='lower', aspect='auto',
               extent=[eta_edges[0], eta_edges[-1], phi_edges[0], phi_edges[-1]],
               cmap='viridis')

    for idx, q in enumerate(qs):
        if idx == 0: plt.scatter(q.eta, q.phi, s=500, color="white", marker="x", label="H decay quarks")
        plt.scatter(q.eta, q.phi, s=500, color="white", marker="x")
#         plt.scatter(qb.eta, q.phi, s=500, color="white", marker="x")
    
    bin_width_eta = eta_edges[1] - eta_edges[0]
    bin_width_phi = phi_edges[1] - phi_edges[0]
    for (i, j) in maxima_coords:
        eta_center = eta_edges[i] + bin_width_eta / 2
        phi_center = phi_edges[j] + bin_width_phi / 2
        plt.gca().add_patch(plt.Rectangle(
            (eta_center - (N/2) * bin_width_eta, phi_center - (N/2) * bin_width_phi),
            N * bin_width_eta,
            N * bin_width_phi,
            fill=False,
            edgecolor='red',
            linewidth=1.5,
            linestyle='--'
        ))

    plt.xlabel('$\\eta$', fontsize=20)
    plt.ylabel('$\\phi$', fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlim([eta_bins[0], eta_bins[-1]])
    plt.ylim([phi_bins[0], phi_bins[-1]])
    plt.title(f'Event = {ev}, nBins = {nBins}, mask size = {N}x{N}', fontsize=20)
    leg = plt.legend(fontsize=12)
    for text in leg.get_texts(): text.set_color("white")
    plt.show()

    
# def plot_histogram_no_quarks(pts, eta_edges, phi_edges, maxima_coords, eta_bins, phi_bins, nBins, N):
#     plt.figure(figsize=(12, 12), facecolor="white")
#     plt.imshow(pts.T, origin='lower', aspect='auto',
#                extent=[eta_edges[0], eta_edges[-1], phi_edges[0], phi_edges[-1]],
#                cmap='viridis')

#     bin_width_eta = eta_edges[1] - eta_edges[0]
#     bin_width_phi = phi_edges[1] - phi_edges[0]
#     for (i, j) in maxima_coords:
#         eta_center = eta_edges[i] + bin_width_eta / 2
#         phi_center = phi_edges[j] + bin_width_phi / 2
#         plt.gca().add_patch(plt.Rectangle(
#             (eta_center - (N/2) * bin_width_eta, phi_center - (N/2) * bin_width_phi),
#             N * bin_width_eta,
#             N * bin_width_phi,
#             fill=False,
#             edgecolor='red',
#             linewidth=1.5,
#             linestyle='--'
#         ))

#     plt.xlabel('$\\eta$', fontsize=20)
#     plt.ylabel('$\\phi$', fontsize=20)
#     plt.xticks(fontsize=16)
#     plt.yticks(fontsize=16)
#     plt.xlim([eta_bins[0], eta_bins[-1]])
#     plt.ylim([phi_bins[0], phi_bins[-1]])
#     plt.title(f'Event = {ev}, nBins = {nBins}, mask size = {N}x{N}', fontsize=20)
#     leg = plt.legend(fontsize=12)
#     for text in leg.get_texts(): text.set_color("white")
#     plt.show()
    

def create_output_data(pts, maxima_coords, eta_bins, phi_bins, bin_width):
    d = {
        "pt": [pts[i[0], i[1]] for i in maxima_coords],
        "eta": [eta_bins[i[0]] + bin_width for i in maxima_coords],
        "phi": [phi_bins[i[1]] + bin_width for i in maxima_coords]
    }
    return ak.with_name(ak.zip(d), "Momentum4D")


def applySubseedThresh(seeds, subseeds, sums, thresh=3):
    mask = subseeds.pt > thresh
    return seeds[mask], subseeds[mask], sums[mask]


def histogrammer(data, ev, nBins, N, genParts, bosonPDG, plot=False, minPt=1, exclusionRegion=1, subseedThresh=1):
    eta, phi, pt = data.eta, data.phi, data.pt

    # Binning and histogram creation
    eta_bins, phi_bins, bin_width = setup_bins(nBins)
    pts, eta_edges, phi_edges = create_histogram(eta, phi, pt, eta_bins, phi_bins)
    
    # Find local maxima
    maxima_coords, secondary, sums = find_local_maxima_with_secondary_exclude_adjacent(pts, N, minPt, exclusionRegion)

    # Optional plotting
    if plot == True:
        q, qb, v, ql, qlb = get_decay_quarks(genParts, bosonPDG, ev)
        plot_histogram(pts, eta_edges, phi_edges, maxima_coords, eta_bins, phi_bins, ev, nBins, N, ql, qlb)
    
    # Create output data
    seeds = create_output_data(pts, maxima_coords, eta_bins, phi_bins, bin_width)
    subseeds = create_output_data(pts, secondary, eta_bins, phi_bins, bin_width)
    sums = np.array(sums)
    
    seeds, subseeds, sums = applySubseedThresh(seeds, subseeds, sums, thresh=subseedThresh)
    
    return seeds, subseeds, sums


def main():
    read = lambda path: uproot.open(path)["Events"]
    
    event = 4
    nBins = 36
    N = 9
    bosonPDG = 25
    plot = False
    seedThresh = 5
    subseedThresh = 3
    exclusionRegion = 3
    
    data = read("lightH.root")
    puppi = getBranch(data, "PUPPICands_")
    gen = getBranch(data, "GenParticles_")
    
    seeds, subseeds, histosums = histogrammer(puppi, event, nBins, N, gen, bosonPDG, plot, seedThresh, exclusionRegion, subseedThresh)
    
    return seeds, subseeds, histosums

if __name__ == "__main__":
    main()