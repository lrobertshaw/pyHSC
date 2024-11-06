import sys
import numpy as np
import uproot
import awkward as ak
from matplotlib import pyplot as plt
from utils import getBranch

from histogrammer import histogrammer
from seeded_cone import seeded_cone
import substructure_functions as sf
read = lambda path: uproot.open(path)["Events"]

args = sys.argv[1:]
fname, nEvents, path, nBins, N, bosonPDG, seedThresh, exclusionRegion, subseedThresh = tuple(args)

fname, nEvents, path, nBins, N, bosonPDG, seedThresh, exclusionRegion, subseedThresh = str(fname), int(nEvents), str(path), int(nBins), int(N), int(bosonPDG), int(seedThresh), int(exclusionRegion), int(subseedThresh)

data = read(path)
puppi = getBranch(data, "PUPPICands_")
gen = None
if nEvents == -1: nEvents = len(puppi)

js = []
for ev in range(nEvents):
    cands = puppi[ev]

    """ Get seeds """
    seeds, subseeds, sums = histogrammer(cands, ev, nBins, N, gen, bosonPDG, False, seedThresh, exclusionRegion, subseedThresh)
    if len(seeds) > 0:
        """ Find basic seed level info"""
        seedsep = seeds.deltaR(subseeds)
        seedpt, subseedpt = seeds.pt, subseeds.pt

        """ Run SC4 corresponding to seed and subseed """
        jets, subjets = seeded_cone(zip(seeds, subseeds), cands, r=0.4)

        """ Run softdrop """
        sd0 = sf.softdrop(jets, subjets, beta=0)
        sd1 = sf.softdrop(jets, subjets, beta=1)

        """ Run N-subjetiness """
        sc8, _ = seeded_cone(zip(seeds, subseeds), cands, r=0.8)
        t2t1, t2 = sf.nsubjetiness(jets, subjets, sums, sc8)

        """ Run reconstructedness """
        rcsn = sf.reconstructedness(jets, subjets, sc8)

        """ Form summed jet """
        summedJets = jets[0] + subjets[0]    # combine jet and subjet
        summedJets = ak.with_field(summedJets, sd0, where="sd0")
        summedJets = ak.with_field(summedJets, sd1, where="sd1")
        summedJets = ak.with_field(summedJets, t2t1, where="tau21")
        summedJets = ak.with_field(summedJets, t2, where="tau2")
        summedJets = ak.with_field(summedJets, rcsn, where="rcsn")
        summedJets = ak.with_field(summedJets, seedsep, where="seedsep")
        summedJets = ak.with_field(summedJets, seedpt, where="seedpt")
        summedJets = ak.with_field(summedJets, subseedpt, where="subseedpt")
        
        summedJets = summedJets[ak.argsort(summedJets.rho, ascending=False)]
        js.append(summedJets)

    print(f"Processed {ev+1} events.")

arr = ak.Array(js)
ak.to_parquet(arr, f"output_data/{fname}.parquet")