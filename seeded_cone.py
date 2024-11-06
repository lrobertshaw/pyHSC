import numpy as np
import awkward as ak
import vector
vector.register_awkward()

def seeded_cone(seeds, cands, r=0.4):
    jets = []
    subjets = []

    jets_constits = []
    subjets_constits = []
    
    for s, ss in seeds:
        sConstits = cands[s.deltaR(cands) < r]
        jets_constits.append(sConstits)
        jet = vector.obj(pt=0,eta=0,phi=0,mass=0)
        for vec in sConstits:
            jet+=vector.obj(pt=vec.pt, eta=vec.eta, phi=vec.phi, mass=vec.mass)
        jets.append(jet)
        
        ssConstits = cands[ss.deltaR(cands) < r]
        subjets_constits.append(ssConstits)
        subjet = vector.obj(pt=0,eta=0,phi=0,mass=0)
        for vec in ssConstits:
            subjet+=vector.obj(pt=vec.pt, eta=vec.eta, phi=vec.phi, mass=vec.mass)
        subjets.append(subjet)
    
    
    arrMaker = lambda js: ak.with_name(ak.zip({"pt": [jet.pt for jet in js], "eta": [jet.eta for jet in js], "phi": [jet.phi for jet in js], "mass": [jet.mass for jet in js]}), "Momentum4D")
    
    jets = arrMaker(jets)
    subjets = arrMaker(subjets)
    jets_constits = arrMaker(jets_constits)
    subjets_constits = arrMaker(subjets_constits)
    
    return (jets, jets_constits), (subjets, subjets_constits)