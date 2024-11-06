import awkward as ak
import numpy as np

def softdrop(jets, subjets, beta=0):
    jetss, _ = jets
    subjetss, _ = subjets
    
    sd = np.array([
        min(jet.pt, subjet.pt) / ((jet.pt+subjet.pt)*(jet.deltaR(subjet)**beta)) for jet, subjet in zip(jetss, subjetss)
    ])
    
    return sd    # return 1-sd to get 1 as 0 as most likely


def reconstructedness(jets, subjets, fullJet):
    jetss, _ = jets
    subjetss, _ = subjets
    fullJets = fullJet[0]

    msd = []
    for jetIdx in range(len(jetss)):
        jet, subjet, fulljet = jetss[jetIdx], subjetss[jetIdx], fullJets[jetIdx]
        msd.append( (jet.pt+subjet.pt) / fulljet.pt )    # >1 corresponds to one prong being reconstructed twice, 1 corresponds to full reconstruction of one or both prongs, <1 corresponds to 1 prong with flat pt dist
    
    return np.array(msd)


def nsubjetiness(jets, subjets, sums, fullJet):
    jets, jets_constits = jets
    subjets, subjets_constits = subjets
    
    fullJets, fullJets_constits = fullJet[0], fullJet[1]
    
    tsj = lambda jet, constits: ak.sum(constits.pt * jet.deltaR(constits))
    
    score = []
    t2s = []
    for jetIdx in range(len(jets)):
        jet, jet_constits = jets[jetIdx], jets_constits[jetIdx]
        subjet, subjet_constits = subjets[jetIdx], subjets_constits[jetIdx]
        t2 = tsj(jet, jet_constits) + tsj(subjet, subjet_constits)
        
        fullJet, fullJet_constits = fullJets[jetIdx], fullJets_constits[jetIdx]
        t1 = tsj(fullJet, fullJet_constits)
                
        score.append(t2 / t1)
        t2s.append(t2 / sums[jetIdx])
        
    return np.array(score), np.array(t2s)