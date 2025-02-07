import uproot
import matplotlib.pyplot as plt
import numpy as np
import awkward as ak
import vector
vector.register_awkward()
import mplhep

plt.style.use(mplhep.style.CMS)
read = lambda path: uproot.open(path)["Events"]

def getBranch(data, l1jet):
    
    # Get list of branches relating to the given jet
    print("Getting list of interesting branches")
    interestingBranches = [branch for branch in data.keys() if l1jet in branch and not f"n{l1jet}" in branch]# and not f"{l1jet}_dau0" in branch] 
    
    # Get only branches corresponding to the desired l1jet
    print("Querying the array with interesting branches")
    l1jetData = data.arrays(interestingBranches)
    
    
    # Get a dictionary relating default branch names to new branch names (ie without leading ak8PuppiJets_...)
    print("Splitting branch names on _ to get renamed fields")
    renamedFields = {field : field.split('_', maxsplit=1)[-1] for field in interestingBranches}

    # Create a new awkward array with the desired l1jet branches and the new branch names
    print("Returning an ak array of the relevant data with the renamed fields")
    arr = ak.Array({renamedFields[field]: l1jetData[field] for field in l1jetData.fields})
    
    
    array_dict = {key: arr[key] for key in arr.fields}
    quarks4mom = ak.zip(array_dict)
    quarks4mom = ak.with_name(quarks4mom, "Momentum4D")
    
    
    return quarks4mom


def plot(parts, l1jet=None, event=0, r=0.8, *specialParts):
    
    fig, ax = plt.subplots(figsize=(12, 12))
    col = "black"
    alpha=1
    
    # apply masks:
    try: stableParts = parts[isStable]
    except: stableParts = parts

    if l1jet is not None:
        try: jets = l1jet[l1jet.genpt > 30]
        except: jets = l1jet
    
    """ PLOT PARTICLES """
    print(f"{len(stableParts[event].pt)} stable gen particles in event {event}")
    parts_eta, parts_phi, parts_pt = stableParts[event].eta, stableParts[event].phi, stableParts[event].pt
    ax.scatter( parts_eta, parts_phi, c=20*np.log(1+parts_pt), cmap="Reds")    # Plot PUPPI candidates
    #ax.scatter( parts_eta, parts_phi, s=20*np.log(1+parts_pt), c="red")

    """ PLOT JETS """
    if l1jet is not None:
        print(f"{len(jets[event].pt)} jets in event {event}")
        eta, phi, pt, mass = jets[event].eta, jets[event].phi, jets[event].pt, jets[event].mass    # 
        drawCircle = lambda eta, phi, r=r: ax.add_patch(plt.Circle((eta, phi), r, edgecolor=col, facecolor='none', linewidth=1, linestyle="--" ))    # DEFINE LAMBDA FN FOR PLOTTING CIRCLE
        for eta, phi, pt, mass in zip(eta, phi, pt, mass):    # ITERATE OVER EACH JET IN EVENT
            
            drawCircle(eta, phi)    # DRAW CIRCLE AROUND JET
            if phi + r > np.pi:  drawCircle(eta, -2*np.pi+phi)    # PHI WRAPPING
            if phi - r < -np.pi: drawCircle(eta,  2*np.pi+phi)    # PHI WRAPPING
    
            ax.scatter(eta, phi, s=100*np.log(1+pt), marker="x", label=f"pT = {np.round(pt,2)} GeV\nMass = {np.round(mass,2)} GeV")    # PLOT JET

    for part in specialParts:
        spec_eta, spec_phi, spec_pt, spec_mass, spec_pdg = part[event].eta, part[event].phi, part[event].pt, part[event].mass, part[event].pdgId
        for eta, phi, pt, mass, pdg in zip(spec_eta, spec_phi, spec_pt, spec_mass, spec_pdg):
            mark = "X" if abs(pdg)==24 else "*"
            ax.scatter( eta, phi, s= 50*np.log(1+pt), marker=mark, label=f"PDG ID: {pdg}\npT = {pt}\nMass = {mass}" )
    
    # Set axis labels, limits, and grid
    mplhep.cms.label(llabel=f"Private work (phase 2 simulation). All final state particles.", rlabel='14 TeV, 200 PU', fontsize=12)
    ax.set_xlabel('η', fontsize=12)
    ax.set_ylabel('ϕ', fontsize=12)
    ax.set_ylim((-np.pi, np.pi))
    ax.set_xlim((-5, 5))
    ax.set_aspect('equal', 'box')
    ax.grid()

    # Add legend manually
    ax.legend()

    # Show the plot
    plt.show()


def getQuarks(gen):
    
    """ STATUS MASKS """
    global isHardProcess
    global isFromHardProcess
    global isFromHardProcessPreFSR
    global isFirstCopy
    global isLastCopy
    global isStable
    
    # this particle is part of the hard process
    isHardProcess = ((gen.statusFlags >> 7) & 1) == 1
    # this particle is the direct descendant of a hard process particle of the same pdg id
    isFromHardProcess = ((gen.statusFlags >> 8) & 1) == 1
    # this particle is the direct descendant of a hard process particle of the same pdg id
    isFromHardProcessPreFSR = ((gen.statusFlags >> 11) & 1) == 1
    # this particle is the first copy of the particle in the chain with the same pdg id
    isFirstCopy = ((gen.statusFlags >> 12) & 1) == 1
    # this particle is the last copy of the particle in the chain with the same pdg id
    isLastCopy = ((gen.statusFlags >> 13) & 1) == 1
    # is particle stable?
    isStable = (gen.status == 1)
    
    """ PDG ID MASKS """
    global isGluon
    global isZBoson
    global isWBoson
    global isHBoson
    
    global isUQuark
    global isDQuark
    global isCQuark
    global isSQuark
    global isTQuark
    global isBQuark
    
    # this particle is a gluon
    isGluon = gen.pdgId == 21
    # this particle is a W, Z or H boson
    isZBoson = abs(gen.pdgId) == 23
    isWBoson = abs(gen.pdgId) == 24
    isHBoson = abs(gen.pdgId) == 25
    # this particle is a c or b quark
    isUQuark = abs(gen.pdgId) == 2
    isCQuark = abs(gen.pdgId) == 4
    isTQuark = abs(gen.pdgId) == 6
    isDQuark = abs(gen.pdgId) == 1
    isSQuark = abs(gen.pdgId) == 3
    isBQuark = abs(gen.pdgId) == 5

    quarkFstMask = ((isUQuark | isDQuark | isCQuark | isSQuark | isTQuark | isBQuark) & isFirstCopy & isFromHardProcess)
    quarkLstMask = ((isUQuark | isDQuark | isCQuark | isSQuark | isTQuark | isBQuark) & isLastCopy & isFromHardProcess)

    fstCopies, lstCopies = gen[quarkFstMask], gen[quarkLstMask]
    
    return fstCopies[fstCopies.pt>0], lstCopies[lstCopies.pt>0]

# def getQuarks(genParts):
#     defmasks(gen=genParts)
#     quarkMask = ((isUQuark | isDQuark | isCQuark | isSQuark | isTQuark | isBQuark) & isFirstCopy & isFromHardProcess)
#     return genParts[quarkMask]

def boostedVs(quarks, lst_quarks, genParts, bosonPDG, minMass=0, r=0.8):
    """
    This function gets all first-copy quarks from the hard process of the event
    It then splits them into particles and antiparticles and finds every combination of the two arrays
    Pairs which don't share the same parent are then discarded
    For each pair the parent is then found and it is asserted that the parent is a "V" boson
    """
    
    # Split array of quarks into quarks and antiquarks
    q, qbar = quarks[(quarks.pdgId > 0)], quarks[quarks.pdgId < 0]
    ql, qlbar = lst_quarks[(lst_quarks.pdgId > 0)], lst_quarks[lst_quarks.pdgId < 0]
    
    # Get a quark-antiquark pair for every possible combination
    qs, qbars = ak.unzip( ak.cartesian([q, qbar]) )
    qls, qlbars = ak.unzip( ak.cartesian([ql, qlbar]) )
    
    # Create mask of if combination comes from same parent - to get quarks which are decay products of same boson. Then apply mask
    sameMother = (qs.genPartIdxMother == qbars.genPartIdxMother)
    sameCone = qls.deltaR(qlbars) < r    # Can also assert that q and qbar must lie within 1.6 of each other to be able to be inside same l1jet cone
    qs, qbars = qs[sameMother & sameCone], qbars[sameMother & sameCone]
    qls, qlbars = qls[sameMother & sameCone], qlbars[sameMother & sameCone]
    
    # Find parent particle for pair
    parents = genParts[qs.genPartIdxMother]
    # Create mask to get only pairs where the parent is a V boson then apply it
    isVboson = abs(parents.pdgId) == bosonPDG
    isValidMass = parents.mass > minMass
    parentsMask = isVboson & isValidMass
    parents, qs, qbars = parents[parentsMask], qs[parentsMask], qbars[parentsMask]
    qls, qlbars = qls[parentsMask], qlbars[parentsMask]
    
    # qls and qlbars are the last quarks in the chain, qs and qbars are first quarks
    return qs, qbars, parents, qls, qlbars


def jetQuarkMatcher(l1jet, q1, q2, genParts, bosonPDG, r=0.8):
    jets, q1s = ak.unzip( ak.cartesian([l1jet, q1]) )
    jets, q2s = ak.unzip( ak.cartesian([l1jet, q2]) )
    
    qs_inConeMask = (jets.deltaR(q1s) < r) & (jets.deltaR(q2s) < r)
    q1s_inCone, q2s_inCone, jets_inCone = q1s[qs_inConeMask], q2s[qs_inConeMask], jets[qs_inConeMask]

    # Get parents whose decay quarks are both captured by l1jet
    parents_inCone = genParts[q1s_inCone.genPartIdxMother]
    
    # Assert that parent must be a W boson
    isVboson = abs(parents_inCone.pdgId) == bosonPDG
    parents_inCone, q1s_inCone, q2s_inCone = parents_inCone[isVboson], q1s_inCone[isVboson], q2s_inCone[isVboson]
    
    return parents_inCone, jets_inCone