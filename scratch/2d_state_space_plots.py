import matplotlib.pyplot as plt
import numpy as np

from charlieTools.ptd_ms import utils
import charlieTools.preprocessing as preproc

site = 'TAR010c'

rec, manager = utils.load_site(site, fs=20, return_baphy_manager=True, recache=True)  
rec = preproc.create_ptd_masks(rec, act_pup_range=2)

# get number of unique targets
targets = [t for t in rec.epochs.name.unique() if 'TAR_' in t]

# get references
references = [r for r in rec.epochs.name.unique() if 'STIM_' in r]

# get list of files
files = [f for f in rec.epochs.name if 'FILE_' in f]

exptparams = manager.get_baphy_exptparams()
# get SNRs (RelTarRefdB)
snrs = dict.fromkeys(files)
for i, f, in enumerate(files):
    _snrs = exptparams[i]['TrialObject'][1]['RelativeTarRefdB']
    snrs[f] = _snrs

# figure out which is the catch "TORC"
catch_torc = dict.fromkeys(files)
for i, f in enumerate(files):
    idx = exptparams[i]['TrialObject'][1]['OverlapRefIdx']
    catch_torc[f] = [torc for torc in references if int(torc.split('_')[-2])==int(idx)][0]

# figure out if each file was tone in noise or pure tone (True if pure tone)
ptd = dict.fromkeys(files)
for i, f in enumerate(files):
    ptd[f] = exptparams[i]['TrialObject'][1]['OverlapRefTar']=='No'


# plot passive data
rec['resp'].extract
