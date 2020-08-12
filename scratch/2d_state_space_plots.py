import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA 

from charlieTools.ptd_ms import utils
import charlieTools.preprocessing as preproc
import charlieTools.plotting as cplt

site = 'TAR010c'
do_TDR = True   
do_PCA = False

rec, manager = utils.load_site(site, fs=20, return_baphy_manager=True, recache=False)  
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
stim_bins = int(0.2 * rec['resp'].fs)   # first 200 ms
prestim_bins = rec['resp'].extract_epoch('PreStimSilence').shape[-1]

# TORC responses
catch_torc_resp = rec['resp'].extract_epoch(catch_torc[files[0]])[:, :, prestim_bins:(prestim_bins+stim_bins)].mean(axis=-1)
# remove early FAs
kidx = np.argwhere(~np.isnan(catch_torc_resp[:, 0])).squeeze()
catch_torc_resp = catch_torc_resp[kidx]

# TARGET responses
target_resp = []
for tar in targets:
    tr = rec['resp'].extract_epoch(tar)[:, :, prestim_bins:(prestim_bins+stim_bins)].mean(axis=-1)
    # remove early trials
    kidx = np.argwhere(~np.isnan(tr[:, 0])).squeeze()
    target_resp.append(tr[kidx])

# z-score responses
m = np.mean([np.concatenate(target_resp, axis=0).mean(axis=0), catch_torc_resp.mean(axis=0)])
sd = np.mean([np.concatenate(target_resp, axis=0).std(axis=0), catch_torc_resp.std(axis=0)])
target_resp = [(t-m)/sd for t in target_resp]
catch_torc_resp -= m
catch_torc_resp /= sd

all_tars = np.stack(target_resp).reshape(-1, tr.shape[-1])

if do_TDR:
    # use the pooled target noise to define NOISE axes
    centered_targets = [t-t.mean(axis=0, keepdims=True) for t in target_resp]
    centered_targets = np.stack(centered_targets).reshape(-1, tr.shape[-1])

    pca = PCA(n_components=1)
    pca.fit(centered_targets)
    noise_axis = pca.components_

    # get DISCRIMINATION axis
    dU = catch_torc_resp.mean(axis=0) - all_tars.mean(axis=0, keepdims=True)
    dU /= np.linalg.norm(dU)

    # ortho axis
    noise_on_dec = (np.dot(noise_axis, dU.T)) * dU
    orth_ax = noise_axis - noise_on_dec
    orth_ax /= np.linalg.norm(orth_ax)

    tdr_weights = np.concatenate([dU, orth_ax], axis=0)

    # project responses and plot
    ref_proj = catch_torc_resp.dot(tdr_weights.T)
    tar_proj = []
    for i, tar in enumerate(targets):
        tar_proj.append(target_resp[i].dot(tdr_weights.T))

elif do_PCA:
    all_resp = np.concatenate([all_tars.mean(axis=0, keepdims=True), catch_torc_resp.mean(axis=0, keepdims=True)], axis=0)
    pca = PCA(n_components=2)
    pca.fit(all_resp)

    # project responses and plot
    ref_proj = catch_torc_resp.dot(pca.components_.T)
    tar_proj = []
    for i, tar in enumerate(targets):
        tar_proj.append(target_resp[i].dot(pca.components_.T))


f, ax = plt.subplots(1, 1, figsize=(5, 5))

ax.axhline(0, linestyle='--', color='grey')
ax.axvline(0, linestyle='--', color='grey')

ax.scatter(ref_proj[:, 0], ref_proj[:, 1], alpha=0.4, s=10)
for i, tar in enumerate(targets):
    ax.scatter(tar_proj[i][:, 0], tar_proj[i][:, 1], alpha=0.4, s=10)


f.tight_layout()

plt.show()

