"""
Compute the noise correlation difference axis between active and pupil matched passive
For the specified target time window
"""
import charlieTools.ptd_ms.utils as utils
import charlieTools.preprocessing as preproc
import charlieTools.ptd_ms.decoding as decoding

import nems.db as nd
import nems

import os
import sys
import numpy as np
import pickle
import logging
log = logging.getLogger(__name__)

# ============================== SAVE PARAMETERS ===================================
path = '/auto/users/hellerc/results/ptd_ms/latent_variables/'

# =============================== SET RNG SEED =====================================
np.random.seed(123)

# ============================ set up dbqueue stuff ================================
if 'QUEUEID' in os.environ:
    queueid = os.environ['QUEUEID']
    nems.utils.progress_fun = nd.update_job_tick
else:
    queueid = 0

if queueid:
    log.info("Starting QUEUEID={}".format(queueid))
    nd.update_job_start(queueid)

# ========================== read and parse system arguments ========================
site = sys.argv[1]  
batch = int(sys.argv[2])
modelname = sys.argv[3]
options = modelname.split('_')

tWin = (0, 0.2)
for op in options:
    if 'tWin-' in op:
        tWin = (np.float(op.split('-')[1].split(',')[0]), np.float(op.split('-')[1].split(',')[1]))
        log.info("Option 'time window defined between': {0} and {1} sec".format(tWin[0], tWin[1]))

# ================================ Load data =====================================
rec = utils.load_site(site, fs=20)
rec = preproc.create_ptd_masks(rec, act_pup_range=2)
rec = rec.and_mask(['HIT_TRIAL', 'PASSIVE_EXPERIMENT'])
rec = rec.and_mask(['PreStimSilence', 'PostStimSilence'], invert=True)


# ================================ Perform analysis on the raw data =======================================
tar_epochs = [e for e in rec.epochs.name.unique() if 'TAR_' in e]
active_dict = rec['resp'].extract_epochs(tar_epochs, mask=rec['a_mask'], allow_incomplete=True)
passive_dict = rec['resp'].extract_epochs(tar_epochs, mask=rec['p_mask'], allow_incomplete=True)

log.info("Collapse responses over the relevant time window")
active_dict = decoding.squeeze_time_dim(active_dict, fs=rec['resp'].fs, twin=tWin, keepdims=True)
passive_dict = decoding.squeeze_time_dim(passive_dict, fs=rec['resp'].fs, twin=tWin, keepdims=True)

log.info("Compute z-scores of responses for noise correlation calculation")
active_dict = preproc.zscore_per_stim(active_dict, d2=active_dict)
passive_dict = preproc.zscore_per_stim(passive_dict, d2=passive_dict)

log.info("Concatenate responses to all stimuli")
eps = list(set(list(active_dict.keys())).intersection(list(passive_dict.keys())))
nCells = active_dict[eps[0]].shape[1]
for i, k in enumerate(eps):
    if i == 0:
        a_matrix = np.transpose(active_dict[k], [1, 0, -1]).reshape(nCells, -1)
    else:
        a_matrix = np.concatenate((a_matrix, np.transpose(active_dict[k], [1, 0, -1]).reshape(nCells, -1)), axis=-1)
for i, k in enumerate(eps):
    if i == 0:
        p_matrix = np.transpose(passive_dict[k], [1, 0, -1]).reshape(nCells, -1)
    else:
        p_matrix = np.concatenate((p_matrix, np.transpose(passive_dict[k], [1, 0, -1]).reshape(nCells, -1)), axis=-1)

# get difference of covariance matrices (exlcuding diagonals)
passive = np.cov(p_matrix)
np.fill_diagonal(passive, 0)
active = np.cov(a_matrix)
np.fill_diagonal(active, 0)
diff = passive - active
evals, evecs = np.linalg.eig(diff)
idx = np.argsort(evals)[::-1]
evals = evals[idx]
evecs = evecs[:, idx]

beta2 = evecs[:, [0]]
beta2_lambda = evals[0]

# save results
lv_results = {}
lv_results['site'] = site
lv_results['beta2'] = beta2
lv_results['beta2_lambda'] = beta2_lambda

# =================================== Shuffle state and repeat analysis =======================================
# randomly draw from all data and compute active / passive diff eigenvalue niter times
shuffled_eval1 = []
niters = 20
all_matrix = np.concatenate((a_matrix, p_matrix), axis=-1)
nreps_active = a_matrix.shape[-1]
nreps_passive = p_matrix.shape[-1]
inidces = np.arange(0, all_matrix.shape[-1])
for k in range(niters):
    # random draw a/p
    pinds = np.random.choice(inidces, nreps_passive, replace=False).tolist()
    ainds = list(set(idx) - set(pinds))
    rd_passive = all_matrix[:, pinds]
    rd_active = all_matrix[:, ainds]

    rand_passive = np.cov(rd_passive)
    np.fill_diagonal(rand_passive, 0)
    rand_active = np.cov(rd_active)
    np.fill_diagonal(rand_active, 0)
    diff = rand_passive - rand_active
    evals, evecs = np.linalg.eig(diff)
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]

    beta2_lambda = evals[0]

    shuffled_eval1.append(beta2_lambda)



# save shuffled results / signficance
mean_shuf_beta2_lambda = np.mean(shuffled_eval1)
sem_shuf_beta2_lambda = np.std(shuffled_eval1) / np.sqrt(niters)
lv_results['shuf_beta2_lambda'] = mean_shuf_beta2_lambda
lv_results['shuf_beta2_lambda_sem'] = sem_shuf_beta2_lambda

# figure out if dim is significant
if (lv_results['beta2_lambda'] - lv_results['shuf_beta2_lambda']) > lv_results['shuf_beta2_lambda_sem']: lv_results['beta2_sig'] = True
else: lv_results['beta2_sig'] = False

# ===================================== SAVE RESULTS FOR SITE =================================================
# pickle the results
fn = path + '{0}_nclv_tWin-{1},{2}.pickle'.format(site, tWin[0], tWin[1])
with open(fn, 'wb') as handle:
    pickle.dump(lv_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

log.info("Saved results to {}".format(fn))

if queueid:
    nd.update_job_complete(queueid)