"""
Procedure:
    1) Load data
    2) Make data masks (Active, pupil matched, passive, miss trials)
    4) Generate est/val sets (each est/val dataset should be shape Neuron X Rep X Stim)
    5) Preprocess est set (apply same preprocessing to val)
    6) Dimensionality reduction (on est, apply same to val)
    7) Compute dprime, save metrics
    8) Split data into active / passive / passive matched / miss trials
"""
import pickle
import numpy as np
from itertools import combinations
import os
import sys
import pandas as pd
import copy

import charlieTools.ptd_ms.utils as utils
import charlieTools.preprocessing as preproc
import charlieTools.ptd_ms.decoding as decoding
import charlieTools.ptd_ms.dim_reduction as dr

import nems.db as nd
import nems

import logging
log = logging.getLogger(__name__)

# ================================ SET RNG STATE ===================================
np.random.seed(123)

# ============================== SAVE PARAMETERS ===================================
path = '/auto/users/hellerc/results/ptd_ms/dprime/'

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

njacks = 10
zscore = False
regress_pupil = False
use_xforms = False

do_Raw = True           # use all the data
do_TDR = False          # do dimensionality reduction with TDR
dc_axis = 'all'         # if 'all', compute discrimination axis using all data (options are pass, act, passBP)
dc_axisWin = (0, 0.2)   # time window (rel. to sound onset) on which to define decoding axis (e.g. pass decAxisWin-0.1,0.3 for (0.1, 0.3))

nc_lv = False           # load delta noise correlation axis (latent variable loadings)
tar_noise_axis = False  # define noise axis (TDR2) based on Target responses only
eev = False             # If True, don't do cross-validation

for op in options:
    if 'jk' in op:
        njacks = int(op[2:])
        log.info("Option 'jackknifes': {}".format(njacks))
    if 'zscore' in op:
        log.info("Option 'zscore': True")
        zscore = True
    if op == 'pr':
        log.info("Option 'regress first order pupil': True")
        regress_pupil = True
    if op == 'rm2':
        log.info("Option 'use xforms': True")
        use_xforms = True
    if 'decAxis-' in op:
        dc_axis = op.split('-')[1]
        log.info("Option 'decoding axis defined on': {}".format(dc_axis))
    if 'decAxisWin-' in op:
        dc_axisWin = (np.float(op.split('-')[1].split(',')[0]), np.float(op.split('-')[1].split(',')[1]))
        log.info("Option 'decoding axis defined between': {0} and {1} sec".format(dc_axisWin[0], dc_axisWin[1]))
    if op == 'nclv':
        log.info("Option 'load noise correlation latent variable': True")
        nc_lv = True
    if op == 'TDR':
        log.info("Option 'perform TDR dim reduction': True")
        do_TDR = True
        do_Raw = False
    if op=='tarNoise':
        log.info("Option 'fix TDR2 to noise axis for Target only data': True")
        tar_noise_axis = True
    if op=='eev':
        log.info("Option 'Don't perform cross-validation': True")
        eev = True

# ================ load LV information for this site =======================
if nc_lv:
    nc_lv_path = '/auto/users/hellerc/results/ptd_ms/LV/nc_zscore_lvs.pickle'
    log.info("loading LV information from {}".format(nc_lv_path))
    with open(fn, 'rb') as handle:
        lv_results = pickle.load(handle)
    beta1 = lv_results[site]['beta1']
    beta2 = lv_results[site]['beta2']
else:
    beta1 = None
    beta2 = None
    
# ================================= load recording ==================================
rec = utils.load_site(site, fs=20)
rec = preproc.create_ptd_masks(rec, act_pup_range=2)
rec = rec.and_mask(['HIT_TRIAL', 'PASSIVE_EXPERIMENT'])
rec = rec.and_mask(['PreStimSilence', 'PostStimSilence'], invert=True)

# =========================== Figure out if multiple Targets ========================
targets = [t for t in rec.epochs.name.unique() if 'TAR_' in t]
if len(targets) > 1:
    log.info("Multiple targets for site {}".format(site))
    combos = list(combinations(targets+['REFERENCE'], 2))
    combos = [c for c in combos if 'REFERENCE' in c]
    combos += [('TARGET', 'REFERENCE')]

else:
    combos = [(targets[0], 'REFERENCE'), ('TARGET', 'REFERENCE')]

if zscore:
    log.info("z-score data for each neuron")
    rdat = rec.apply_mask()['resp'].as_continuous()
    m = rdat.mean(axis=-1)
    sd = rdat.std(axis=-1)
    rnew = ((rec['resp'].as_continuous().T - m) / sd).T
    rec['resp'] = rec['resp']._modified_copy(rnew)
else:
    log.info("Center data for each neuron")
    rdat = rec.apply_mask()['resp'].as_continuous()
    m = rdat.mean(axis=-1)
    rnew = ((rec['resp'].as_continuous().T - m)).T
    rec['resp'] = rec['resp']._modified_copy(rnew)

# =============================== reshape / sort data ===============================
sound_epochs = targets + ['REFERENCE', 'TARGET']
resp_dict = rec['resp'].extract_epochs(sound_epochs, mask=rec['mask'], allow_incomplete=True)
a_mask = rec['a_mask'].extract_epochs(sound_epochs, mask=rec['mask'], allow_incomplete=True)
p_mask = rec['p_mask'].extract_epochs(sound_epochs, mask=rec['mask'], allow_incomplete=True)
pb_mask = rec['pb_mask'].extract_epochs(sound_epochs, mask=rec['mask'], allow_incomplete=True)
# not really enough data to do much with misses here, I don't think
#miss_mask = rec['miss_mask'].extract_epochs(sound_epochs, mask=rec['mask'], allow_incomplete=True)

# ============================ generate est / val sets ==============================
# sort of dumb, but think the easiest way to do this is on a per mask, per epoch basis
# since number of repetitions aren't necessarily balanced across these categories
# this means that for each state/sound combination, there will be a list of est / val sets
all_data, act_data, pass_data, pb_data = decoding.get_est_val_sets(resp_dict, 
                                                                   njacks=njacks,
                                                                   masks=[
                                                                       a_mask,
                                                                       p_mask,
                                                                       pb_mask
                                                                   ], tolerance=50, min_reps=10)
# NOTE - If computing decoding axis across all states, will still need to make sure to balance
# across state. e.g. if there are more active reps, this would make the decoding axis biased to 
# active. This will get handled below, when defining "decoding_data"

# ============================ Define decoding axis data =============================
if dc_axis == 'all':
    decoding_data = copy.deepcopy(all_data)
elif dc_axis == 'pass':
    decoding_data = copy.deepcopy(pass_data)
elif dc_axis == 'act':
    decoding_data = copy.deepcopy(act_data)
elif dc_axis == 'passBP':
    decoding_data = copy.deepcopy(pb_data)
elif dc_axis not in ['all', 'pass', 'act', 'passBP']:
    raise ValueError("Unknown option for decoding axis specification. Should be one of " \
                            "['all', 'pass', 'act', 'passBP']")

# squeeze / collapse out time dimension, depending on value of dc_axisWin
decoding_data = decoding.squeeze_time_dim(decoding_data, fs=rec['resp'].fs, twin=dc_axisWin)
act_data = decoding.squeeze_time_dim(act_data, fs=rec['resp'].fs, twin=dc_axisWin)
pass_data = decoding.squeeze_time_dim(pass_data, fs=rec['resp'].fs, twin=dc_axisWin)
pb_data = decoding.squeeze_time_dim(pb_data, fs=rec['resp'].fs, twin=dc_axisWin)

# =================================== get TDR2 axis ==================================
# define TDR axis using the same data as will be used for defining decoding axis
if do_TDR:
    if tar_noise_axis:
        # get noise axis on TARGET epochs only
        target_keys = [t for t in decoding_data.keys() if 'TAR_' in t]
        tdr_axes = dr.get_noise_axis_per_est(decoding_data, keys=target_keys, njacks=njacks)
    else:
        # use all data, even though each ref is low rep count
        un_keys = [t for t in decoding_data.keys() if t not in ['REFERENCE', 'TARGET']]
        tdr_axes = dr.get_noise_axis_per_est(decoding_data, keys=un_keys, njacks=njacks)


# ============================== RUN DPRIME ANALYSIS ================================
# for each discrimination, compute dprime. Save associated statistics in a per combo 
ar_list = []
pr_list = []
prbp_list = []
for c in combos:
    try:
        for ev in range(njacks):
            # get est / val / decoding data for this combination / jackknife
            # active data
            a_ed = (act_data[c[0]]['est'][ev].squeeze(), act_data[c[1]]['est'][ev].squeeze())
            a_vd = (act_data[c[0]]['val'][ev].squeeze(), act_data[c[1]]['val'][ev].squeeze())
            # passive data
            p_ed = (pass_data[c[0]]['est'][ev].squeeze(), pass_data[c[1]]['est'][ev].squeeze())
            p_vd = (pass_data[c[0]]['val'][ev].squeeze(), pass_data[c[1]]['val'][ev].squeeze())
            # pupil matched active
            pb_ed = (pb_data[c[0]]['est'][ev].squeeze(), pb_data[c[1]]['est'][ev].squeeze())
            pb_vd = (pb_data[c[0]]['val'][ev].squeeze(), pb_data[c[1]]['val'][ev].squeeze())

            # decoding data
            dd = (decoding_data[c[0]]['est'][ev].squeeze(), decoding_data[c[1]]['est'][ev].squeeze())

            if do_TDR:
                results_active = decoding.do_tdr_dprime_analysis(train_data=a_ed,
                                                                    test_data=a_vd,
                                                                    decoding_data=dd,
                                                                    tdr2_axis=tdr_axes[ev])
                results_passive = decoding.do_tdr_dprime_analysis(train_data=p_ed,
                                                                    test_data=p_vd,
                                                                    decoding_data=dd,
                                                                    tdr2_axis=tdr_axes[ev])
                results_passiveBP = decoding.do_tdr_dprime_analysis(train_data=pb_ed,
                                                                        test_data=pb_vd,
                                                                        decoding_data=dd,
                                                                        tdr2_axis=tdr_axes[ev])
                results_active.update({
                    'jack_idx': ev,
                    'n_components': 2,
                    'combo': c,
                    'site': site
                })
                results_passive.update({
                    'jack_idx': ev,
                    'n_components': 2,
                    'combo': c,
                    'site': site
                })
                results_passiveBP.update({
                    'jack_idx': ev,
                    'n_components': 2,
                    'combo': c,
                    'site': site
                })
            
            elif do_RAW:
                raise ValueError("TODO: Set up raw decoding analysis (no dim reduction)")

            else:
                raise ValueError("No decoding method specified")

            # append results
            ar_list.append(results_active)
            pr_list.append(results_passive)
            prbp_list.append(results_passiveBP)

    except:
        log.info("Not sufficient data for combo {}".format(c))

active_results = pd.DataFrame(ar_list)
passive_results = pd.DataFrame(pr_list)
passiveBP_results = pd.DataFrame(prbp_list)


# convert combos to str
active_results.loc[:, 'combo'] = ['{0}_{1}'.format(c[0], c[1]) for c in active_results.combo.values]
passive_results.loc[:, 'combo'] = ['{0}_{1}'.format(c[0], c[1]) for c in passive_results.combo.values]
passiveBP_results.loc[:, 'combo'] = ['{0}_{1}'.format(c[0], c[1]) for c in passiveBP_results.combo.values]

# cast dtypes for easy combining over jackknifes
active_results = decoding.cast_dtypes(active_results)
passive_results = decoding.cast_dtypes(passive_results)
passiveBP_results = decoding.cast_dtypes(passiveBP_results)

# collapse over results to save disk space by packing into "DecodingResults object"
log.info("Compressing results into DecodingResults object... ")
active_results = decoding.DecodingResults(active_results) 
passive_results = decoding.DecodingResults(passive_results) 
passiveBP_results = decoding.DecodingResults(passiveBP_results) 

# Save results
log.info("Saving results to {}".format(path))
if not os.path.isdir(os.path.join(path, site)):
    os.mkdir(os.path.join(path, site))

for state, results in zip(['active', 'passive', 'passiveBP'], [active_results, passive_results, passiveBP_results]):
    mn = modelname + '_{}'.format(state)
    results.save_pickle(os.path.join(path, site, mn+'.pickle'))

if queueid:
    nd.update_job_complete(queueid)