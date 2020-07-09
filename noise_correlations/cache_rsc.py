# for each site, compute noise correlations in the first bin (200 ms) of target
# response. Just meant as quick way to show that corr. var reduced in active
# compared to passive.
import numpy as np
import pandas as pd
from itertools import combinations
import nems_lbhb.baphy as nb
from nems.recording import Recording
import charlieTools.preprocessing as preproc
import sys
sys.path.append('/auto/users/hellerc/code/projects/PTD/')
import ptd_utils as pu

sites = ['TAR010c', 'BRT026c', 'BRT033b', 'BRT034f', 'BRT036b', 'BRT037b',
    'BRT039c', 'bbl102d', 'AMT018a', 'AMT020a', 'AMT022c', 'AMT026a']

path = '/auto/users/hellerc/results/ptd_ms/noise_correlations/'

# define start and end window of noise correlation calculation. Masking prestim, so start at 0
ts = 0   # sec
te = 0.2
filename = 'tar_rsc_{0}_{1}'.format(str(ts).replace('.', ','), str(te).replace('.', ','))

for site in sites:
    print('analyzing site: {}'.format(site))
    batch = 307
    fs = 20
    rawid = pu.which_rawids(site)
    ops = {'batch': batch, 'pupil': 1, 'rasterfs': fs, 'siteid': site, 'stim': 0,
            'rawid': rawid}
    uri = nb.baphy_load_recording_uri(**ops)
    rec = Recording.load(uri)
    rec['resp'] = rec['resp'].rasterize()
    rec = rec.and_mask(['HIT_TRIAL', 'PASSIVE_EXPERIMENT'])

    rec = rec.and_mask(['PreStimSilence', 'PostStimSilence'], invert=True)
    rec = rec.apply_mask(reset_epochs=True)

    rec = preproc.create_ptd_masks(rec)

    rec_act = rec.copy()
    rec_act['mask'] = rec['a_mask']
    rec_act = rec_act.apply_mask(reset_epochs=True)

    rec_pass = rec.copy()
    rec_pass['mask'] = rec['p_mask']
    rec_pass = rec_pass.apply_mask(reset_epochs=True)

    rec_passBig = rec.copy()
    rec_passBig['mask'] = rec['pb_mask']
    rec_passBig = rec_passBig.apply_mask(reset_epochs=True)

    rec_passSmall = rec.copy()
    rec_passSmall['mask'] = rec['ps_mask']
    rec_passSmall = rec_passSmall.apply_mask(reset_epochs=True)

    all_tar_epochs = np.unique([e for e in rec.epochs.name if (('STIM_' in e) | ('TAR_' in e)) & ('TORC' not in e)]).tolist()
    t_act_epochs = np.unique([e for e in rec_act.epochs.name if (('STIM_' in e) | ('TAR_' in e)) & ('TORC' not in e)]).tolist()
    t_pass_epochs = np.unique([e for e in rec_pass.epochs.name if (('STIM_' in e) | ('TAR_' in e)) & ('TORC' not in e)]).tolist()
    t_pb_epochs = np.unique([e for e in rec_passBig.epochs.name if (('STIM_' in e) | ('TAR_' in e)) & ('TORC' not in e)]).tolist()
    t_ps_epochs = np.unique([e for e in rec_passSmall.epochs.name if (('STIM_' in e) | ('TAR_' in e)) & ('TORC' not in e)]).tolist()

    # make sure targets exists in all state epochs at least 3 times
    tar_epochs = [t for t in all_tar_epochs if ((rec_act.epochs.name==t).sum() > 2) & \
                                            ((rec_pass.epochs.name==t).sum() > 2) & \
                                            ((rec_passBig.epochs.name==t).sum() > 2) & \
                                            ((rec_passSmall.epochs.name==t).sum() > 2)]
    small_pupil = True
    if tar_epochs == []:
        print("Small pupil didn't have enough data")
        small_pupil = False
        tar_epochs = [t for t in all_tar_epochs if ((rec_act.epochs.name==t).sum() > 2) & \
                                        ((rec_pass.epochs.name==t).sum() > 2) & \
                                        ((rec_passBig.epochs.name==t).sum() > 2)]
    
    print('number of targets: {}'.format(len(tar_epochs)))

    act = rec_act['resp'].extract_epochs(tar_epochs)
    pas = rec_pass['resp'].extract_epochs(tar_epochs)
    pb = rec_passBig['resp'].extract_epochs(tar_epochs)
    if small_pupil:
        ps = rec_passSmall['resp'].extract_epochs(tar_epochs)

    # extract time window
    s = int(ts * fs)
    e = int(te * fs)
    for k in tar_epochs:
        act[k] = act[k][:, :, s:e].mean(axis=-1)
        pas[k] = pas[k][:, :, s:e].mean(axis=-1)
        pb[k] = pb[k][:, :, s:e].mean(axis=-1)
        if small_pupil:
            ps[k] = ps[k][:, :, s:e].mean(axis=-1)

    for i, k in enumerate(tar_epochs):
        a_resp = act[k]
        print('{0} active reps of target: {1}'.format(a_resp.shape[0], k))
        a_mean = a_resp.mean(axis=0)
        a_std = a_resp.std(axis=0)
        a_residual = a_resp - a_mean
        a_residual /= a_std

        p_resp = pas[k]
        print('{0} passive reps of target: {1}'.format(p_resp.shape[0], k))
        p_mean = p_resp.mean(axis=0)
        p_std = p_resp.std(axis=0)
        p_residual = p_resp - p_mean
        p_residual /= p_std

        pb_resp = pb[k]
        print('{0} passive big pupil reps of target: {1}'.format(pb_resp.shape[0], k))
        pb_mean = pb_resp.mean(axis=0)
        pb_std = pb_resp.std(axis=0)
        pb_residual = pb_resp - pb_mean
        pb_residual /= pb_std

        if small_pupil:
            ps_resp = ps[k]
            print('{0} passive small pupil reps of target: {1}'.format(ps_resp.shape[0], k))
            ps_mean = ps_resp.mean(axis=0)
            ps_std = ps_resp.std(axis=0)
            ps_residual = ps_resp - ps_mean
            ps_residual /= ps_std

        if i == 0:
            a_residuals = a_residual
            p_residuals = p_residual
            pb_residuals = pb_residual
            if small_pupil:
                ps_residuals = ps_residual
        else:
            a_residuals = np.concatenate((a_residuals, a_residual), axis=0)
            p_residuals = np.concatenate((p_residuals, p_residual), axis=0)
            pb_residuals = np.concatenate((pb_residuals, pb_residual), axis=0)
            if small_pupil:
                ps_residuals = np.concatenate((ps_residuals, ps_residual), axis=0)

    # compute corr matrix
    a_corr = np.corrcoef(a_residuals.T)
    p_corr = np.corrcoef(p_residuals.T)
    pb_corr = np.corrcoef(pb_residuals.T)
    if small_pupil:
        ps_corr = np.corrcoef(ps_residuals.T)

    # save to dataframe
    pairs = list(combinations(np.arange(0, rec['resp'].shape[0]), 2))
    pair_idx = [rec['resp'].chans[p[0]]+'_'+rec['resp'].chans[p[1]] for p in pairs]
    df = pd.DataFrame(index=pair_idx, columns=['act_rsc', 'pass_rsc', 'passBig_rsc', 'passSmall_rsc', 'site'])
    for i, p in enumerate(pairs):
        df.loc[pair_idx[i]]['act_rsc'] = a_corr[p[0], p[1]]
        df.loc[pair_idx[i]]['pass_rsc'] = p_corr[p[0], p[1]]
        df.loc[pair_idx[i]]['passBig_rsc'] = pb_corr[p[0], p[1]]
        if small_pupil:
            df.loc[pair_idx[i]]['passSmall_rsc'] = ps_corr[p[0], p[1]]
        df.loc[pair_idx[i]]['site'] = site

    print('saving results for site: {}'.format(site))
    df.to_csv(path+filename+'_'+site+'.csv')