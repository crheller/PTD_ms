# for each site, compute noise correlations during the target
# response. Just meant as quick way to show that corr. var reduced in active
# compared to passive.
import numpy as np
import pandas as pd
from itertools import combinations
import nems_lbhb.baphy as nb
from nems.recording import Recording
import charlieTools.preprocessing as preproc
import charlieTools.ptd_ms.utils as pu
import charlieTools.noise_correlations as nc

sites = ['TAR010c', 'BRT026c', 'BRT033b', 'BRT034f', 'BRT036b', 'BRT037b',
    'BRT039c', 'bbl102d', 'AMT018a', 'AMT020a', 'AMT022c', 'AMT026a']

path = '/auto/users/hellerc/results/ptd_ms/noise_correlation/'

sd_from_active_pupil = 2  # number of sd from mean active pupil that counts as large pupil

# define start and end window of noise correlation calculation. Masking prestim, so start at 0
# defined in seconds
count_windows = [
    (0, 0.1),
    (0.1, 0.2),
    (0.2, 0.3),
    (0.3, 0.4),
    (0, 0.2),
    (0.1, 0.3),
    (0.2, 0.4)
]

for ts, te in count_windows:

    filename = 'tar_rsc_{0}_{1}'.format(str(ts).replace('.', ','), str(te).replace('.', ','))

    for site in sites:
        print('analyzing site: {}'.format(site))
        batch = 307
        fs = 20
        rawid = pu.which_rawids(site)
        ops = {'batch': batch, 'pupil': 1, 'rasterfs': fs, 'cellid': site, 'stim': 0,
                'rawid': rawid}
        rec = nb.baphy_load_recording_file(**ops)
        rec['resp'] = rec['resp'].rasterize()
        rec = rec.and_mask(['HIT_TRIAL', 'PASSIVE_EXPERIMENT'])

        rec = rec.and_mask(['PreStimSilence', 'PostStimSilence'], invert=True)
        rec = rec.apply_mask(reset_epochs=True)

        rec = preproc.create_ptd_masks(rec, act_pup_range=sd_from_active_pupil)

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
            act[k] = act[k][:, :, s:e].mean(axis=-1, keepdims=True)
            pas[k] = pas[k][:, :, s:e].mean(axis=-1, keepdims=True)
            pb[k] = pb[k][:, :, s:e].mean(axis=-1, keepdims=True)
            if small_pupil:
                ps[k] = ps[k][:, :, s:e].mean(axis=-1, keepdims=True)

        adf = nc.compute_rsc(act, chans=rec['resp'].chans)
        adf = adf.rename(columns={'rsc': 'act', 'pval': 'act_p'}) 
        pdf = nc.compute_rsc(pas, chans=rec['resp'].chans)
        pdf = pdf.rename(columns={'rsc': 'pass', 'pval': 'pass_p'}) 
        pbdf = nc.compute_rsc(pb, chans=rec['resp'].chans)
        pbdf = pbdf.rename(columns={'rsc': 'pb', 'pval': 'pb_p'}) 
        if small_pupil:
            psdf = nc.compute_rsc(ps, chans=rec['resp'].chans)
            psdf = psdf.rename(columns={'rsc': 'ps', 'pval': 'ps_p'}) 

            df = pd.concat([adf, pdf, pbdf, psdf], axis=1) 
        else:
            psdf = pbdf.copy()
            psdf = psdf.rename(columns={'rsc': 'ps', 'pval': 'ps_p'}) 
            psdf['ps'] = np.nan
            psdf['ps_p'] = np.nan

            df = pd.concat([adf, pdf, pbdf, psdf], axis=1) 
            
        print('saving results for site: {}'.format(site))
        df.to_csv(path+filename+'_'+site+'.csv')

