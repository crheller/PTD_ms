import nems.db as nd
import numpy as np

batch = 307
force_rerun = True
TDR = True
use_target_noise = True

sites = ['TAR010c', 'BRT026c', 'BRT033b', 'BRT034f', 'BRT036b', 'BRT037b',
    'BRT039c', 'bbl102d', 'AMT018a', 'AMT020a', 'AMT022c', 'AMT026a']
base_model = 'nclv'

# choose time windows to perform decoding
time_windows = [
    (0, 0.1),
    (0.1, 0.2),
    (0.2, 0.3),
    (0.3, 0.4),
    (0, 0.2),
    (0.2, 0.4),
]
modellist = [base_model+'_tWin-{},{}'.format(t1, t2) for (t1, t2) in time_windows]

script = '/auto/users/hellerc/code/projects/ptd_ms/noise_correlations/cache_nclv.py'
python_path = '/auto/users/hellerc/anaconda3/envs/lbhb/bin/python'

nd.enqueue_models(celllist=sites,
                  batch=batch,
                  modellist=modellist,
                  executable_path=python_path,
                  script_path=script,
                  user='hellerc',
                  force_rerun=force_rerun,
                  reserve_gb=4)