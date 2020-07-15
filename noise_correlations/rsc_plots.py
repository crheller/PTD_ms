"""
Plot state-dependent noise correlations for each site for each time window.
""" 

import charlieTools.ptd_ms.utils as ut
import matplotlib.pyplot as plt
import scipy.stats as ss
import numpy as np


count_windows = [
    (0, 0.1),
    (0.1, 0.2),
    (0.2, 0.3),
    (0.3, 0.4),
    (0, 0.2),
    (0.1, 0.3),
    (0.2, 0.4)
]

for cw in count_windows:
    modelname = 'tar_rsc_{0}_{1}'.format(str(cw[0]).replace('.', ','), str(cw[1]).replace('.', ','))    
    df = ut.load_noise_correlations(modelname)

    f, ax = plt.subplots(1, 2, figsize=(8, 4))

    for state, label in zip(['act', 'pass', 'pb', 'ps'], ['active', 'passive', 'passive big', 'passive small']):

        # plot pdf
        idx = ~np.isnan(df[state].values)
        m, sd = ss.norm.fit(df[state].values[idx])
        xmin, xmax = -1, 1
        x = np.linspace(xmin, xmax, 1000)
        p = ss.norm.pdf(x, m, sd)
        ax[0].plot(x, p, linewidth=2, label=label)

    # plot per site
    vals = df.groupby(by='site').mean()[['act', 'pass', 'pb', 'ps']]
    x = np.arange(0, 4)
    ax[1].plot(np.tile(x, [vals.shape[0], 1]).T, vals.T, 'o-', color='k')
    
    ax[1].set_xticks(np.arange(0, 4))
    ax[1].set_xticklabels(['A', 'P', 'Pb', 'Ps'])
    ax[1].set_ylabel('Noise Correlation')
    ax[1].set_xlabel("Behavior state")

    ax[0].set_title('pdf')
    ax[0].set_xlabel('Noise Correlation')
    ax[0].legend(frameon=False)
    f.canvas.set_window_title(modelname)

    f.tight_layout()

plt.show()