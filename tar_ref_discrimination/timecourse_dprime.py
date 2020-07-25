"""
Normalize dprime for each site across states / timeWindows
Plot dprime over the course of 400 ms of target for each state
"""

import charlieTools.ptd_ms.decoding as decoding
import matplotlib.pyplot as plt

decoding_axis = 'all'  # pass, passBP, act
tWin = [(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4)]        # time window on which dprime is evaluated
states = ['active', 'passive', 'passiveBP']

r = []
for tw in tWin:
    basemodel = 'DecTarRef_jk100_zscore_decAxis-{0}_decAxisWin-{1},{2}_TDR_tarNoise'.format(decoding_axis, tw[0], tw[1])
    results = decoding.dprime_load_wrapper(basemodel, states=states)
    r.append(results)
sites = r[0]['active'].keys()

# get dprime for each site / time / state
adp = []
pdp = []
pbdp = []
for i in range(0, len(tWin)):
    adp.append([r[i]['active'][k].get_result('dp_opt_test', 'TARGET_REFERENCE', 2)[0].values[0] for k in sites])
    pdp.append([r[i]['passive'][k].get_result('dp_opt_test', 'TARGET_REFERENCE', 2)[0].values[0] for k in sites])
    pbdp.append([r[i]['passiveBP'][k].get_result('dp_opt_test', 'TARGET_REFERENCE', 2)[0].values[0] for k in sites])

adp = np.stack(adp)
pdp = np.stack(pdp)
pbdp = np.stack(pbdp)

# normalize by mean across all states / times
m = np.stack([adp, pdp, pbdp]).mean(axis=(0, 1))
adp /= m
pdp /= m
pbdp /= m

# plot over time (only 4 bins... Should add prestim so we can see baseline)
x = np.arange(0, adp.shape[0])
f, ax = plt.subplots(1, 1, figsize=(8, 4))

m = adp.mean(axis=-1)
sem = adp.std(axis=-1) / np.sqrt(len(sites))
ax.plot(m, 'o-', color='red', lw=2, label='active')
ax.fill_between(x, m-sem, m+sem, color='red', alpha=0.3, lw=0)

m = pdp.mean(axis=-1)
sem = pdp.std(axis=-1) / np.sqrt(len(sites))
ax.plot(m, 'o-', color='blue', lw=2, label='passive')
ax.fill_between(x, m-sem, m+sem, color='blue', alpha=0.3, lw=0)

m = pbdp.mean(axis=-1)
sem = pbdp.std(axis=-1) / np.sqrt(len(sites))
ax.plot(m, 'o-', color='gold', lw=2, label='passive (pupil-matched)')
ax.fill_between(x, m-sem, m+sem, color='gold', alpha=0.3, lw=0)

ax.axvline(1, linestyle='--', color='k', label='sound onset (200ms)')

ax.legend(frameon=False, fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(['0-100ms', '100-200ms', '200-300ms', '300ms-400ms'])
ax.set_ylabel(r"$d'^2$ (normalized)")

f.tight_layout()

plt.show()
