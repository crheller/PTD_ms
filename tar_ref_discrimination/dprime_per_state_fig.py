"""
"exploratory" figure(s) showing ref vs. tar neural dprime per 
behavior state.
Group all TARGETs / REFERENCEs (e.g. don't look at SNR yet)
"""

import charlieTools.ptd_ms.decoding as decoding
import matplotlib.pyplot as plt

decoding_axis = 'all'  # pass, passBP, act
tWin = (0.3, 0.4)        # time window on which dprime is evaluated
basemodel = 'DecTarRef_jk100_zscore_decAxis-{0}_decAxisWin-{1},{2}_TDR_tarNoise'.format(decoding_axis, tWin[0], tWin[1])
states = ['active', 'passive', 'passiveBP']

results = decoding.dprime_load_wrapper(basemodel, states=states)

# simple bar/line plot of TAR vs. REF dprime for each site
sites = results[states[0]].keys()
active_dp = np.array([results['active'][k].get_result('dp_opt_test', 'TARGET_REFERENCE', 2)[0].values[0] for k in sites])
passive_dp = np.array([results['passive'][k].get_result('dp_opt_test', 'TARGET_REFERENCE', 2)[0].values[0] for k in sites])
passiveBP_dp = np.array([results['passiveBP'][k].get_result('dp_opt_test', 'TARGET_REFERENCE', 2)[0].values[0] for k in sites])

# normalize each value to its mean across states
m = (active_dp + passive_dp + passiveBP_dp) / 3
active_dp /= m
passive_dp /= m
passiveBP_dp /= m

f, ax = plt.subplots(1, 1)

ax.bar([0, 1, 2], [active_dp.mean(), passiveBP_dp.mean(), passive_dp.mean()],  facecolor='lightgrey', edgecolor='k')
ax.plot(np.stack([active_dp, passiveBP_dp, passive_dp]), 'o-', color='k')
ax.set_xlabel(r"Behavior state")
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(['Active', 'Passive (pup matched)', 'Passive'])
ax.set_ylabel(r"$d'^2$ (Normalized across states)")

f.tight_layout()

plt.show()