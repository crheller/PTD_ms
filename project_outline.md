## Go/no-go pure tone detection project

#### Prior work
Previous work has demonstrated neural enhancement of target tone representation during active go/no-go pure tone detection behavior in A1 (Fritz, David, Bagur, etc.). Only one study (Bagur) has explicitly addressed this question at the population level, however, in that study they used pseudo-populations and therefore couldn't address questions of shared neural variability. Other work (Downer 2015) has also showed that go/no-go task engagement significantly reduces correlated variability. Whether these effects contribute to state-dependent enhancement of target sound representation remains unclear. 

Additionally, recent work (Saderi) has demonstrated that some state-dependent modulations are misattributed to task when in fact they are a result of changes in global arousal levels. Thus, it remains to be seen if these arousal-dependent modulations can account for previously observed target enhancement (Fritz, David, Bagur) and/or decorrelation (Downer). 

Finally, it is still unclear if the target enhancement previously observed, particularly in single cells (Fritz, David etc.) is accompanied by a sharpening of tuning curves across A1, or alternatively, if the enehancement in coding accuracy is selective only for the target tone. In order to address this, neural population metrics are required e.g. it's unknown what the OFF-BF neurons are doing. Are they also sharpening their respective tuning curves? Are their responses non-specifically suppressed to facilitate a "pop-out" of the target? This could be hard with the amount of data we have. Could just be a point of motivation for the variable reward learning task we're working on...

#### Goals
* Replicate (roughly) prior work done in single units. 
    * Could do this with state-dependent STRF measurements. Could do this with PSHT's. In any case, highlight the fact that the array recordings sampled diverse populations of neurons, many not tuned to task stimuli (or tuned at all). Repeat the analysis after regressing out pupil?
    * Could do this with a gain/DC analysis (more similar to Sean's paper). This would be nice because it would facilitate comparison with population axes, I think.

* Replicate Downer work on noise correlations (significant reduction during active behavior), even when controlling for arousal.

* Target vs. Reference discrimination is enhanced during active behavior, even when controlling for arousal.

* Target vs. Reference discrimination is not affected by decreased correlated variability.

* Reductions in correlated variability occur in same cells that show task-dependent gain modulation. Can be interpreted as a reduction in the variability of an engagement gain signal. OR, this might not be true... Either way, address this question.

* Relationship between behavioral performance and increased discriminability / decreased correlations.

* Reference vs. Reference decoding. Arousal effects vs. behavior effects. Did it on Cosyne poster, but we dont' have a lot of small pupil data, or many reps of each Ref, which makes this question difficult. Supplemental results and cite my NAT paper?
