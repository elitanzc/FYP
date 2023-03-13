# FYP
FYP codes on AccAltProj, IRCUR and their unrolled models

### Source of truth:
Refer to `Learned_AccAltProj_onIRCUR_newThres copy 2.ipynb` for the results.

### Differences between the copies:
* `Learned_AccAltProj_onIRCUR_newThres.ipynb` ==> `max_iter` = 50 (double for IRCUR), classical algos run full iterations
* `Learned_AccAltProj_onIRCUR_newThres copy.ipynb` ==> `max_iter` = 50, classical algos run full iterations
* `Learned_AccAltProj_onIRCUR_newThres copy 2.ipynb` ==> `max_iter` = 50, classical algos stop when err < tol
* `Learned_AccAltProj_onIRCUR_newThres copy 3.ipynb` ==> `max_iter` = 20, classical algos stop when err < tol

### Other files:
* `Learned_AccAltProj.ipynb` ==> training samples are estimated using AccAltProj instead of IRCUR
* `Learned_IRCUR.ipynb` ==> unrolled model cased on IRCUR
* `Learned_RieCUR.ipynb` ==> unrolled model cased on RieCUR
* `helper.py` ==> contains functions used across all files
