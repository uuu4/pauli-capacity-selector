"""Recompute one n = 20 split with the code of the notebook that produced it (GPU with CuPy if available, else CPU in float64).

Unit: ionosphere, n = 20, depth 8, eps = 1, bandwidth 0.15, split 0 (about 75 min on one CPU core, ~3 GB RAM).
Usage: python run_split_cpu.py notebooks/real/d1off_t11iobign20v3_FULL.ipynb <empty work dir>
The split checkpoint is written to <work dir>/t11iobign20v3_ckpt_FULL/.
"""
import json, os, sys, time, pickle
NB = sys.argv[1]; WORK = sys.argv[2]; MODE_T = sys.argv[3] if len(sys.argv) > 3 else 'run'
os.environ['HOPE_WORKDIR'] = WORK
nb = json.load(open(NB))
src = [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']
c3 = src[3].rsplit('_validate()', 1)[0]
c4 = src[4].split('RES_A, RES_S, RES_M, RES_E, RES_C, RES_L, RES_D = {}', 1)[0]
g = {'__name__': '__main__'}
for code in (src[0], src[1], src[2], c3, c4):
    exec(code, g)
np = g['np']
X_, y_ = g['load_real']('ionosphere')
SPL = g['stratified_splits'](y_, g['N_SPLIT'], g['SEED0'], g['TRAIN_FRAC'])
if MODE_T == 'time':
    n = 20; Xs = np.random.default_rng(0).uniform(0, 2*np.pi, (4, n))
    lay = g['pool_layout'](n, 2, max_flips=g['max_flips_for'](n), seed=g['SEED0'], keep_low=2)
    t = time.time(); psi = g['enc'](Xs, n, 8, 1.0, 0.15); t1 = time.time()
    E = g['pauli_expect_fast'](psi, n, lay); t2 = time.time()
    print(f'encode 4: {t1-t:.1f}s  expect 4: {t2-t1:.1f}s  terms {E.shape}')
    sys.exit()
t0 = time.time()
out = g['run_unit']('ionosphere', X_, y_, 20, 8, 1.0, 0.15, SPL[:1], log=t0)
print('bitti', time.time() - t0)
