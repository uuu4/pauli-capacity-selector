"""Compare a split recomputed on CPU (float64) with the stored GPU checkpoint (float32) of the same split.

Section 5.1 states that where two GPU platforms computed the same n = 20 splits, inputs and selected
terms agree and accuracies differ only through ties in single precision. The second GPU computation of
that check was not kept; this script repeats the check against an independent CPU recomputation.
Usage: python compare_split.py runs/n20_consistency/gpu_*.pkl runs/n20_consistency/cpu_*.pkl
Writes runs/n20_consistency/comparison.json.
"""
import json, os, pickle, sys
import numpy as np

a = pickle.load(open(sys.argv[1], 'rb'))
b = pickle.load(open(sys.argv[2], 'rb'))
m_te = int(np.asarray(a['meta']['m_te'])[0])
pt = 100 / m_te
out = dict(test_points=m_te,
           inputs_max_abs_diff=max(float(np.abs(np.asarray(a['X'][k]) - np.asarray(b['X'][k])).max()) for k in a['X']),
           selection_profiles_max_abs_diff=max(float(np.abs(np.asarray(a['mix'][k], float) - np.asarray(b['mix'][k], float)).max())
                                               for k in a['mix'] if k.endswith('|w')))
rows = []
for k in sorted(set(a['accs']) & set(b['accs'])):
    x, y = float(np.asarray(a['accs'][k])[0]), float(np.asarray(b['accs'][k])[0])
    if np.isfinite(x) and np.isfinite(y):
        rows.append((k, y - x))
for name, sel in (('noise_free', lambda k: k.endswith('|inf')), ('finite_shots', lambda k: not k.endswith('|inf'))):
    d = np.array([v for k, v in rows if sel(k)])
    out[name] = dict(entries=int(len(d)), different=int(np.sum(np.abs(d) > 1e-9)),
                     whole_test_points=bool(np.all(np.abs(d / pt - np.round(d / pt)) < 1e-6)),
                     max_abs_diff_points=float(np.abs(d).max()),
                     effect_on_30_split_mean=float(np.abs(d).max() / 30))
out['full_kernel_max_abs_diff'] = max(abs(v) for k, v in rows if k.split('|')[2] == 'A0')
out['note'] = ('Shot noise is drawn with the same seeded NumPy generator on both sides; the larger differences at '
               'finite shots come from float32 (GPU) versus float64 (CPU) arithmetic in the noisy kernels.')
json.dump(out, open(os.path.join(os.path.dirname(sys.argv[1]), 'comparison.json'), 'w'), indent=1)
print(json.dumps(out, indent=1))
