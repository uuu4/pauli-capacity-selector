"""Compare a recomputed n = 20 split with the stored GPU checkpoint of the same split.

Section 5.1 states that where two GPU platforms computed the same n = 20 splits, inputs and selected
terms agree and accuracies differ only through ties in single precision. The second GPU computation of
the original check was not kept; this script compares the stored split with a recomputation.
Usage: python compare_split.py STORED.pkl RECOMPUTED.pkl OUT.json
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
per_arm = {}
for k, v in rows:
    arm, S = k.split('|')[2], k.split('|')[3]
    e = per_arm.setdefault(arm, dict(noise_free_max=0.0, finite_max=0.0))
    key = 'noise_free_max' if S == 'inf' else 'finite_max'
    e[key] = max(e[key], abs(v))
for e in per_arm.values():
    e['effect_on_30_split_mean'] = max(e['noise_free_max'], e['finite_max']) / 30
out['per_rule'] = per_arm
out['note'] = 'Shot noise is drawn with the same seeded NumPy generator on both sides.'
json.dump(out, open(sys.argv[3], 'w'), indent=1)
print(json.dumps(out, indent=1))
