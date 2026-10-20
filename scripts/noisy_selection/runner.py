import subprocess, os, glob, json, time
W = '/content/nsel'; os.chdir(W)
env = dict(os.environ, HOPE_ROOT=W)
units = sorted(glob.glob('units/*.json'), key=lambda f: (json.load(open(f))['n'], f))
log = open('run.log', 'a')
log.write(f'BASLADI {time.ctime()} {len(units)} birim\n'); log.flush()
for f in units:
    k = os.path.basename(f)[:-5]
    if os.path.exists(f'out/{k}.pkl'):
        continue
    t = time.time()
    p = subprocess.run(['python', 'nsel_unit.py', f, 'out'], env=env, capture_output=True, text=True)
    log.write(f'{k} rc={p.returncode} {time.time()-t:.0f}sn\n' + p.stdout[-400:] + (p.stderr[-1500:] if p.returncode else '') + '\n'); log.flush()
log.write(f'HEPSI BITTI {time.ctime()}\n'); log.close()
subprocess.run(['tar', 'czf', '/content/nsel_out.tgz', 'out', 'run.log'])
open('DONE', 'w').write('ok')
