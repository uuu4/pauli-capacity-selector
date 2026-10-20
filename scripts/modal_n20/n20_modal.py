"""HOPE n=20 (ionosphere, Fashion-MNIST) on Modal A100, resumed from checkpoints.

Colab CLI sessions died after ~25-30 min four times (2026-10-17), so the run moved here.
Notebooks: d1off_t11{io,fm}bign20v3_FULL.ipynb (self-contained, D1 off), workdir = Volume /vol
(data/, *_ckpt_FULL, *_kernels_FULL). Checkpoints are committed to the Volume every 10 min.

  uv run --with modal modal deploy n20_modal.py
  python: modal.Function.from_name("hope-n20", "run").spawn("iono")  /  .spawn("fm")
Results: /vol/out/<job>_FULL_output.ipynb, /vol/t11*bign20v3_FULL_*.json, /vol/out/<job>.log
"""
from pathlib import Path
import modal

HERE = Path(__file__).parent
app = modal.App("hope-n20")
vol = modal.Volume.from_name("hope-n20-vol", create_if_missing=True)
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("build-essential")
    .pip_install("cupy-cuda12x==13.6.0", "numpy==1.26.4", "scipy==1.13.1", "scikit-learn==1.6.1",
                 "matplotlib==3.8.4", "jupyter", "nbconvert", "ipykernel", "pillow", "tqdm")
    .add_local_dir(HERE / "nb", "/root/nb", copy=True)
)
NB = {"iono": "d1off_t11iobign20v3_FULL.ipynb", "fm": "d1off_t11fmbign20v3_FULL.ipynb"}


@app.function(image=image, gpu="A100", volumes={"/vol": vol}, timeout=23 * 3600, memory=32768)
def run(job: str):
    import os, subprocess, threading, time, glob
    os.makedirs("/vol/out", exist_ok=True)
    log = open(f"/vol/out/{job}.log", "a")
    def L(s):
        log.write(time.strftime("%Y-%m-%d %H:%M:%S UTC ", time.gmtime()) + s + "\n"); log.flush()
    tag = "t11iobign20v3" if job == "iono" else "t11fmbign20v3"
    cnt = lambda: len(glob.glob(f"/vol/{tag}_ckpt_FULL/*.pkl"))
    L(f"basladi ckpt={cnt()}")
    stop = threading.Event()
    def committer():
        while not stop.wait(600):
            try:
                vol.commit(); L(f"commit ckpt={cnt()}")
            except Exception as e:
                L(f"commit hata {e!r}")
    threading.Thread(target=committer, daemon=True).start()
    env = dict(os.environ, HOPE_WORKDIR="/vol", HOPE_BUDGET_H="60")
    t = time.time()
    p = subprocess.run(["jupyter", "nbconvert", "--to", "notebook", "--execute", f"/root/nb/{NB[job]}",
                        "--output", f"/vol/out/{job}_FULL_output.ipynb", "--ExecutePreprocessor.timeout=-1"],
                       env=env, capture_output=True, text=True)
    stop.set()
    L(f"BITTI rc={p.returncode} {(time.time() - t) / 3600:.2f} sa ckpt={cnt()}\n{p.stderr[-3000:]}")
    vol.commit()
    return p.returncode
