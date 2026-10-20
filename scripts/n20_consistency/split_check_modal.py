"""Recompute ionosphere n=20 d8 e1.0 bw0.15 split 0 on a Modal T4 GPU (CuPy, float32) with the notebook's own code.

Usage (from the repository root): modal run scripts/n20_consistency/split_check_modal.py
Writes runs/n20_consistency/gpu_t4_ionosphere_n20_d8_e1p0_bw0p15__s00.pkl (about 7 min on a T4).
"""
from pathlib import Path
import modal

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
app = modal.App("hope-n20-split-check")
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.12")
    .pip_install("cupy-cuda12x==13.6.0", "numpy==1.26.4", "scipy==1.13.1", "scikit-learn==1.6.1",
                 "matplotlib==3.8.4")
    .add_local_file(ROOT / "notebooks/real/d1off_t11iobign20v3_FULL.ipynb", "/root/nb.ipynb")
    .add_local_file(HERE / "run_split_cpu.py", "/root/run_split.py")
    .add_local_dir(Path.home() / "scikit_learn_data" / "openml", "/root/scikit_learn_data/openml")
)


@app.function(image=image, gpu="T4", timeout=3 * 3600, memory=16384)
def run():
    import os, subprocess
    os.makedirs("/tmp/work", exist_ok=True)
    p = subprocess.run(["python", "-u", "/root/run_split.py", "/root/nb.ipynb", "/tmp/work", "run"],
                       capture_output=True, text=True)
    f = "/tmp/work/t11iobign20v3_ckpt_FULL/ionosphere_n20_d8_e1p0_bw0p15__s00.pkl"
    pk = open(f, "rb").read() if os.path.exists(f) else b""
    return p.returncode, p.stdout[-4000:], p.stderr[-4000:], pk


@app.local_entrypoint()
def main():
    rc, out, err, pk = run.remote()
    print("rc", rc); print(out); print(err[-2000:])
    if pk:
        (ROOT / "runs/n20_consistency/gpu_t4_ionosphere_n20_d8_e1p0_bw0p15__s00.pkl").write_bytes(pk)
        print("saved", len(pk), "bytes")
