"""Simulator dogrulamasi: `qk_core` vs PennyLane `default.qubit`.

Calistir (PennyLane ayri venv'de):  venvs/qml/Scripts/python.exe src/pennylane_esdegerlik.py

Defterler PennyLane kullanmaz: durum vektoru dogrudan kurulur (`encode`) ve ayni flip
desenini paylasan Pauli'ler tek Walsh-Hadamard donusumuyle okunur (`pauli_expect_fast`).
Bu betik ikisinin de standart devre simulasyonuyla AYNI sayilari verdigini gosterir:

  devre (her katman, `depth` kez):  H^{(x)n}  ->  RZ(2 bw x_i) her i  ->  IsingZZ(2 ent bw^2 x_i x_j) her i<j
  kubit 0 = en anlamli bit (PennyLane ile ayni). Pauli indeksinde 4-tabanli k. hane (0=I 1=X 2=Y 3=Z)
  z'nin k. bitine, yani kubit n-1-k'ya etki eder.
"""
import os
import numpy as np
import pennylane as qml

# qk_core'un bekledigi ortam (defterin 0. hucresi, CPU yolu, float64)
xp, GPU = np, False
CDT, FDT = np.complex128, np.float64
to_xp = lambda a, dtype=None: np.asarray(a, dtype=dtype)
to_np = np.asarray
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'qk_core.py'),
          encoding='utf-8').read())

TOL = 1e-10
PAULI = {1: qml.PauliX, 2: qml.PauliY, 3: qml.PauliZ}


def pl_circuit(n, depth, ent, bw):
    dev = qml.device('default.qubit', wires=n)

    def layers(x):
        for _ in range(depth):
            for w in range(n):
                qml.Hadamard(w)
            for i in range(n):
                qml.RZ(2 * bw * x[i], wires=i)
            for i in range(n):
                for j in range(i + 1, n):
                    qml.IsingZZ(2 * ent * bw ** 2 * x[i] * x[j], wires=[i, j])

    @qml.qnode(dev)
    def state(x):
        layers(x)
        return qml.state()

    def expvals(x, idx):
        obs = []
        for p in idx:
            terms = [PAULI[(int(p) // 4 ** k) % 4](n - 1 - k)
                     for k in range(n) if (int(p) // 4 ** k) % 4]
            obs.append(qml.prod(*terms) if len(terms) > 1 else terms[0])

        @qml.qnode(dev)
        def f(x):
            layers(x)
            return [qml.expval(o) for o in obs]
        return np.array(f(x))

    return state, expvals


def main():
    rng = np.random.default_rng(20260916)
    worst_s = worst_p = 0.0
    for n, H in ((4, 4), (6, 2), (6, 4), (8, 2)):
        lay = pool_layout(n, H)
        idx = lay[2][1:]                                   # kimlik (indeks 0) haric
        for ent, bw in ((1.0, 0.15), (0.5, 0.25), (1.0, 0.3)):
            X = rng.uniform(0, 2 * np.pi, (3, n))
            st, ev = pl_circuit(n, 8, ent, bw)
            psi = encode(X, n, 8, ent, bw)
            E = pauli_expect_fast(psi, n, lay)[:, 1:]
            for m in range(len(X)):
                ref = np.asarray(st(X[m]))
                ds = np.abs(psi[m] - ref).max()             # global faz yok: kapilar birebir
                dp = np.abs(E[m] - ev(X[m], idx)).max()
                assert ds < TOL and dp < TOL, (n, H, ent, bw, ds, dp)
                worst_s, worst_p = max(worst_s, ds), max(worst_p, dp)
        print(f'n={n} H<={H}: {len(idx)} Pauli, durum + beklenen degerler PennyLane ile ayni')
    print(f'\nEN KOTU SAPMA  durum {worst_s:.1e}   Pauli {worst_p:.1e}   (esik {TOL:.0e})')
    print(f'PennyLane {qml.__version__} · numpy {np.__version__}')


if __name__ == '__main__':
    main()
