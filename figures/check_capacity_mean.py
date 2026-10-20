import numpy as np, itertools
rng=np.random.default_rng(0)
I=np.eye(2); X=np.array([[0,1],[1,0]]); Y=np.array([[0,-1j],[1j,0]]); Zm=np.diag([1.,-1.])
def paulis(n):
    P=[];W=[]
    for t in itertools.product(range(4),repeat=n):
        m=np.array([[1.]])
        for k in t: m=np.kron(m,[I,X,Y,Zm][k])
        P.append(m); W.append(sum(k!=0 for k in t))
    return np.array(P),np.array(W)
def states(xs,n,depth=8,eps=1.0,c=0.3):
    d=2**n; bits=((np.arange(d)[:,None]>>np.arange(n)[::-1])&1); z=1-2*bits   # Z eigenvalues
    Hn=np.array([[1.]]); h=np.array([[1,1],[1,-1]])/np.sqrt(2)
    for _ in range(n): Hn=np.kron(Hn,h)
    out=[]
    for x in xs:
        ph=c*(z*x).sum(1)+eps*c*c*sum(x[i]*x[j]*z[:,i]*z[:,j] for i in range(n) for j in range(i+1,n))
        D=np.exp(-1j*ph); psi=np.zeros(d,complex); psi[0]=1
        for _ in range(depth): psi=D*(Hn@psi)
        out.append(psi)
    return np.array(out)
for n in (4,5,6):
    P,W=paulis(n); M=1500 if n<6 else 600
    S=states(rng.uniform(0,2*np.pi,(M,n)),n)
    E=np.real(np.einsum('si,pij,sj->sp',S.conj(),P,S))/np.sqrt(2**n)   # psi_i(x)
    gam=(E**2).mean(0); mu2=E.mean(0)**2; m=W>0
    ratio=mu2[m]/np.maximum(gam[m],1e-30)
    print(f"n={n}: sum gamma (I haric)={gam[m].sum():.3f} | ortalamadan gelen pay (agirlikli)={mu2[m].sum()/gam[m].sum():.3f} | terimlerin %{100*(ratio<0.1).mean():.0f}'inde ortalama payi <%10 | gamma_I={gam[~m][0]:.4f} (2^-n={2**-n:.4f})")
