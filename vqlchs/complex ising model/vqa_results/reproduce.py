from pathlib import Path
import json
import numpy as np
from scipy.linalg import expm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out=Path(__file__).resolve().parent
n=6; times=np.arange(50)*0.1
I=np.eye(2); X=np.array([[0,1],[1,0]],complex); Z=np.diag([1,-1])
def kron(ops):
    a=np.array([[1]],complex)
    for op in ops: a=np.kron(a,op)
    return a
xs=[kron([X if k==j else I for k in range(n)]) for j in range(n)]
zs=[kron([Z if k==j else I for k in range(n)]) for j in range(n)]
obs=sum(zs)/n
def pqc(theta):
    psi=np.zeros(64,complex); psi[0]=1
    ix=0
    for layer in range(5):
        for j in range(n):
            for pauli in [xs[j],zs[j],xs[j]]:
                angle=theta[ix];ix+=1
                psi=np.cos(angle/2)*psi-1j*np.sin(angle/2)*(pauli@psi)
        for j in range(n-1):
            bits=np.arange(64)
            psi[((bits>>(n-1-j))&1).astype(bool)&((bits>>(n-2-j))&1).astype(bool)] *= -1
    return psi
fig,axes=plt.subplots(2,3,figsize=(9.3,4.8),sharex='col',gridspec_kw={'height_ratios':[2,1]})
stats={}
for col,(tag,gi) in enumerate([('05',.5),('1',1),('2',2)]):
    params=np.loadtxt(out/f'param_{tag}_layer_5.txt')
    vqs=np.array([np.vdot(s,obs@s).real for s in map(pqc,params[:50])])
    archived=np.loadtxt(out/f'ising_gi_{tag}.csv',delimiter=',',skiprows=1)[:,2]
    exact_archive=np.loadtxt(out/f'ising_gi_{tag}.csv',delimiter=',',skiprows=1)[:,1]
    H=-sum(zs[j]@zs[j+1] for j in range(n-1))-(2+1j*gi)*sum(xs)
    E=expm(-.1j*H); psi=np.zeros(64,complex);psi[0]=1;exact=[]
    for t in times:
        exact.append(np.vdot(psi,obs@psi).real)
        psi=E@psi;psi/=np.linalg.norm(psi)
    exact=np.array(exact)
    stats[tag]={'archived_vqs_shape':list(archived.shape),'parameter_reconstruction_difference':float(np.max(abs(archived[:50]-vqs))), 'exact_solver_difference':float(np.max(abs(exact-exact_archive))), 'max_observable_error':float(np.max(abs(vqs-exact))), 'rms_observable_error':float(np.sqrt(np.mean((vqs-exact)**2)))}
    axes[0,col].plot(times,exact_archive,color='black',lw=1.5,label='Direct evolution')
    axes[0,col].plot(times,vqs,color='#1565c0',ls='--',lw=1.4,label='Noiseless PQC')
    axes[0,col].set_title(rf'$g_{{\mathrm{{i}}}}={gi:g}$')
    axes[0,col].set_ylim(-.7,1.05)
    axes[1,col].plot(times,abs(vqs-exact_archive),color='#b24b24',lw=1.3)
    axes[1,col].ticklabel_format(axis='y',style='sci',scilimits=(0,0),useMathText=True)
    axes[1,col].set_xlabel(r'$t$')
    for ax in axes[:,col]:
        ax.spines[['top','right']].set_visible(False)
        ax.grid(alpha=.15)
axes[0,0].set_ylabel(r'$\langle S_z\rangle$')
axes[1,0].set_ylabel('Absolute difference')
axes[0,0].legend(fontsize=8,frameon=False)
fig.tight_layout()
fig.savefig(out/'reproduced_comparison.pdf',bbox_inches='tight')
fig.savefig(out/'reproduced_comparison.png',dpi=170,bbox_inches='tight')
(out/'reproduction_verification.json').write_text(json.dumps(stats,indent=2),encoding='utf-8')
print(json.dumps(stats,indent=2))
