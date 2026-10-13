"""APS full-width 1x3: PQC depth, standard HSF, paired replacement error."""
import argparse
from pathlib import Path
import numpy as np
from experiment import ROOT
from xxz_core import read_json, file_hash
from figure_style import apply_aps_style, plt
from matplotlib.ticker import MaxNLocator, AutoMinorLocator, FormatStrFormatter


def plot(output):
    manifest=read_json(output/'manifest.json')
    if manifest['status']!='completed':
        raise ValueError('All five insertion jobs must complete before plotting')
    libraries=read_json(output/'inputs/selected_libraries.json')
    apply_aps_style()
    plt.rcParams.update({'font.size':9,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,
                         'legend.fontsize':8,'lines.linewidth':1.05,'lines.markersize':3.2,
                         'axes.linewidth':.65,'xtick.major.width':.6,'ytick.major.width':.6})
    figure,axes=plt.subplots(1,3,figsize=(7.0,2.35))
    figure.subplots_adjust(left=.063,right=.983,bottom=.165,top=.955,wspace=.30)
    colors=['black','#0072B2','#009E73','#E69F00','#D55E00']
    styles=['--','-','-.',(0,(4,1.5)),(0,(1.5,1))]
    markers=['o','s','^','D','v']
    handles=[];point_rows=[];curve_rows=[];max_error=0.;minimum_fidelity=1.;maximum_depth=1
    for index,(label,library) in enumerate(sorted(libraries.items(),key=lambda item:item[1]['shots'])):
        shots=library['shots'];legend='Infinite shots' if shots==0 else f'{shots} shots'
        blocks=library['blocks'];ks=sorted(map(int,blocks))
        depths=[blocks[str(k)]['depth'] for k in ks];fidelities=[blocks[str(k)]['hs_fidelity'] for k in ks]
        color=colors[index%len(colors)];style=styles[index%len(styles)];marker=markers[index%len(markers)]
        line,=axes[0].plot(ks,depths,color=color,ls=style,marker=marker,ms=3.0,label=legend,mew=.55)
        handles.append(line)
        axes[1].plot(ks,fidelities,color=color,ls=style,marker=marker,ms=3.0,mew=.55)
        data=np.load(output/f'shots_{label}'/'curves.npz')
        summary=read_json(output/f'shots_{label}'/'summary.json')
        if file_hash(output/f'shots_{label}'/'curves.npz') != summary['files_sha256']['curves.npz']:
            raise ValueError('Insertion curve checksum mismatch')
        axes[2].plot(data['time'],data['error_to_kraus'],color=color,ls=style,lw=1.05)
        max_error=max(max_error,float(np.max(abs(data['error_to_kraus']))))
        minimum_fidelity=min(minimum_fidelity,min(fidelities));maximum_depth=max(maximum_depth,max(depths))
        point_rows += [dict(shots=shots,s=k,depth=blocks[str(k)]['depth'],hs_fidelity=blocks[str(k)]['hs_fidelity']) for k in ks]
        curve_rows += [dict(shots=shots,time=float(t),imbalance=float(v),error_to_kraus=float(e))
                       for t,v,e in zip(data['time'],data['imbalance'],data['error_to_kraus'])]
    for ax in axes[:2]:
        ax.set_xlabel(r'Repetitions, $s$')
        ax.set_xlim(.5,manifest['specification']['cutoff']+.5)
        ax.set_xticks([1,5,10,15,20] if manifest['specification']['cutoff']==20 else ks)
    axes[0].set_ylabel('PQC depth');axes[0].set_ylim(0,maximum_depth+.7)
    axes[0].yaxis.set_major_locator(MaxNLocator(integer=True,nbins=5))
    axes[1].set_ylabel(r'HSF to $(U_0)^s$')
    lower=np.floor((minimum_fidelity-.001)*100)/100
    axes[1].set_ylim(lower,1.002)
    axes[1].set_yticks(np.arange(1.,lower-.0001,-.01)[::-1])
    axes[1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axes[1].axhline(.99,color='.6',ls=':',lw=.7)
    axes[2].set_xlabel(r'Time, $t$');axes[2].set_ylabel(r'Replace error, $\Delta I$')
    axes[2].set_xlim(0,float(data['time'][-1]));axes[2].set_xticks(np.linspace(0,data['time'][-1],5))
    axes[2].xaxis.set_major_formatter(FormatStrFormatter('%g'))
    limit=max(.025,np.ceil(max_error*1.06/.025)*.025)
    axes[2].set_ylim(-limit,limit);axes[2].yaxis.set_major_locator(MaxNLocator(nbins=5))
    axes[2].axhline(0,color='.6',ls=':',lw=.7)
    for index,ax in enumerate(axes):
        ax.xaxis.set_minor_locator(AutoMinorLocator(2));ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        panel=rf'\textbf{{({chr(97+index)})}}' if plt.rcParams['text.usetex'] else f'({chr(97+index)})'
        ax.text(-.19,1.0,panel,transform=ax.transAxes,ha='left',va='center',fontsize=10,fontweight='bold')
    axes[2].legend(handles,[h.get_label() for h in handles],loc='lower left',
                   ncol=1,frameon=False,handlelength=2.1,fontsize=8,labelspacing=.2,borderaxespad=.5)
    dest=output/'figures';dest.mkdir(exist_ok=True)
    stem=dest/'figure_shots_insert_full_width'
    for ext in ['pdf','png']:
        figure.savefig(stem.with_suffix('.'+ext),dpi=400)
    plt.close(figure)
    print(f'Saved full-width figure: {stem}.pdf/png',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'data/insert')
    plot(parser.parse_args().output)
