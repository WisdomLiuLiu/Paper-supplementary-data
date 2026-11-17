"""One row, two line plots; paper style; second-order, one-step data only."""
from pathlib import Path
import csv
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator, MaxNLocator, ScalarFormatter
import scienceplots

HERE = Path(__file__).resolve().parent
OUT = HERE / 'figures'
OUT.mkdir(parents=True,exist_ok=True)
plt.style.use('science')
mpl.rcParams.update({
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{newtxtext,newtxmath}',
    'font.size': 19, 'axes.labelsize': 23,
    'xtick.labelsize': 20, 'ytick.labelsize': 20,
    'legend.fontsize': 16, 'axes.linewidth': 1.2,
    'lines.linewidth': 1.7, 'savefig.bbox':'tight',
    'savefig.pad_inches':0.08,
})
with (HERE/'pauli_order_comparison.csv').open(encoding='utf-8-sig') as f:
    data = [r for r in csv.DictReader(f) if int(r['order'])==2 and int(r['repetitions'])==1]
ns = [2,4,6,8,10]
index = {int(r['n']):r for r in data}
colors = mpl.rcParams['axes.prop_cycle'].by_key()['color']
fig,axes=plt.subplots(1,2,figsize=(297/25.4,4.7))

# Color identifies gate arity; line style identifies before/after.
for metric,color,marker,arity in [('single',colors[0],'o','1q'),('double',colors[2],'s','2q')]:
    for prefix,style,label in [('original','-','Before'),('paper_reverse','--','After')]:
        values=[int(index[n][prefix+'_'+metric]) for n in ns]
        axes[0].plot(ns,values,color=color,linestyle=style,marker=marker,markersize=6,
                     markerfacecolor=color if prefix=='original' else 'white',
                     markeredgewidth=1.3,label=label+' ('+arity+')')
axes[0].set_ylabel('Gate count')
axes[0].set_ylim(0,7400)
axes[0].yaxis.set_major_locator(MaxNLocator(nbins=4,integer=True))
axes[0].legend(loc='upper left',frameon=False,handlelength=2.0,labelspacing=.35)

values=[float(index[n]['overlap_complex_difference']) for n in ns]
axes[1].plot(ns,values,color=colors[0],marker='o',markersize=6,
             markerfacecolor='white',markeredgewidth=1.3)
axes[1].set_ylim(0,.023)
axes[1].set_yticks([0,.005,.010,.015,.020])
formatter=ScalarFormatter(useMathText=True)
formatter.set_powerlimits((-2,-2))
axes[1].yaxis.set_major_formatter(formatter)
axes[1].set_ylabel(r'Overlap deviation $\epsilon_X$')
axes[1].text(.96,.94,'Second-order Trotter',ha='right',va='top',
             transform=axes[1].transAxes,fontsize=16)
for ax,letter in zip(axes,['a','b']):
    ax.set_xticks(ns)
    ax.set_xlim(1.6,10.4)
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel(r'System qubits $n$')
    ax.tick_params(axis='both',pad=6)
    ax.text(-.19,1.04,rf'\textbf{{({letter})}}',transform=ax.transAxes,fontsize=25)
fig.subplots_adjust(left=.10,right=.985,bottom=.19,top=.88,wspace=.38)
for extension in ('pdf','png','svg'):
    fig.savefig(OUT/f'hn_second_order_scalability.{extension}',dpi=240)
plt.close(fig)
with (OUT/'plotted_data.csv').open('w',newline='',encoding='utf-8-sig') as f:
    writer=csv.DictWriter(f,fieldnames=['n','before_single','after_single','before_double','after_double','epsilon_X'])
    writer.writeheader()
    for n in ns:
        row=index[n]
        writer.writerow(dict(n=n,before_single=row['original_single'],
            after_single=row['paper_reverse_single'],before_double=row['original_double'],
            after_double=row['paper_reverse_double'],epsilon_X=row['overlap_complex_difference']))
print(OUT)
