import matplotlib.pyplot as plt
import numpy as np

# Parse raw data
raw = """adamw 130m 1 3.5291428565979004
adamw 130m 2 3.4077868461608887
adamw 130m 4 3.3216564655303955
adamw 130m 8 3.260563611984253
nadamw 130m 1 3.5311388969421387
nadamw 130m 2 3.393378496170044
nadamw 130m 4 3.319291353225708
nadamw 130m 8 3.2486026287076416
mars 130m 1 3.536329746246338
mars 130m 2 3.394932270050049
mars 130m 4 3.320751428604126
mars 130m 8 3.2465016841888428
muon 130m 1 3.4626309871673584
muon 130m 2 3.369025468826294
muon 130m 4 3.2946722507476807
muon 130m 8 3.238452911376953
kron 130m 1 3.4907355308532715
kron 130m 2 3.388023376464844
kron 130m 4 3.3028366565704346
kron 130m 8 3.238511800765991
soap 130m 1 3.4821648597717285
soap 130m 2 3.374239921569824
soap 130m 4 3.294482469558716
soap 130m 8 3.2374308109283447
adamw 520m 1 3.109556198120117
adamw 520m 2 3.0231785774230957
adamw 520m 4 2.958468198776245
adamw 520m 8 2.912761926651001
nadamw 520m 1 3.0993051528930664
nadamw 520m 2 3.0127241611480713
nadamw 520m 4 2.954136371612549
nadamw 520m 8 2.9072420597076416
mars 520m 1 3.098572254180908
mars 520m 2 3.0136630536125732
mars 520m 4 2.952986717224121
mars 520m 8 2.905677080154419
muon 520m 1 3.070978879928589
muon 520m 2 3.001875162124634
muon 520m 4 2.9442315101623535
muon 520m 8 2.900200366973877
kron 520m 1 3.0836100578308105
kron 520m 2 3.008941411972046
kron 520m 4 2.946164131164551
kron 520m 8 2.90018892288208
soap 520m 1 3.0788798332214355
soap 520m 2 3.0036733150482178
soap 520m 4 2.944326400756836
soap 520m 8 2.8988614082336426"""

# Build data dictionary
data = {}
for line in raw.splitlines():
    opt, size, chin, loss = line.split()
    data.setdefault(size, {}).setdefault(opt, {})[int(chin)] = float(loss)

# Configuration
optimizers = ['muon', 'soap', 'kron', 'nadamw', 'mars']
settings = [('130m', 1), ('130m', 8), ('520m', 1), ('520m', 8)]
color_map = {
    'mars': '#1f77b4', 'muon': '#ff7f0e', 'kron': '#8c564b',
    'nadamw': '#9467bd', 'soap': '#bcbd22'
}

# Create 2x2 subplots with independent x-limits and full bar height
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()


speedups = {
    '130m': {
        1: {'nadamw':0.99, 'mars':0.97, 'muon':1.42, 'kron':1.22, 'soap':1.27},
        8: {'nadamw':1.18, 'mars':1.21, 'muon':1.37, 'kron':1.37, 'soap':1.39}
    },
    '520m': {
        1: {'nadamw':1.07, 'mars':1.08, 'muon':1.33, 'kron':1.20, 'soap':1.25},
        8: {'nadamw':1.10, 'mars':1.13, 'muon':1.26, 'kron':1.26, 'soap':1.29}
    }
}
for idx, (ax, (size, chin)) in enumerate(zip(axes, settings)):
    base = data[size]['adamw'][chin]
    deltas = [base - data[size][opt][chin] for opt in optimizers]
    y = np.arange(len(optimizers))

    # Horizontal bars with height=1.0
    bars = ax.barh(y, deltas, color=[color_map[opt] for opt in optimizers], height=1.0)
    ax.set_title(f'{size} at {chin}× Chinchilla', fontsize=20)

    # Only show y-labels on left column subplots
    if idx % 2 == 0:
        ax.set_yticks(y)
        labels = ax.set_yticklabels(optimizers, fontsize=20)
        for label, opt in zip(labels, optimizers):
            label.set_color(color_map[opt])
    else:
        ax.set_yticks(y)
        ax.set_yticklabels([])
    # Annotate speedup on each bar, adjust for negative small bars at 130m 1x
    for bar, opt, speed in zip(bars, optimizers, [speedups[size][chin][o] for o in optimizers]):
        width = bar.get_width()
        # For 130m @1x and negative/small widths, shift annotation to fixed positive offset
        if idx == 0 and opt in ('mars', 'nadamw'):
            x_pos = 0.005
            ha = 'left'
        else:
            x_pos = width + (0.002 if width >= 0 else -0.002)
            ha = 'left' if width >= 0 else 'right'
        ax.text(x_pos, bar.get_y() + bar.get_height()/2,
                f'{speed:.2f}×', va='center', ha=ha, fontsize=20, color=color_map[opt])
    # Remove whitespace around bars
    ax.set_ylim(-0.5, len(optimizers) - 0.5)
    ax.margins(y=0)

    # Individual x-limits and 0.01 ticks
    deltas.append(0)
    if idx == 0:
        gap = 0.02
    else:
        gap = 0.01
    xmin = np.floor(min(deltas) * (1/gap)) / (1/gap)
    xmax = np.ceil(max(deltas) * (1/gap)) / (1/gap)
    if idx % 2 == 0:
        xmax += 0.01
    ax.set_xlim(xmin, xmax)
    ax.set_xticks(np.arange(xmin, xmax + gap, gap))
    ax.tick_params(axis='x', labelsize=18)
    # ax.grid(axis='x', linestyle='--', alpha=0.4)

# Legend and common label
handles = [plt.Line2D([0], [0], color=color_map[opt], lw=8) for opt in optimizers]
# fig.legend(handles, optimizers, title='Optimizer', loc='upper center', ncol=5, fontsize=14, title_fontsize=16)
fig.text(0.5, 0.01, 'Reduced Loss Over AdamW', ha='center', fontsize=20)

# Adjust spacing
# fig.subplots_adjust(left=0.1, right=0.95, bottom=0.08, top=0.9, wspace=0.3, hspace=0.4)
# plt.tight_layout()
plt.savefig('optimizer_loss_delta.pdf')
