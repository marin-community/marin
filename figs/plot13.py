import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Set global font size for plots
plt.rcParams.update({'font.size': 20})

# Load the dataset
import pickle
losses = {'mars': [3.100747585296631, 3.110823154449463, 'NaN', 3.130775451660156, 3.1032514572143555, 3.127875804901123, 3.105170726776123, 3.112410068511963, 17.2548828125, 3.1078317165374756, 3.1070127487182617, 3.108074903488159, 3.098572254180908, 3.1018712520599365, 3.0997989177703857, 3.103288412094116, 3.1134979724884033, 3.100747585296631, 3.101029872894287, 3.100539445877075, 3.101011037826538, 3.1016435623168945, 3.1301145553588867, 3.2235424518585205], 'muon': [3.073499917984009, 3.0952091217041016, 3.146660089492798, 7.88614559173584, 7.899570941925049, 3.133925437927246, 3.09061598777771, 3.09755539894104, 3.0799994468688965, 3.0726981163024902, 3.0726425647735596, 3.073541164398194, 3.0731754302978516, 3.0899195671081543, 3.0800628662109375, 3.0735838413238525, 3.0730509757995605, 3.0728700160980225, 3.072497606277466, 3.072497606277466, 3.072497606277466, 3.072497606277466, 3.0715272426605225, 3.070978879928589, 3.134270429611206, 3.1011009216308594, 3.085721969604492, 3.0763001441955566, 3.0764167308807373, 3.071852207183838, 3.080350399017334, 3.114563226699829, 3.187739372253418], 'soape': [3.0788798332214355, 3.0792086124420166, 5.761875152587891, 6.105731964111328, 3.5269508361816406, 3.1179521083831787, 4.163177013397217, 3.080887794494629, 4.6299567222595215, 4.315766334533691, 3.0965120792388916, 3.0902762413024902, 3.084770679473877, 3.080277681350708, 3.0823607444763184, 3.082611083984375, 5.395188808441162, 4.391687393188477, 3.08154821395874, 3.080942153930664, 3.0850071907043457, 4.215353012084961]}

color_map = {
    'mars': '#1f77b4',    # blue
    'muon': '#ff7f0e',    # orange
    'lion': '#2ca02c',    # green
    'adamw': '#d62728',   # red
    'nadamw': '#9467bd',  # purple
    'kron': '#8c564b',    # brown
    'scion': '#e377c2',   # pink
    'cautious': '#7f7f7f', # gray
    'soape': '#bcbd22',    # yellow-green
    'sophia': '#17becf',  # cyan
    'mini': '#aec7e8',    # light blue
}

correct_name = {
    'adamw': 'AdamW',
    'lion': 'Lion',
    'mini': 'Adam-Mini',
    'scion': 'Scion',
    'cautious': 'Cautious',
    'mars': 'Mars',
    'nadamw': 'NAdamW',
    'muon': 'Muon',
    'soape': 'Soap',
    'kron': 'Kron',
}


line_style = {
    'adamw': '--',
    'mars': '--',
    'nadamw': '--',
    'muon': '-',
    'soape': '-',
    'kron': '-',
    'lion': '--',
    'mini': '--',
    'scion': '-',
    'cautious': '--',
}

# Define the scaling model
def scaling_model(D, alpha, B, beta):
    return alpha * D**(-B) + beta

# Fit AdamW baseline parameters for each model size
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Ensure user-preferred font size
plt.rcParams['font.size'] = 18

# Filtered loss data
loss1 = losses['muon']
loss2 = losses['mars']
loss3 = losses['soape']

def clean_loss(loss_list, ths = 3.12):
    return [loss for loss in loss_list if type(loss) == float and loss < ths]

loss1 = clean_loss(loss1)
loss2 = clean_loss(loss2)
loss3 = clean_loss(loss3)

# … after you’ve cleaned loss1, loss2, loss3 …

# pack them into a list, and build matching labels & colors
datasets = [loss1, loss2, loss3]
labels   = [correct_name[k] for k in ['muon','mars','soape']]
colors   = [color_map[k]     for k in ['muon','mars','soape']]

plt.figure()
plt.hist(
    datasets,
    bins=10,
    label=labels,
    color=colors,
    alpha=0.6,          # make bars semi-transparent so you can see overlaps
    histtype='barstacked'      # other options: 'barstacked', 'step', 'stepfilled'
)

plt.yticks([0, 5, 10, 15], fontsize=18)

plt.xlabel('Loss', fontsize=18)
plt.ylabel('Number of runs', fontsize=18)
plt.legend(fontsize=18)
plt.tight_layout()
plt.savefig('loss_hist.pdf', bbox_inches='tight')

# # Create x-axis range
# x = np.linspace(min(min(loss1), min(loss2), min(loss3)), max(max(loss1), max(loss2), max(loss3)), 200)


# # Plot fitted distributions
# plt.figure()
# plt.hist(loss1, bins=10, density=False, label=correct_name['muon'])
# plt.hist(loss2, bins=10, density=False, label=correct_name['mars'])
# plt.hist(loss3, bins=10, density=False, label=correct_name['soape'])
# # plt.title('Fitted Normal Distributions of Loss')
# plt.xlabel('Loss')
# plt.ylabel('Number of runs')
# plt.legend()
# plt.tight_layout()
# plt.savefig('loss_hist.pdf', bbox_inches='tight')
