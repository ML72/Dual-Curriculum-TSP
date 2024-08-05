import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches



# Define constants, modify as necessary to obtain different training gap plots
BASE_DIR = 'results/tsp'
RESULT_FILENAME = 'train_gap_results.png' # 'train_gap_prelim.png'
DATASETS = {
    'tsp_unif50_test_seed1234': "Uniform",
    'tsp_gmm50_test_seed1234': "Gaussian Mixture",
    'tsp_poly50_test_seed1234': "Polygon",
    'tsp_diag50_test_seed1234': "Diagonal"
}
MODELS = {
    'baseline_unif': 'Uniform (Baseline)',
    'hac_fullbatch': 'HAC 100% (Baseline)',
    'hac_halfbatch': 'HAC 50% (Baseline)',
    'hac_ewc_curriculum': 'Double Curriculum'
}
COLORS = {
    'baseline_unif': 'purple',
    'hac_fullbatch': 'red',
    'hac_halfbatch': 'green',
    'hac_ewc_curriculum': 'blue'
}
TRIALS = 5
FIELD_NAME = 'Gap_Rel_Avg'

# Utility function for reading a json file
def read_json_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# Utility function for plotting a subplot of a certain graph
def subplot_embedding(subplot, dataset, title):
    # Collect data
    data = {}
    for model in MODELS.keys():
        for trial in range(1, TRIALS+1):
            file_path = f"{BASE_DIR}/{dataset}/{model}_{trial}-epoch_data.json"
            data[f"{model}_{trial}"] = read_json_file(file_path)

    epochs = data[list(data.keys())[0]].keys()
    epochs = [int(epoch) for epoch in epochs]
    epochs.sort()

    # Plot data
    for model in MODELS.keys():
        values = np.zeros((len(epochs), TRIALS))
        for i in range(len(epochs)):
            for j in range(TRIALS):
                values[i, j] = float(data[f"{model}_{j+1}"][str(epochs[i])][FIELD_NAME])
        means = values.mean(axis=1)
        stds = values.std(axis=1)
        subplot.plot(epochs, means, label=MODELS[model], color=COLORS[model])
        if not np.allclose(stds, 0):
            subplot.fill_between(epochs, means-stds, means+stds, color=COLORS[model], alpha=0.2)

    subplot.set_title(title)
    subplot.margins(x=0)
    subplot.xlim = (0, max(epochs))
    subplot.set_xlabel('Epoch')
    subplot.set_ylabel('Gap Relative to Concorde')

# Draw plot visualizations
fig, axs = plt.subplots(1, len(DATASETS), figsize=(4 * len(DATASETS), 4))
plt.subplots_adjust(hspace=0.1)

for i, dataset in enumerate(DATASETS.keys()):
    print(f"Plotting from {DATASETS[dataset]}")
    subplot_embedding(
        axs[i],
        dataset,
        f"Gap on {DATASETS[dataset]}"
    )

# Add custom legend for plot colors
custom_handles = [
    mpatches.Patch(color=COLORS[model], label=name) for model, name in MODELS.items()
]

fig.legend(handles=custom_handles, loc='upper center', ncol=4)
plt.tight_layout(rect=[0.025, 0.015, 0.95, 0.94])

if not os.path.exists("results/plots"):
    os.makedirs("results/plots")
plt.savefig(f"results/plots/{RESULT_FILENAME}", format='png')
plt.close()
