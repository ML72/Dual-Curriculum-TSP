import os
import pickle
import numpy as np
import matplotlib.pyplot as plt



# Define constants
BASE_DIR = 'data/tsp'
RESULT_FILENAME = 'distribution_visualization.png'
DISTRIBUTIONS = {
    'Uniform': 'tsp_unif50_test_seed1234',
    'Gaussian Mixture': 'tsp_gmm50_test_seed1234',
    'Polygon': 'tsp_poly50_test_seed1234',
    'Diagonal': 'tsp_diag50_test_seed1234'
}
NUM_SAMPLES = 3
IDX_LIST = {
    'Uniform': [13, 17, 19],
    'Gaussian Mixture': [1500, 3500, 7500],
    'Polygon': [0, 2000, 4000],
    'Diagonal': [2000, 5000, 9000]
}

# Utility function for reading a pickle file
def read_pickle_file(file_path):
    with open(file_path, 'rb') as f:
        pickle_data = pickle.load(f)
    return np.array(pickle_data)

# Load in data
data = {}
for distribution in DISTRIBUTIONS.keys():
    file_path = f"{BASE_DIR}/{DISTRIBUTIONS[distribution]}.pkl"
    data[distribution] = read_pickle_file(file_path)

# Utility function for plotting subplot
def subplot_embedding(subplot, graph, title):
    subplot.scatter(graph[:,0], graph[:,1], color='black')
    subplot.set_xticks([])
    subplot.set_yticks([])
    subplot.set_title(title)

# Plot it!
div = 2 * NUM_SAMPLES
fig, axs = plt.subplots(2, div, figsize=(div * 4, 2 * 4))
plt.subplots_adjust(hspace=0.1)
plt.tight_layout(rect=[0.01, 0, 0.97, 0.99])

for j, distribution in enumerate(DISTRIBUTIONS.keys()):
    for i in range(NUM_SAMPLES):
        idx = IDX_LIST[distribution][i]
        sum = j * NUM_SAMPLES + i
        subplot_embedding(axs[sum // div, sum % div], data[distribution][idx], f"{distribution} Example {i+1}")

if not os.path.exists("results/plots"):
    os.makedirs("results/plots")
plt.savefig(f"results/plots/{RESULT_FILENAME}", format='png')
plt.close()
