Source code for BROIL:

#First install all dependencies via conda
conda env create --file broil.yml


## Machine Replacement Experiment:

#Generate plot of efficient frontier
conda activate cvarirl
python machine_replacement_experimentv2_efficient_frontier.py

#generate plot of action probs
conda activate cvarirl
python machine_replacement_experiment_v2_action_probs.py

#generate histogram plot of return dists
conda activate cvarirl
python machine_replacement_experiment_v2_plot_return_dists.py


## Ambiguous Demo Experiment:
To reproduce the ambiguous demonstration experiment run the following ipython notebook

conda activate cvarirl
jupyter notebook AmbiguousDemos.ipynb
