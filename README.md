# Bayesian Robust Optmization for Imitatation Learning 

Daniel Brown, Scott Niekum, Marek Petrik

### Read the paper: [arXiv Link](https://arxiv.org/abs/2007.12315).

## Source Code

This repository contains code to reproduce the experiments.

If you find this repository is useful in your research, please cite our paper:
```
@InProceedings{brown2020broil,
  title = {Bayesian Robust Optimization for Imitation Learning},
  author = {Brown, Daniel S. and Niekum, Scott and Petrik, Marek},
  booktitle={Advances in neural information processing systems (NeurIPS)},
  year={2020}
}

```


### First install all dependencies via conda
```
conda env create --file broil.yml
```

Before running any of the commands below, first activate the conda environment
```
conda activate cvarirl
```

## Machine Replacement Experiment:

### Generate plot of efficient frontier
```
python machine_replacement_experimentv2_efficient_frontier.py
```
### generate plot of action probs
```
python machine_replacement_experiment_v2_action_probs.py
```

### generate histogram plot of return dists
```
python machine_replacement_experiment_v2_plot_return_dists.py
```

## Ambiguous Demo Experiment:
To reproduce the ambiguous demonstration experiment run the following ipython notebook

```
jupyter notebook AmbiguousDemos.ipynb
```
