# Summary

This repository contains the source code along with the IPython notebooks to reproduce the results presented in the article _"Learning Block-sparse Partial Correlation Graphs"_.

# Dependencies
In order to properly run the provided code, please install the dependencies provided in the file `environment.yml` or `environment_versioned.yml`. The first file does not contain the versions of the required packages, whereas the latter does.
In particular, the former is best suited for cross-platform sharing (see [here](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#sharing-an-environment))

As per [conda documentation](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file), you can create the conda environment _BSPCG_ by running the following command in your terminal:
```shell
    conda env create -f environment.yml
```
or 
```shell
    conda env create -f environment_versioned.yml
```

__Errors.__
If you obtain _Pip subprocess error_ concerning the installation of `jaxlib`, please refer to [this](https://github.com/google/jax#installation).

Once the environment has been successfully created, if using VS Code, then type
```shell
    code .
```
and select the BSPCG environment from those available.
Otherwise, please activate it
```shell
    conda activate BSPCG
```
and run the provided IPython notebooks.
To deactivate the environment, simply run 
```shell
    conda deactivate
```

# Data
The data sets used in our experiments are given in the `data` folder.
Specifically:
* `1.0_dataset_{N_bar}_50.pkl` are pickle files containing 50 data sets for the settings $\bar{N}_{\mathcal{Y}} \in \{5,10,20,50,100,1000\}$;
* `1.0_gt_estimation.pkl` contains info related to the ground truth;
* `1.0_hyperparameters.pkl` contains the values of $\lambda$ and $\eta$ used in the results reported in Section 5 of the article;
* `1.1_performances_synth_experiments.pkl` contains the performances depicted in Fig.1 of the paper;
* `2.0_returns_2018_2019.pkl` and `2.0_returns_2020_2021.pkl` contain the linear returns of 17 industrial portfolios over 2018/19 and 2020/21, respectively;
* `2.0_sp_bef.pkl` and `2.0_sp_aft.pkl` contain the smoothed periodogram values for 2018/19 and 2020/21, respectively;
* `2.0_results_bef.pkl`, `2.0_results_aft.pkl`, `2.0_Rl2_ours_bef.pkl`, and `2.0_Rl2_ours_aft.pkl` contain the results underlying Fig.2 in the paper.

# Source code
The JAX implementations of our methods and the TSGLASSO baseline are provided in the `src\models` folder, in particular in `nonconvex.py` and `convex.py` files, respectively.

In addition, `src\utils.py` contains useful functions called throught the code, while `src\metrics.py` provides several metrics for evaluating the learned block-sparse partial correlation graph.

# IPython notebooks
To reproduce the results presented in the aritcle, three IPython notebooks are provided.

* `1.0_synthetic_experiments_test_example.ipynb`: This notebook demonstrates how to replicate the results from Section 5 of the paper. To ensure manageable computational time, by default the models are applied to $3$ out of $50$ data sets, using the setting $\bar{N}_{\mathcal{Y}}=10$. However, the code can be easily adjusted to reproduce other settings described in the article, as long as the hyper-parameters are modified according to the information provided in the supplemental material. Lastly, the comparison between the obtained results and those reported in the paper is provided.
* `1.1_synthetic_experiments_figures.ipynb`: This notebook loads and plots the results presented in Section 5 of the paper.
* `2.0_real-world_case_study.ipynb`: This notebook allows to reproduce the financial time series case study presented in Section 6 of the paper.
In addition, the results obtained from the mere application of the _Naive_ estimator are shown as well.