# Real-time anomaly detection with superexperts

If you want to see the results of the paper there is no need to run anything. The visualisation of results is available [here](https://nbviewer.jupyter.org/github/RaisaDZ/anomaly_with_experts/blob/master/results_plots.ipynb) and the analysis of losses and classification metrics is available [here](https://nbviewer.jupyter.org/github/RaisaDZ/anomaly_with_experts/blob/master/results_analysis.ipynb).

If you want to run the project the easiest way is to use [Binder](https://mybinder.org). Click on [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/RaisaDZ/anomaly_with_experts/master).

On Binder you need to first run `main.ipynb` to calculate the predictions of Fixed-share and Variable-share and output the results (this will take around 50 minutes on Binder). After that you can run `results_analysis.ipynb` and `results_plots.ipynb`.

If you want to run the project locally follow the installation instructions below.

## Installation

Install anaconda or miniconda https://docs.conda.io/en/latest/miniconda.html.

Clone the repository (note that the flag `--recursive` is important as the repository contains the submodule [NAB](https://github.com/numenta/NAB)):
```bash
git clone --recursive https://github.com/RaisaDZ/anomaly_with_experts.git anomaly_with_experts
cd anomaly_with_experts
```

Create a conda environment:

```bash
conda create -n anomaly_with_experts python=3.7
```

Activate the environment:

```bash
conda activate anomaly_with_experts
```

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the requirements:
```bash
pip install -r requirements.txt --use-feature=2020-resolver
```

If you do not have Jupyter Notebook installed:
```bash
pip install notebook
```

To launch it:
```bash
jupyter notebook
```

After that you should be able to run `main.ipynb` which calculates the predictions of Fixed-share and Variable-share on NAB and outputs the results. 
Then you can run `results_analysis.ipynb` to analyse the losses and classification metrics and `results_plots.ipynb` to visualise the plots from the paper.
