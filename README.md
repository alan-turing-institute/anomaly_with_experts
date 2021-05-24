# Anomaly detection with superexperts under delayed feedback

The visualisation of results is available [here](https://nbviewer.jupyter.org/github/alan-turing-institute/anomaly_with_experts/blob/master/results_plots.ipynb) and the analysis of losses and classification metrics is available [here](https://nbviewer.jupyter.org/github/alan-turing-institute/anomaly_with_experts/blob/master/results_analysis.ipynb). To run the project locally follow the installation instructions below.

## Installation

Install anaconda or miniconda https://docs.conda.io/en/latest/miniconda.html.

Clone the repository (note that the flag `--recursive` is important as the repository contains the submodule [NAB](https://github.com/numenta/NAB)):
```bash
git clone --recursive https://github.com/alan-turing-institute/anomaly_with_experts.git anomaly_with_experts
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

After that, you should first run `calculate_predictions.ipynb` which calculates the predictions of Fixed-share and Variable-share on NAB and outputs the results. 
Then you can run `results_analysis.ipynb` to analyse the losses and classification metrics and `results_plots.ipynb` to visualise the plots from the paper. The main functions of the implementation are available in folder `anomaly_delays`.
