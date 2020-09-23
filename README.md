# anomaly_with_experts

Install anaconda or miniconda https://docs.conda.io/en/latest/miniconda.html

git clone --recursive https://github.com/RaisaDZ/anomaly_with_experts.git anomaly_with_experts

cd anomaly_with_experts

Option --recursive is important as the repository contains the submodule https://github.com/numenta/NAB

Create conda environment:

conda create -n anomaly_with_experts python=3.7

To activate this environment, use

$ conda activate anomaly_with_experts

pip install -r requirements.txt --use-feature=2020-resolver

If you do not have Jupyter Notebook installed

pip install notebook

To launch it

jupyter notebook

After that you should be able to run main.ipynb which calculates the predictions of Fixed-share and Variable-share on NAB and outputs the results. 
Then you can run results_analysis.ipynb to analyse the losses and classification metrics and results_plots.ipynb to visualise the plots from the paper.
