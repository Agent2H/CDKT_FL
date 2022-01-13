CDKT Algorithm Implementation
======

This repository implements all experiments for knowledge transfer in federated learning


Running Instruction
=====

**Environment:** Python 3.8, pytorch

**Downloading dependencies**  
```
pip3 install -r requirements.txt  
```
To plot the figure, we use the result files from `./results_fig` by running CDKT_plot_summary.py

**Main File:** CDKT_main.py, **Setting File:** Setting.py, **Plot Summary Results:**  CDKT_plot_summary.py

 **PATH:** Output: `./results_fig`, Figures: `./figs`, Dataset: `./data`

**Modify parameters setting in `Setting.py` to run the code**

Please refer the file `Tuning results` to access different parameter settings of this  work.

--
Example:

CDKT + Mnist + RepFull + Subset of Users + Homogeneous Model: **RUNNING_ALGS**[1], **DATASETS**[0], **Full_model** = False, **Rep_Full** = True , **Subset** = True, **Same_model** = True

CDKT + Cifar-10 + Full + Fixed Users + Heterogeneous Model: **RUNNING_ALGS**[1], **DATASETS**[2], **Full_model** = True, **Rep_Full** = True , **Subse**t = False, **Same_model** = False

**To run the code, use the command in terminal**:
```
python CDKT_main.py
```



The Results are stored in `./results` and figures in `./figs`







