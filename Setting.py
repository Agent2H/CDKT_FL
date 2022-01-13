"FIX PROGRAM SETTINGS"
import random
import numpy as np
import torch

READ_DATASET = True  # True or False => Set False to generate dataset.
Full_model =False # True: Using only outcomes of full models
Rep_Full= False # True: Using the embedding features of representation + outcomes of full models # False: Using only embedding features of rep
Subset = False# True: Using fraction of users
Same_model= True  # True: Using homogeneous model, False: using heterogeneous model
Ensemble = False #Ensemble alpha=same with avg method, beta=0.05
Moving_Average = False
Accelerated = False
Tune_output= True
PLOT_PATH = "./figs/"
RS_PATH = "./results/"
FIG_PATH = "./results_fig/"
# FIG_PATH = "./results_KSC2021/"
PLOT_PATH_FIG="./figs/"
# PLOT_PATH_FIG="./KSC_Figs/"

#Dataset selection
DATASETS= ["mnist","fmnist","Cifar10","Cifar100"]
DATASET = DATASETS[3]
#Algorithm selection
RUNNING_ALGS = ["fedavg","CDKT"]
RUNNING_ALG = RUNNING_ALGS[0]
#Metric selection
CDKT_metrics = ["KL","Norm2","JSD"]
Global_CDKT_metric = CDKT_metrics[0]   # Global distance metric
Local_CDKT_metric = CDKT_metrics[1]    # Local distance metric

#Algorithm Parameter
alpha =0.02# trade-off parameter of local training loss
beta =0.15# trade-off parameter of global distillation loss (Mnist:rep+full loss 0.2)
gamma=0.5


local_learning_rate = 0.03
global_learning_rate = 0.07
global_generalized_epochs = 2
LOCAL_EPOCH = 2
NUM_GLOBAL_ITERS = 100 # Number of global rounds








SEED = 1
random.seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
### Agorithm Parameters ###
if Subset:
 N_clients = 50
 Frac_users = 0.2  #20% of users will be selected
else:
 N_clients = 10
 Frac_users = 1.  #All users will be selected
K_Levels =1

if(DATASET == "Cifar100"):
    # NUM_GLOBAL_ITERS = 100
    NUMBER_LABEL=100
else:
    NUMBER_LABEL = 10





rs_file_path = "{}_{}_I{}_s{}_f{}_a{}_b{}_RF{}_SS{}_acc{}_gm{}_lm{}.h5".format(RUNNING_ALG, DATASET, NUM_GLOBAL_ITERS,
                                Same_model, Full_model,alpha,beta,Rep_Full,Subset,Accelerated,Global_CDKT_metric,Local_CDKT_metric )
rs_file_path = FIG_PATH + rs_file_path
PLOT_PATH += DATASET+'_'
print("Result Path ", rs_file_path)

# complex_file_path = "{}_{}_I{}_time_.h5".format(DATASET, RUNNING_ALG, NUM_GLOBAL_ITERS)





