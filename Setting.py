"FIX PROGRAM SETTINGS"
import random
import numpy as np
import torch

#Algorithm selection
RUNNING_ALGS = ["fedavg","FedProx","CDKT_Rep","CDKT_RepFull","CDKT_Full","fedkc","MOON","Scaffold","FedDyn"]
RUNNING_ALG = RUNNING_ALGS[6]
READ_DATASET = True  # True or False => Set False to generate dataset.
Full_model = True  # True: Using only outcomes of full models
Rep_Full = True  #
if RUNNING_ALG == "CDKT_Rep":
    Full_model =False # True: Using only outcomes of full models
    Rep_Full= False # True: Using the embedding features of representation + outcomes of full models # False: Using only embedding features of rep
elif RUNNING_ALG == "CDKT_RepFull":
    Full_model =False # True: Using only outcomes of full models
    Rep_Full= True # True: Using the embedding features of representation + outcomes of full models # False: Using only embedding features of rep
elif RUNNING_ALG == "CDKT_Full":
    Full_model =True # True: Using only outcomes of full models
    Rep_Full= True # True: Using the embedding features of representation + outcomes of full models # False: Using only embedding features of rep
Subset = True# True: Using fraction of users
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
DATASET = DATASETS[2]

#Metric selection
CDKT_metrics = ["KL","Norm2","JSD","Con"]
Global_CDKT_metric = CDKT_metrics[0]   # Global distance metric
Local_CDKT_metric = CDKT_metrics[1]    # Local distance metric
EAAI= True
F1_score=True
#Algorithm Parameter
alpha =0.8# trade-off parameter of local training loss
beta =0.1# trade-off parameter of global distillation loss (Mnist:rep+full loss 0.2)
gamma=0.5
mu=0.1

learning_rate_prox=0.04
local_learning_rate = 0.05
global_learning_rate = 0.02
global_generalized_epochs = 2
if RUNNING_ALG=="FedProx":
    if DATASET=="Cifar10":
        LOCAL_EPOCH = 2
    else:
        LOCAL_EPOCH = 2
else:
    LOCAL_EPOCH=2
NUM_GLOBAL_ITERS = 100 # Number of global rounds








SEED = 1
random.seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
### Agorithm Parameters ###
if Subset:
 N_clients = 50
 Frac_users = 0.5  #20% of users will be selected
else:
 N_clients = 10
 Frac_users = 1.  #All users will be selected
K_Levels =1

if(DATASET == "Cifar100"):
    # NUM_GLOBAL_ITERS = 100
    NUMBER_LABEL=100
else:
    NUMBER_LABEL = 10





rs_file_path = "{}_{}_srate{}_numclient{}_I{}_s{}_f{}_a{}_b{}_RF{}_SS{}_acc{}_gm{}_lm{}_eaai{}_f1score{}.h5".format(RUNNING_ALG, DATASET, Frac_users, N_clients, NUM_GLOBAL_ITERS,
                                Same_model, Full_model,alpha,beta,Rep_Full,Subset,Accelerated,Global_CDKT_metric,Local_CDKT_metric,EAAI,F1_score )
rs_file_path = FIG_PATH + rs_file_path
PLOT_PATH += DATASET+'_'
print("Result Path ", rs_file_path)

# complex_file_path = "{}_{}_I{}_time_.h5".format(DATASET, RUNNING_ALG, NUM_GLOBAL_ITERS)





