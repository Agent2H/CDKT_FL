import torch
import os
import torch.multiprocessing as mp
from FLAlgorithms.users.userpFedMe import UserpFedMe
from FLAlgorithms.servers.serverbase import Server
from utils.model_utils import read_data, read_user_data
import numpy as np
 
# Implementation for pFedMe Server

class pFedMe(Server):
    def __init__(self, experiment, device, dataset, algorithm, model, batch_size, learning_rate, beta, L_k, num_glob_iters,
                 local_epochs, optimizer, num_users, K, personal_learning_rate, times, cutoff):
        super().__init__(experiment, device, dataset,algorithm, model[0], batch_size, learning_rate, beta, L_k, num_glob_iters,
                         local_epochs, optimizer, num_users, times)

        # Initialize data for all  users
        self.K = K
        self.personal_learning_rate = personal_learning_rate

        total_users = len(dataset[0][0])
        self.sub_data = cutoff
        if(self.sub_data):
            randomList = self.get_partion(total_users)  
            
        for i in range(total_users):
            id, train , test = read_user_data(i, dataset[0], dataset[1])
            if(self.sub_data):
                if(i in randomList):
                    train, test = self.get_data(train, test)
                    
            user = UserpFedMe(device, id, train, test, model, batch_size, learning_rate, beta, L_k, local_epochs, optimizer, K, personal_learning_rate)
            self.users.append(user)
            self.total_train_samples += user.train_samples
        print("Fraction number of users / total users:",num_users, " / " ,total_users)
        print("Finished creating pFedMe server.")

    def send_grads(self):
        assert (self.users is not None and len(self.users) > 0)
        grads = []
        for param in self.model.parameters():
            if param.grad is None:
                grads.append(torch.zeros_like(param.data))
            else:
                grads.append(param.grad)
        for user in self.users:
            user.set_grads(grads)

    def train(self):
        for glob_iter in range(self.num_glob_iters):
            if(self.experiment):
                self.experiment.set_epoch( glob_iter + 1)
            print("-------------Round number: ",glob_iter, " -------------")
            # send all parameter for users 
            self.send_parameters()

            # Evaluate gloal model on user for each interation
            print("Evaluate global model")
            print("")
            self.evaluate()

            # do update for all users not only selected users
            #for user in self.users:
            #    user.train(self.local_epochs) #* user.train_sample

            for user in self.users:
                user.train(self.local_epochs) 
            # choose several users to send back upated model to server
            # self.personalized_evaluate()
            self.selected_users = self.select_users(glob_iter,self.num_users)

            # Evaluate gloal model on user for each interation
            #print("Evaluate persionalized model")
            #print("")
            self.evaluate_personalized_model()
            #self.aggregate_parameters()
            self.persionalized_aggregate_parameters()

            
        self.save_results()
        self.save_model()
        self.save_results1()

    def save_results1(self):
        RS_PATH = "./results/"
        DATASETS = ["mnist", "fmnist","femnist"]
        Global_rounds = [60,100,100]
        rs_file_path = RS_PATH + "{}_{}_I{}.h5".format(DATASETS[2], "pFedMe", Global_rounds[2])
        print(rs_file_path)
        self.write_file(file_name=rs_file_path, root_test=self.rs_glob_acc, cs_avg_data_test=self.rs_glob_acc_per, cg_avg_data_test=self.rs_c_gen_acc,
                        N_clients=self.num_users)

    
  
