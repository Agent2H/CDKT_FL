import torch
import os
import torch.multiprocessing as mp
from tqdm import tqdm

from FLAlgorithms.users.useravg import UserAVG
from FLAlgorithms.servers.serverbase_dem import Dem_Server
from Setting import rs_file_path, N_clients
from utils.data_utils import write_file
from utils.dem_plot import plot_from_file
from utils.model_utils import read_data, read_user_data
import numpy as np

# Implementation for FedAvg Server

class FedAvg(Dem_Server):
    def __init__(self, experiment, device, dataset,algorithm, model, client_model,batch_size, learning_rate, beta, L_k, num_glob_iters, local_epochs, optimizer, num_users, times , cutoff,args):
        super().__init__(experiment, device, dataset,algorithm, model[0], client_model,batch_size, learning_rate, beta, L_k, num_glob_iters,local_epochs, optimizer, num_users, times,args)

        # Initialize data for all  users
        self.K = 0

        # total_users = len(dataset[0][0])
        self.sub_data = cutoff
        if(self.sub_data):
            randomList = self.get_partion(self.total_users)
        for i in range(self.total_users):
            id, train , test, public = read_user_data(i, dataset[0], dataset[1])
            # print("public",len(public))
            print("User ", id, ": Numb of Training data", len(train))
            if(self.sub_data):
                if(i in randomList):
                    train, test = self.get_data(train, test)
            user = UserAVG(device, id, train, test, public, model,client_model, batch_size, learning_rate,beta,L_k, local_epochs, optimizer)
            self.users.append(user)
            self.total_train_samples += user.train_samples

        print("Fraction number of users / total users:",num_users, " / " ,self.total_users)
        print("Finished creating FedAvg server.")

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

            self.selected_users = self.select_users(glob_iter,self.num_users)
            # self.selected_users = self.users
            # print("selected users are",self.selected_users)
            if(self.experiment):
                self.experiment.set_epoch( glob_iter + 1)
            print("-------------Round number: ",glob_iter, " -------------")

            # ============= Test each client =============
            tqdm.write('============= Test Client Models - Specialization ============= ')
            stest_acu, strain_acc ,sf1_acc= self.evaluating_clients(glob_iter, mode="spe")
            self.cs_avg_data_test.append(stest_acu)
            self.cs_avg_data_train.append(strain_acc)
            self.rs_c_spec_f1.append(sf1_acc)
            tqdm.write('============= Test Client Models - Generalization ============= ')
            gtest_acu, gtrain_acc,gf1_acc = self.evaluating_clients(glob_iter, mode="gen")
            self.cg_avg_data_test.append(gtest_acu)
            self.cg_avg_data_train.append(gtrain_acc)
            self.rs_c_gen_f1.append(gf1_acc)
            tqdm.write('============= Test Global Models  ============= ')
            #loss_ = 0
            self.send_parameters()   #Broadcast the global model to all clients
            self.evaluating_global(glob_iter)



            # # Evaluate model each interation
            # self.evaluate()

            # self.selected_users = self.select_users(glob_iter,self.num_users)
            # self.selected_users = self.users

            #NOTE: this is required for the ``fork`` method to work
            for user in self.selected_users:
                if(glob_iter==0):
                    user.train(self.local_epochs)
                else:
                    user.train(self.local_epochs)
                    # user.train_distill(self.local_epochs)
                    # user.train_prox(self.local_epochs)
                    # print(user.model.state_dict())
            self.aggregate_parameters()
            # print(self.model.state_dict())
            #if Using_public_data: TODO:
            # for server_param in self.model.parameters():
            #     print(server_param.data)

        self.save_results1()
        self.save_model()

    def save_results1(self):
        write_file(file_name=rs_file_path, root_test=self.rs_glob_acc, root_train=self.rs_train_acc,
                   cs_avg_data_test=self.cs_avg_data_test, cs_avg_data_train=self.cs_avg_data_train,
                   cg_avg_data_test=self.cg_avg_data_test, cg_avg_data_train=self.cg_avg_data_train,
                   cs_data_test=self.cs_data_test, cs_data_train=self.cs_data_train, cg_data_test=self.cg_data_test,
                   cg_data_train=self.cg_data_train,global_f1=self.rs_global_f1,
                   spec_f1=self.rs_c_spec_f1,gen_f1=self.rs_c_gen_f1, N_clients=[N_clients])
        print('result file path',rs_file_path)
        # plot_from_file()
