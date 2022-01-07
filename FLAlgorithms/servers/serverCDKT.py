import torch
import os
import torch.multiprocessing as mp
from tqdm import tqdm
import torch.nn as nn
from FLAlgorithms.users.userCDKT import UserCDKT
from FLAlgorithms.users.userbase_dem import User
from FLAlgorithms.servers.serverbase_dem import Dem_Server
from Setting import rs_file_path, N_clients
from utils.data_utils import write_file
from utils.dem_plot import plot_from_file
from utils.model_utils import read_data, read_user_data, read_public_data
from torch.utils.data import DataLoader
import numpy as np
from FLAlgorithms.optimizers.fedoptimizer import DemProx_SGD
from FLAlgorithms.trainmodel.models import *
# Implementation for Server
from utils.train_utils import KL_Loss, JSD
from Setting import *
class CDKT(Dem_Server):
    def __init__(self, experiment, device, dataset, algorithm, model,  client_model, batch_size, learning_rate, beta, L_k, num_glob_iters, local_epochs, optimizer, num_users, times , cutoff,args):
        super().__init__(experiment, device, dataset, algorithm, model[0],  client_model,batch_size, learning_rate, beta, L_k, num_glob_iters,local_epochs, optimizer, num_users, times,args)

        # Initialize data for all  users
        self.K = 0
        self.loss = nn.CrossEntropyLoss()
        self.mu=args.mu
        self.optimizer = DemProx_SGD(self.model.parameters(), lr=global_learning_rate, mu=0)

        # self.optimizer = torch.optim.Adagrad(self.model.parameters(), lr=global_learning_rate)
        self.criterion_KL = KL_Loss(temperature=3.0)
        self.criterion_JSD = JSD()
        self.avg_local_dict_prev_1 = dict()
        self.gamma = gamma

        # total_users = len(dataset[0][0])
        self.sub_data = cutoff
        if(self.sub_data):
            randomList = self.get_partion(self.total_users)

        self.publicdatasetloader = DataLoader(read_public_data(dataset[0], dataset[1]), self.batch_size, shuffle=True)  # no shuffle
        # self.publicloader= list(enumerate(self.publicdatasetloader))
        self.enum_publicDS = enumerate(self.publicdatasetloader)
        self.publicloader =[]
        for b, (x, y) in self.enum_publicDS:
            self.publicloader.append((b,(x,y)))
            if(b<1): print(y)

        # self.publicdatasetlist= DataLoader(public_data, self.batch_size, shuffle=False)  # no shuffle
        sample=[]
        testing_sample=[]
        for i in range(self.total_users):
            id, train , test, public = read_user_data(i, dataset[0], dataset[1])
            print("User ", id, ": Numb of Training data", len(train))
            sample.append(len(train)+len(test))
            # print("public len",len(public))
            if(self.sub_data):
                if(i in randomList):
                    train, test = self.get_data(train, test)
            user = UserCDKT(device, id, train, test, public, model,client_model, batch_size, learning_rate,beta,L_k, local_epochs, optimizer)
            user.publicdatasetloader = self.publicloader
            # self.publicloader = user.publicdatasetloader
            self.users.append(user)
            self.total_train_samples += user.train_samples
            # print(user.train_samples)
        # print("train sample median :", np.median(training_sample))
        # print("test sample median :", np.median(testing_sample))
        print("sample is", np.median(sample))


        self.local_model = user.local_model
        self.train_samples = len(train)

            
        print("Fraction number of users / total users:",num_users, " / " ,self.total_users)

        print("Finished creating server.")

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

    def generalized_knowledge_construction(self, epochs,dataset,glob_iter):
        LOSS = 0

        self.model.train()


        # user = User
        # local_model = UserCDKT.local_model
        # local_model = copy.deepcopy(list(self.model))
        # avg_local_output_public=[]

        avg_local_dict = dict()
        avg_rep_local_dict = dict()

        local_dict= dict()
        local_dict_prev=dict()

        avg_local_dict_prev_2=dict()
        agg_local_dict=dict()
        clients_local_dict = dict()
        clients_rep_local_dict = dict()
        clients_rep_local_dict_2 = dict()
        rep_local_dict = dict()
        c=0
        for user in self.selected_users:
            c+=1
            local_dict.clear()
            # print(self.publicloader[0])
            # for batch_idx, (X_public, y_public) in enumerate(self.publicloader):
            for batch_idx, (X_public, y_public) in self.publicloader:
                X_public, y_public = X_public.to(self.device), y_public.to(self.device)

                # local_output_public = users.model(X_public)
                # user.model.fc1.register_forward_hook(get_activation('fc1'))
                # local_output_public = user.model(X_public)
                # rep_local_output_public = activation['fc1']

                # local_output_public, rep_local_output_public = user.client_model(X_public)
                if Same_model:
                    local_output_public, rep_local_output_public = user.model(X_public)
                else:
                    local_output_public, rep_local_output_public = user.client_model(X_public)

                rep_local_output_public = rep_local_output_public.cpu().detach().numpy()
                local_output_public = local_output_public.cpu().detach().numpy()

                local_dict[batch_idx]=local_output_public
                rep_local_dict[batch_idx] = rep_local_output_public
                # if Moving_Average:
                #     agg_local_dict = (1 - gamma) * local_dict_prev[batch_idx] + gamma * local_dict[batch_idx]
                #     local_dict_prev.clear()
                # local_dict_prev = local_dict
            clients_local_dict[c] = local_dict
            clients_rep_local_dict[c] = rep_local_dict
            clients_rep_local_dict_2[c] = rep_local_dict

        # print(clients_rep_local_dict_2.keys())
        #Avg local output
        n=0
        for client_idx in clients_local_dict.keys():

            c_logits = clients_local_dict[client_idx]

            for batch_idx, (X_public, y_public) in enumerate(self.publicloader):
                if (n == 0):
                    avg_local_dict[batch_idx] =  c_logits[batch_idx] / self.total_users
                else:
                    avg_local_dict[batch_idx] +=  c_logits[batch_idx] / self.total_users

            n+=1

        # TODO: Avg rep local output according num of samples of each users
        # TODO: Sum of DIR regularizer :done
        #
        if self.gamma < 0.8 and glob_iter>=30:
            self.gamma += (0.8-0.5)/(NUM_GLOBAL_ITERS-30)
            # self.gamma += (0.5 - 0.05) / NUM_GLOBAL_ITERS


        # #Accelerated avg knowledge
        # print(self.avg_local_dict_prev_1.keys())




        for batch_idx, (X_public, y_public) in enumerate(self.publicloader):
            if ( glob_iter== 0):
                agg_local_dict[batch_idx] = avg_local_dict[batch_idx]
            else:
                agg_local_dict[batch_idx] = self.gamma*self.avg_local_dict_prev_1[batch_idx] + (1-self.gamma)*avg_local_dict[batch_idx]
        self.avg_local_dict_prev_1 = agg_local_dict

        print("gamma", self.gamma)


        # print(agg_local_dict.keys())
        # print(avg_local_dict.keys())

        # Avg rep local output according to num of users, num of samples of each users
        m = 0
        for client_idx in clients_rep_local_dict.keys():

            c_rep_logits = clients_rep_local_dict[client_idx]


            # for batch_idx, _ in enumerate(self.publicloader):
            for batch_idx, _ in self.publicloader:
                if (m == 0):
                    avg_rep_local_dict[batch_idx] = c_rep_logits[batch_idx] / self.total_users
                else:
                    avg_rep_local_dict[batch_idx] += c_rep_logits[batch_idx] / self.total_users

            m += 1

        # Avg rep local output according to num of samples of each user

        # k = 0
        # for client_idx in clients_rep_local_dict_2.keys():
        #
        #     c_rep_logits_2 = clients_rep_local_dict_2[client_idx]
        #     # print(self.total_train_samples)
        #
        #     print(self.train_samples)
        #     for batch_idx, (X_public, y_public) in enumerate(self.publicloader):
        #
        #         if (k == 0):
        #             avg_rep_local_dict[batch_idx] = c_rep_logits_2[batch_idx] / self.total_train_samples
        #         else:
        #             avg_rep_local_dict[batch_idx] += c_rep_logits_2[batch_idx] / self.total_train_samples
        #
        #     k += 1


        # print(local_dict)
        # avg_local_dict = sum_local_dict/self.total_users
        # print(c)
        # print("finish local dict")

        #Global distillation
        # TODO: implement several global iterations to construct generalized knowledge :done
        for epoch in range(1, epochs+1):

            self.model.train()

            # for batch_idx, (X_public, y_public) in enumerate(self.publicloader):
            for batch_idx, (X_public, y_public) in self.publicloader:
                # print(batch_idx)
                if Moving_Average:
                    batch_logits = torch.from_numpy(agg_local_dict[batch_idx]).float().to(self.device)
                else:
                    batch_logits = torch.from_numpy(avg_local_dict[batch_idx]).float().to(self.device)
                batch_rep_logits = torch.from_numpy(avg_rep_local_dict[batch_idx]).float().to(self.device)
                X_public, y_public = X_public.to(self.device), y_public.to(self.device)


                # self.model.fc1.register_forward_hook(get_activation('fc1'))
                # output_public=self.model(X_public)
                # rep_output_public = activation['fc1']

                # print(batch_rep_logits)
                # print("output_pub", output_public)

                output_public, rep_output_public = self.model(X_public)
                #
                if Tune_output:
                    y_onehot = F.one_hot(y_public, num_classes=NUMBER_LABEL)
                    batch_logits = (batch_logits + y_onehot)/2.0

                lossTrue = self.loss(output_public, y_public)
                lossKD=lossJSD=norm2loss=0
                if Full_model:
                    lossKD = self.criterion_KL(output_public, batch_logits)
                    norm2loss = torch.dist(output_public, batch_logits, p=2)
                    lossJSD = self.criterion_JSD(output_public, batch_logits)
                else:
                    if Rep_Full:
                        lossKD = self.criterion_KL(output_public, batch_logits)
                        norm2loss = torch.dist(output_public, batch_logits, p=2)
                        lossJSD = self.criterion_JSD(output_public, batch_logits)
                        lossKD += self.criterion_KL(rep_output_public, batch_rep_logits)
                        lossJSD += self.criterion_JSD(rep_output_public, batch_rep_logits)
                        norm2loss = norm2loss + torch.dist(rep_output_public, batch_rep_logits, p=2)
                    else:
                        lossKD = self.criterion_KL(rep_output_public, batch_rep_logits)
                        lossJSD = self.criterion_JSD(rep_output_public, batch_rep_logits)
                        norm2loss = torch.dist(rep_output_public, batch_rep_logits, p=2)

                if Global_CDKT_metric == "KL":
                    loss = lossTrue + beta * lossKD
                    # loss = beta*lossTrue + (1-beta) * lossKD
                elif Global_CDKT_metric == "Norm2":
                    loss = lossTrue + beta * norm2loss
                elif Global_CDKT_metric == "JSD":
                    loss = lossTrue + beta * lossJSD

                self.optimizer.zero_grad()
                loss.backward()

                self.optimizer.step()


    def generalized_knowledge_ensemble(self, epochs):
        LOSS = 0
        self.model.train()


        avg_local_dict = dict()
        avg_rep_local_dict = dict()
        local_dict = dict()
        clients_local_dict = dict()
        clients_rep_local_dict = dict()
        rep_local_dict = dict()
        c = 0
        for user in self.selected_users:
            c += 1
            local_dict.clear()
            # for batch_idx, (X_public, y_public) in enumerate(self.publicloader):
            for batch_idx, (X_public, y_public) in self.publicloader:
                X_public, y_public = X_public.to(self.device), y_public.to(self.device)

                # local_output_public = users.model(X_public)
                # user.model.fc1.register_forward_hook(get_activation('fc1'))
                # local_output_public = user.model(X_public)
                # rep_local_output_public = activation['fc1']

                if Same_model:
                    local_output_public, rep_local_output_public = user.model(X_public)
                else:
                    local_output_public, rep_local_output_public = user.client_model(X_public)

                rep_local_output_public = rep_local_output_public.cpu().detach().numpy()
                local_output_public = local_output_public.cpu().detach().numpy()

                local_dict[batch_idx] = local_output_public
                rep_local_dict[batch_idx] = rep_local_output_public
            clients_local_dict[c] = local_dict
            clients_rep_local_dict[c] = rep_local_dict


        for epoch in range(1, epochs+1):

            self.model.train()

            # for batch_idx, (X_public, y_public) in enumerate(self.publicloader):
            for batch_idx, (X_public, y_public) in self.publicloader:
                X_public, y_public = X_public.to(self.device), y_public.to(self.device)
                # if batch_idx ==0:
                #     print('global batch',X_public[0])


                output_public, rep_output_public = self.model(X_public)


                # print(batch_rep_logits)
                # print("output_pub", output_public)
                lossTrue = self.loss(output_public, y_public)
                lossKD = lossJSD = norm2loss =0
                for client_index in clients_rep_local_dict:
                    client_rep_logits = clients_rep_local_dict[client_index][batch_idx]
                    batch_rep_logits = torch.from_numpy(client_rep_logits).float().to(self.device)
                    client_logits = clients_local_dict[client_index][batch_idx]
                    batch_logits = torch.from_numpy(client_logits).float().to(self.device)
                    if Full_model:
                      lossKD += self.criterion_KL(output_public, batch_logits).to(self.device)
                      lossJSD += self.criterion_JSD(output_public, batch_logits)
                      norm2loss += torch.dist(output_public, batch_logits, p=2)

                    else:
                        lossKD += self.criterion_KL(output_public, batch_logits).to(self.device)
                        lossJSD += self.criterion_JSD(output_public, batch_logits)
                        norm2loss += torch.dist(output_public, batch_logits, p=2)

                        lossKD+= self.criterion_KL(rep_output_public,batch_rep_logits).to(self.device)
                        lossJSD += self.criterion_JSD(rep_output_public, batch_rep_logits)
                        norm2loss += torch.dist(rep_output_public, batch_rep_logits, p=2)

                if Global_CDKT_metric == "KL":
                    loss = lossTrue + beta * lossKD
                    # loss =  beta * lossKD
                elif Global_CDKT_metric == "Norm2":
                    loss = lossTrue + beta * norm2loss
                elif Global_CDKT_metric == "JSD":
                    loss = lossTrue + beta * lossJSD

                self.optimizer.zero_grad()
                loss.backward()

                updated_model, _ = self.optimizer.step()

    def train(self):
        for glob_iter in range(self.num_glob_iters):

            self.selected_users = self.select_users(glob_iter,self.num_users)
            # self.selected_users = self.users

            if(self.experiment):
                self.experiment.set_epoch( glob_iter + 1)
            print("-------------Round number: ",glob_iter, " -------------")

            # ============= Test each client =============
            tqdm.write('============= Test Client Models - Specialization ============= ')
            stest_acu, strain_acc = self.evaluating_clients(glob_iter, mode="spe")
            self.cs_avg_data_test.append(stest_acu)
            self.cs_avg_data_train.append(strain_acc)
            tqdm.write('============= Test Client Models - Generalization ============= ')
            gtest_acu, gtrain_acc = self.evaluating_clients(glob_iter, mode="gen")
            self.cg_avg_data_test.append(gtest_acu)
            self.cg_avg_data_train.append(gtrain_acc)
            tqdm.write('============= Test Global Models  ============= ')
            #loss_ = 0
            # self.send_parameters()   #Broadcast the global model to all clients
            # self.evaluating_global(glob_iter)
            self.evaluating_global_CDKT(glob_iter)


            # # Evaluate model each interation
            # self.evaluate()
            # global alpha
            # if alpha < 0.3 and glob_iter>=0:
            #     alpha += (0.3 - 0.15) / (NUM_GLOBAL_ITERS - 0)
            # print("alpha 1= ", alpha)
            
            #NOTE: this is required for the ``fork`` method to work
            for user in self.selected_users:
                if(glob_iter==0):
                    user.train(self.local_epochs)
                else:
                    # user.train(self.local_epochs)

                    user.train_distill(self.local_epochs,self.model,glob_iter,alpha)

                    # user.train_prox(self.local_epochs)

            #

            # self.aggregate_parameters()
            # print("gamma:", self.gamma)
            # if self.gamma == 0:
            #     self.gamma = 0.5
            #     self.gamma = max(self.gamma / 1.02, 0.1)

            if Ensemble == True:
                self.generalized_knowledge_ensemble(global_generalized_epochs)
            else:
                self.generalized_knowledge_construction(global_generalized_epochs,self.args.dataset,glob_iter)

        self.save_results1()
        self.save_model()

    def save_results1(self):
        write_file(file_name=rs_file_path, root_test=self.rs_glob_acc, root_train=self.rs_train_acc,
                   cs_avg_data_test=self.cs_avg_data_test, cs_avg_data_train=self.cs_avg_data_train,
                   cg_avg_data_test=self.cg_avg_data_test, cg_avg_data_train=self.cg_avg_data_train,
                   cs_data_test=self.cs_data_test, cs_data_train=self.cs_data_train, cg_data_test=self.cg_data_test,
                   cg_data_train=self.cg_data_train, N_clients=[N_clients])
        plot_from_file()
