import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader
# from FLAlgorithms.users.userbase import User
from FLAlgorithms.optimizers.fedoptimizer import DemProx_SGD
from FLAlgorithms.users.userbase_dem import User
from FLAlgorithms.trainmodel.models import *
from torch import Tensor
from collections import OrderedDict
# Implementation for clients
from utils.train_utils import KL_Loss, JSD

from Setting import *

class UserCDKT(User):
    def __init__(self, device, numeric_id, train_data, test_data, public_data, model,client_model, batch_size, learning_rate, beta, L_k,
                 local_epochs, optimizer):
        super().__init__(device, numeric_id, train_data, test_data, public_data,  model[0],client_model, batch_size, learning_rate, beta, L_k,
                         local_epochs)

        # if(model[1] == "Mclr_CrossEntropy"):
        #     self.loss = nn.CrossEntropyLoss()
        # else:
        #     self.loss = nn.NLLLoss()
        self.loss = nn.CrossEntropyLoss()
        self.criterion_KL = KL_Loss(temperature=3.0)
        self.criterion_JSD = JSD()

        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        if Same_model:
            self.optimizer = DemProx_SGD(self.model.parameters(), lr=local_learning_rate, mu=0)
        else:
            self.optimizer = DemProx_SGD(self.client_model.parameters(), lr=local_learning_rate, mu=0)

    def set_grads(self, new_grads):
        if isinstance(new_grads, nn.Parameter):
            for model_grad, new_grad in zip(self.model.parameters(), new_grads):
                model_grad.data = new_grad.data
        elif isinstance(new_grads, list):
            for idx, model_grad in enumerate(self.model.parameters()):
                model_grad.data = new_grads[idx]

    def train(self, epochs):
        LOSS = 0
        if Same_model:
         self.model.train()
        else:
         self.client_model.train()
        for epoch in range(1, epochs + 1):
            if Same_model:
                self.model.train()
            else:
                self.client_model.train()
            for X,y in self.trainloader:
                X, y = X.to(self.device), y.to(self.device)#self.get_next_train_batch()
                self.optimizer.zero_grad()
                # output = self.model(X)
                if Same_model:
                    output,_ = self.model(X)
                else:
                    output, _ = self.client_model(X)
                loss = self.loss(output, y)
                loss.backward()
                self.optimizer.step()
        # self.clone_model_paramenter(self.model.parameters(), self.local_model)
        return LOSS



    def train_distill(self, epochs, global_model,glob_iter,alpha1=alpha):
        LOSS = 0
        gen_model = copy.deepcopy(global_model)
        if Same_model:
            self.model.train()
        else:
            self.client_model.train()
        # gen_model.train()
        b=0
        # public_data = enumerate(self.publicdatasetloader)
        # public_data = self.publicdatasetloader
        batch_num = 0

        # if self.alpha < 0.3:
        #     self.alpha += (0.3 - 0.15) / (NUM_GLOBAL_ITERS - 20)
        # print("alpha = ", self.alpha)

        # if alpha < 0.3 and glob_iter>=20:
        #     old_alpha = alpha
        #     alpha += (0.3 - 0.15) / (NUM_GLOBAL_ITERS - 20)
        #     alpha = old_alpha
        # print("alpha 2 = ", alpha1)

        for epoch in range(1, epochs + 1):  # local updatec


            if Same_model:
                self.model.train()
            else:
                self.client_model.train()

            # gen_model.train()
            # batch_idx=0
            # TODO: loop all of batch of public data : done
            private_data = enumerate(self.trainloader)
            public_data = self.publicdatasetloader[batch_num:]
            # print(type(private_data))
            # print(type(public_data))
            for [local_batch_idx,(X, y)], [batch_idx, (X_public,y_public)] in zip(private_data, public_data):
                # print("local batch id: ", local_batch_idx, " public batch ", batch_idx)
                X, y = X.to(self.device), y.to(self.device)  # self.get_next_train_batch()
                X_public, y_public = X_public.to(self.device), y_public.to(self.device)
                # if batch_idx==0:
                #     print('local batch',X_public[0])

                #### Get activation and output from local model
                # self.model.fc1.register_forward_hook(get_activation('fc1'))
                # output_public = self.model(X_public)
                # rep_output_public = activation['fc1']
                #
                # #### Get activation and output from global model
                # gen_model.fc1.register_forward_hook(get_activation('fc1'))
                # gen_output_public = gen_model(X_public)
                # rep_gen_output_public = activation['fc1']
                # output = self.model(X)

                if Same_model:
                    output_public, rep_output_public = self.model(X_public)
                    output, _ = self.model(X)
                else:
                    output_public, rep_output_public = self.client_model(X_public)
                    output, _ = self.client_model(X)

                gen_output_public, rep_gen_output_public = gen_model(X_public)




                # if(batch_idx<1):
                #     print('user output_public:',F.softmax(output_public))

                lossTrue = self.loss(output, y)
                lossKD=lossJSD=norm2loss=0
                if Full_model:
                    lossKD = self.criterion_KL(output_public, gen_output_public)
                    norm2loss = torch.dist(output_public, gen_output_public, p=2)
                    lossJSD = self.criterion_JSD(output_public, gen_output_public)
                else:
                    lossKD = self.criterion_KL(output_public, gen_output_public)
                    norm2loss = torch.dist(output_public, gen_output_public, p=2)
                    lossJSD = self.criterion_JSD(output_public, gen_output_public)
                    lossKD += self.criterion_KL(rep_output_public,rep_gen_output_public)
                    lossJSD += self.criterion_JSD(rep_output_public, rep_gen_output_public)
                    norm2loss = norm2loss+ torch.dist(rep_output_public, rep_gen_output_public, p=2)


                if Local_CDKT_metric == "KL":
                    loss = lossTrue + alpha1*lossKD
                elif Local_CDKT_metric == "Norm2":
                    loss = lossTrue + alpha1*norm2loss
                elif Local_CDKT_metric == "JSD":
                    loss = lossTrue + alpha1*lossJSD

                self.optimizer.zero_grad()
                loss.backward()

                updated_model, _ = self.optimizer.step()
                batch_num +=1
            # print("num of public batch", c)

        # logits_dict = dict()
        # for batch_idx, (X_public,y_public) in enumerate(self.publicdatasetloader):
        #     X_public, y_public = X_public.to(self.device), y_public.to(self.device)
        #     logits_pub = self.model(X_public)
        #     logits_pub = logits_pub.cpu().detach().numpy()
        #     logits_dict[batch_idx] = logits_pub
        # print("all logit")
        # print(logits_dict)
        # print("finish all logits")


        self.clone_model_paramenter(self.model.parameters(), self.local_model)

    def train_prox(self, epochs):
        LOSS = 0
        gen_model = copy.deepcopy(self.model)
        self.model.train()

        for epoch in range(1, epochs + 1):  # local update
            self.model.train()

            # batch_idx=0
            for X, y in self.trainloader:
                # print(f"batch index{batch_idx}:")
                # batch_idx+= 1
                X, y = X.to(self.device), y.to(self.device)  # self.get_next_train_batch()
                self.optimizer.zero_grad()
                output = self.model(X)

                loss = self.loss(output, y)
                loss.backward()

                updated_model, _ = self.optimizer.step(mu_t=1, gen_weights=(gen_model,1.0))

        # update local model as local_weight_upated
        self.clone_model_paramenter(self.model.parameters(), self.local_model)

        # self.update_parameters(updated_model)

