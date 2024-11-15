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
from FLAlgorithms.optimizers.fedoptimizer import SCAFFOLDOptimizer
# Implementation for FedAvg clients
from utils.train_utils import KL_Loss


class UserScaffold(User):
    def __init__(self, device, numeric_id, train_data, test_data, public_data, model, client_model,batch_size, learning_rate, beta, L_k,
                 local_epochs, optimizer):
        super().__init__(device, numeric_id, train_data, test_data, public_data,  model[0],client_model, batch_size, learning_rate, beta, L_k,
                         local_epochs)

        # if(model[1] == "Mclr_CrossEntropy"):
        #     self.loss = nn.CrossEntropyLoss()nghi
        # else:
        #     self.loss = nn.NLLLoss()
        self.learning_rate_scaffold=0.003
        self.loss = nn.CrossEntropyLoss()
        self.criterion_KL = KL_Loss(temperature=3.0)
        self.optimizer = SCAFFOLDOptimizer(self.model.parameters(), lr=self.learning_rate_scaffold)
        self.learning_rate_decay=False
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer,
            gamma=0.99
        )

        self.client_c = []
        for param in self.model.parameters():
            self.client_c.append(torch.zeros_like(param))
        self.global_c = None
        self.global_model = None
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        # self.optimizer = DemProx_SGD(self.model.parameters(), lr=self.learning_rate, mu=0)
    def set_grads(self, new_grads):
        if isinstance(new_grads, nn.Parameter):
            for model_grad, new_grad in zip(self.model.parameters(), new_grads):
                model_grad.data = new_grad.data
        elif isinstance(new_grads, list):
            for idx, model_grad in enumerate(self.model.parameters()):
                model_grad.data = new_grads[idx]

    def train(self, epochs):
        LOSS = 0
        # print(torch.__version__)
        self.model.train()
        for epoch in range(1, epochs + 1):
            self.model.train()
            for X,y in self.trainloader:
                X, y = X.to(self.device), y.to(self.device)#self.get_next_train_batch()
                self.optimizer.zero_grad()
                # output = self.model(X)
                output,_ = self.model(X)
                loss = self.loss(output, y)
                loss.backward()
                self.optimizer.step(self.global_c, self.client_c)
        self.num_batches = len(self.trainloader)
        self.update_yc()
        # self.delta_c, self.delta_y = self.delta_yc()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()
        self.clone_model_paramenter(self.model.parameters(), self.local_model)
        return LOSS

    def update_yc(self):
        for ci, c, x, yi in zip(self.client_c, self.global_c, self.global_model.parameters(), self.model.parameters()):
            ci.data = ci - c + 1 / self.num_batches / self.learning_rate_scaffold * (x - yi)

    def set_new_parameters(self, model, global_c):
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()

        self.global_c = global_c
        self.global_model = model

    def delta_yc(self):
        delta_y = []
        delta_c = []
        for c, x, yi in zip(self.global_c, self.global_model.parameters(), self.model.parameters()):
            delta_y.append(yi - x)
            delta_c.append(- c + 1 / self.num_batches / self.learning_rate_scaffold * (x - yi))

        return delta_y, delta_c
    def train_distill(self, epochs):
        LOSS = 0
        gen_model = copy.deepcopy(self.model)
        self.model.train()
        # gen_model.train()

        for epoch in range(1, epochs + 1):  # local update
            self.model.train()
            # gen_model.train()
            # batch_idx=0
            for X, y in self.trainloader:
                # print(f"batch index{batch_idx}:")
                # batch_idx+= 1
                X, y = X.to(self.device), y.to(self.device)  # self.get_next_train_batch()
                self.optimizer.zero_grad()
                # output = self.model(X)
                output,_ = self.model(X)
                # gen_output = gen_model(X)
                gen_output,_ = gen_model(X)


                lossTrue = self.loss(output, y)

                lossKD= self.criterion_KL(output,gen_output)
                loss = lossTrue + 1* lossKD
                loss.backward()

                updated_model, _ = self.optimizer.step()

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

