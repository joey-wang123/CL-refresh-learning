# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from utils.buffer import Buffer
from torch.nn import functional as F
from models.utils.continual_model import ContinualModel
from utils.args import *
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
from copy import deepcopy
import torch
import copy
from collections import OrderedDict
epsilon = 1E-20





def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Dark Experience Replay++.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--alpha', type=float, required=True,
                        help='Penalty weight.')
    parser.add_argument('--beta', type=float, required=True,
                        help='Penalty weight.')
    return parser


class Derpp(ContinualModel):
    NAME = 'derpp'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(Derpp, self).__init__(backbone, loss, args, transform)

        self.buffer = Buffer(self.args.buffer_size, self.device)

        self.temp = copy.deepcopy(self.net).to(self.device)
        self.temp_opt = torch.optim.SGD(self.temp.parameters(), lr=0.01)

        lr = self.args.lr
        weight_decay = 0.0001
        self.delta = 0.00001
        self.tau = 0.00001

        self.logsoft = nn.LogSoftmax(dim=1)
        self.checkpoint = None
        self.fish = {}
        for name, param in self.net.named_parameters():
            self.fish[name] = torch.zeros_like(param).to(self.device)

        self.opt = torch.optim.SGD(self.net.parameters(), lr=lr, weight_decay=weight_decay)

    def observe(self, inputs, labels, not_aug_inputs):

        self.opt.zero_grad()
        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)


        if not self.buffer.is_empty():
            buf_inputs, _, buf_logits = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            buf_outputs = self.net(buf_inputs)
            loss += self.args.alpha * F.mse_loss(buf_outputs, buf_logits)

            buf_inputs, buf_labels, _ = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            buf_outputs = self.net(buf_inputs)
            loss += self.args.beta * self.loss(buf_outputs, buf_labels)

        loss.backward()
        self.opt.step()



        self.unlearn(inputs=inputs, labels=labels)

        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels,
                             logits=outputs.data)

        return loss.item()


    def unlearn(self, inputs, labels):


        self.temp.load_state_dict(self.net.state_dict())
        self.temp.train()
        outputs = self.temp(inputs)
        loss = - F.cross_entropy(outputs, labels)
        self.temp_opt.zero_grad()
        loss.backward()
        self.temp_opt.step()

        for (model_name, model_param), (temp_name, temp_param) in zip(self.net.named_parameters(), self.temp.named_parameters()):
                weight_update = temp_param - model_param
                model_param_norm = model_param.norm()
                weight_update_norm = weight_update.norm() + epsilon
                norm_update = model_param_norm / weight_update_norm * weight_update
                identity = torch.ones_like(self.fish[model_name])
                with torch.no_grad():
                    model_param.add_(self.delta * torch.mul(1.0/(identity + 0.001*self.fish[model_name]), norm_update + 0.001*torch.randn_like(norm_update)))







    def end_task(self, dataset):


        self.temp.load_state_dict(self.net.state_dict())
        fish = {}
        for name, param in self.temp.named_parameters():
            fish[name] = torch.zeros_like(param).to(self.device)

        for j, data in enumerate(dataset.train_loader):
            inputs, labels, _ = data
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            for ex, lab in zip(inputs, labels):
                self.temp_opt.zero_grad()
                output = self.temp(ex.unsqueeze(0))
                loss = - F.nll_loss(self.logsoft(output), lab.unsqueeze(0),
                                    reduction='none')
                exp_cond_prob = torch.mean(torch.exp(loss.detach().clone()))
                loss = torch.mean(loss)
                loss.backward()


                for name, param in self.temp.named_parameters():
                    fish[name] +=  exp_cond_prob * param.grad ** 2

        for name, param in self.temp.named_parameters():
            fish[name] /= (len(dataset.train_loader) * self.args.batch_size)
       
        for key in self.fish:
                self.fish[key] *= self.tau
                self.fish[key] += fish[key].to(self.device)


        self.checkpoint = self.net.get_params().data.clone()
        self.temp_opt.zero_grad()


