from abc import abstractmethod
import torch
import time
import numpy as np
import copy
import os
import torch.nn as nn
from util.training_util import create_loader, load_args, set_seed
import fitlog
from abc import ABC, abstractmethod
import pandas as pd
from split import split
from transformers import AdamW


class Train(ABC):

    def __init__(self, config):
        
        # get ready
        self.args = load_args(config)
        log_dir = f'log/{self.args.method}_{self.args.ood_method}'
        if not os.path.isdir(log_dir):
            os.mkdir(log_dir)
        fitlog.set_log_dir(log_dir)
        set_seed(self.args.seed)

        # load data
        intent_set = list(pd.read_csv(self.args.intent)['intent'])
        valid_ood_set = split[self.args.dataset][self.args.split_index]['valid_ood_label'].split(',')
        test_ood_set = split[self.args.dataset][self.args.split_index]['test_ood_label'].split(',')
        ind_set = [intent for intent in intent_set if intent not in test_ood_set and intent not in valid_ood_set]

        label_category = {
            'train': ind_set,
            'valid': ind_set + valid_ood_set,
            'test': ind_set + test_ood_set
        }
        
        self.ind_set = ind_set
        self.args.ind_intent_num = len(ind_set)

        fitlog.add_hyper(self.args)
        
        self.train_loader = create_loader(self.args.batch_size, self.args.train, label_category, True)
        self.valid_loader = create_loader(self.args.batch_size, self.args.valid, label_category, False)
        self.test_loader = create_loader(self.args.batch_size, self.args.test, label_category, False)

        if self.args.ood_method == 'lof':
            from model.Classifier_for_LOF import Classifier
        else:
            from model.Classifier import Classifier

        self.mdl = Classifier(self.args).to(self.args.device)
    
        # for training
        self.step = 0
        self.current_epoch = 0
        self.model_parameter = None
        self.softmax = nn.Softmax(dim=-1)

        bert_param = list(self.mdl.encoder.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        bert_grouped_param = [
            {'params': [p for n, p in bert_param if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in bert_param if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        self.bert_optimizer = AdamW(bert_grouped_param, lr=self.args.bert_lr)

        print(self.args)

    @abstractmethod
    def evaluate(self, loader):
        pass

    @abstractmethod
    def train_epoch(self):
        pass

    def save_model(self, result):
        self.args.result = result
        args_dict = vars(self.args)
        if result < self.args.save_threshold:
            print('Exit without saving model parameter.')
        else:
            dir_name = f'{self.args.param_dir}/{self.args.dataset}/{self.args.loss}_{result}'
            is_exists = os.path.exists(dir_name)
            if is_exists:
                print(f'{dir_name} has existed')
                dir_name += f'_{time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())}'
                print(f'New dir name: {dir_name}')
            os.makedirs(dir_name)
            torch.save(self.model_parameter, f'{dir_name}/params.pkl')
            with open(f'{dir_name}/hyper_parameter.txt', 'w') as f:
                for key, value in args_dict.items():
                    f.write(f'{key}: {value} \n')


    def train(self):
        max_auroc = 0
        max_auroc_epoch = -1
        print("Start training")
        while self.current_epoch < self.args.epoch:
            self.mdl.train()            
            self.train_epoch()
    
            self.mdl.eval()
            auroc = self.evaluate(self.valid_loader)
            
            if auroc > max_auroc:
                max_auroc = auroc
                max_auroc_epoch = self.current_epoch
                self.model_parameter = copy.deepcopy(self.mdl.state_dict())
            elif max_auroc != 0 and (self.current_epoch - max_auroc_epoch == self.args.early_stop):
                print("Early Stop")
                break

        print("End training")
        print(f"Max auroc on valid is {max_auroc}")
         
    def test(self):
        self.mdl.eval()
        if self.model_parameter is not None: 
            self.mdl.load_state_dict(self.model_parameter)
        metric = self.evaluate(self.test_loader)
        self.save_model(metric)
    
    def save_score(self, score, auroc):
        dir_name = f'score/{self.args.dataset}_{self.args.split_index}'
        if not os.path.isdir(dir_name):
            os.mkdir(dir_name)
        np.save(f'{dir_name}/{self.args.ood_method}_{auroc}.npy', score)
