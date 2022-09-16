from util.ClassifierDataset import ClassifierDataset
import torch
import yaml
import argparse
import numpy as np
import random
from collections import defaultdict

def collate_fn(batch):
    utts = []
    labels = []
    is_oods = []
    for instance in batch:
        utts.append(instance[0])
        labels.append(instance[1])
        is_oods.append(instance[2])
    return utts, labels, is_oods

def create_loader(batch_size, file_name, label_category, shuffle):
    dataset = ClassifierDataset(file_name, label_category)
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False, collate_fn=collate_fn)
    return loader

def load_args(config, base_config='configs/base.yaml'):
    with open(base_config) as f:
        base = yaml.load(f, Loader=yaml.FullLoader)
    
    with open(config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    
    cfg.update(base)

    # cfg to args
    parser = argparse.ArgumentParser()
    for key, value in cfg.items():
        parser.add_argument('--{}'.format(key), type=type(value), default=value)
    args = parser.parse_args()

    device = 'cpu'
    if args.gpu >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        print("Using device:{}".format(torch.cuda.current_device()))
        device = 'cuda'

    device = torch.device(device)
    args.device = device
    args.train = f'data/{args.dataset}/train.csv'
    args.test = f'data/{args.dataset}/test.csv'
    args.valid = f'data/{args.dataset}/valid.csv'
    args.intent = f'data/{args.dataset}/intent.csv'

    return args

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def init_mmc_center(args, ind_set):
    def get_mmc(intent_num, feature_num, C):
        mean = torch.zeros(intent_num, feature_num)
        mean[0][0] = 1
        for k in range(1, intent_num):
            for j in range(k):
                mean[k][j] = - (1 / (intent_num - 1) + torch.dot(mean[k], mean[j])) / mean[j][j]
            mean[k][k] = torch.sqrt(torch.abs(1 - torch.norm(mean[k]) ** 2))
        mean = mean * C
        return mean
        
    domain_count = defaultdict(int)
    for intent in ind_set:
        domain_count[intent.split('-')[1]] += 1

    mean = torch.zeros(len(ind_set), args.hidden_size)
    first_mean = get_mmc(len(domain_count), args.hidden_size, 5)

    for i1, (domain, count) in enumerate(domain_count.items()):
        last_mean = get_mmc(count, args.hidden_size, 2)
        used_count = 0
        for i2, intent in enumerate(ind_set):
            if intent.split('-')[1] == domain:
                mean[i2] = first_mean[i1] + last_mean[used_count]
                used_count += 1

    mean = mean.to(args.device)
    return mean