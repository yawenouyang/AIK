from torch.utils.data import Dataset
import pandas as pd
import torch

class ClassifierDataset(Dataset):
    def __init__(self, file_name, label_category):
        self.x = []
        self.y = []
        self.is_ood = []
        if file_name.find('train') != -1:
            split = 'train'
        elif file_name.find('valid') != -1:
            split = 'valid'
        else:
            split = 'test'
        data = pd.read_csv(file_name)
        utts = data['utt']

        ind_intent_set = set(label_category['train'])
        if split == 'train':
            for utt, intents in zip(utts, data['intent']):
                intent_set = set(intents.split('#'))
                if len(intent_set - ind_intent_set) == 0:
                    self.x.append(utt)
                    self.y.append(sorted([label_category['train'].index(intent) for intent in intent_set]))
                    self.is_ood.append(0)
        else:
            split_intent_set = set(label_category[split])
            for utt, intents in zip(utts, data['intent']):
                intent_set = set(intents.split('#'))
                if len(intent_set - ind_intent_set) == 0:
                    self.is_ood.append(0)
                elif len(intent_set - split_intent_set) == 0:
                    self.is_ood.append(1)
                else:
                    continue
                self.x.append(utt)
                self.y.append(sorted([label_category[split].index(intent) for intent in intent_set]))
        
    def __getitem__(self, index):
        return self.x[index], self.y[index], self.is_ood[index]

    def __len__(self):
        return len(self.x)