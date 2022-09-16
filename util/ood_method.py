from sklearn import metrics
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
from itertools import combinations
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor

def get_auc(y, pred):
    # y: ood is 1，ind is 0
    # pred: ood is larger
    fpr, tpr, _ = metrics.roc_curve(y, pred, pos_label=1)
    auroc = metrics.auc(fpr, tpr)

    precision, recall, _ = metrics.precision_recall_curve(y, pred, pos_label=1)
    aupr_out = metrics.auc(recall, precision)

    pos_pred = [pred[i] for i in range(len(y)) if y[i] == 1]
    pos_pred.sort()
    neg_pred = [pred[i] for i in range(len(y)) if y[i] == 0]
    threshold = pos_pred[int(len(pos_pred) * (1 - 0.95))]
    fpr95 = sum(np.array(neg_pred) > threshold) / len(neg_pred)
    
    pred = [-1 * one for one in pred]
    precision, recall, _ = metrics.precision_recall_curve(y, pred, pos_label=0)
    aupr_in = metrics.auc(recall, precision)

    return auroc, fpr95, aupr_out, aupr_in


def lof(train_loader, test_loader, mdl, k=20):
    def lof_value(data, predict=None, k=20, n_jobs=-1):
        predict = pd.DataFrame(predict)
        # lof, distances_X, neighbors_indices_X = localoutlierfactor(data, predict, k, n_jobs)
        clf = LocalOutlierFactor(n_neighbors=k + 1, algorithm='auto', contamination=1e-5, novelty=True, n_jobs=n_jobs)
        clf.fit(data)
        lof = -clf._decision_function(predict)
        distances_X, neighbors_indices_X = clf.kneighbors(predict, n_neighbors=k + 1)
        return lof, distances_X, neighbors_indices_X

    train_rep = []
    for x, _, _ in train_loader:
        rep = mdl.test(x)
        train_rep.append(rep)
    train_rep = torch.cat(train_rep, dim=0).detach().cpu().numpy()

    test_rep = []
    all_is_ood = []
    for x, _, is_ood in test_loader:
        rep = mdl.test(x)
        test_rep.append(rep)
        all_is_ood += is_ood
    test_rep = torch.cat(test_rep, dim=0).detach().cpu().numpy()
            
    score, _, _ = lof_value(train_rep, test_rep, k=k)
    return all_is_ood, score

def max_logit(all_logit):
    return - np.max(all_logit, axis=1)

def energy(all_logit):
    return - np.log(np.sum(np.exp(all_logit), axis=-1))
    
def ith_logit(all_logit, all_intent_num_pred):
    ind_intent_num = all_logit.shape[1]
    all_intent_num_pred_index = [min(max(all_intent_num_pred[i] - 1, 0), ind_intent_num - 1) for i in range(len(all_intent_num_pred))]
    neg_all_logit = np.array(all_logit) * -1
    sorted_all_logit = np.sort(neg_all_logit, axis=-1) * -1  # from high to low
    ith_max_logit = sorted_all_logit[range(len(neg_all_logit)), all_intent_num_pred_index].tolist()
    neg_ith_max_logit = [-1 * ith_max_logit for ith_max_logit in ith_max_logit]
    return neg_ith_max_logit

def mahalanobis(train_loader, test_loader, mdl, use_intent_num=True):
    train_rep = defaultdict(list)
    for x, y, _ in train_loader:
        rep, _ = mdl(x)
        for index, one_ys in enumerate(y):
            for one_y in one_ys:
                train_rep[one_y].append(rep[index][one_y].unsqueeze(0))

    mu_intent = {}
    sigma = torch.zeros((rep.shape[2], rep.shape[2])).to(rep.device)
    train_num = 0
    for intent in train_rep.keys():
        train_rep[intent] = torch.cat(train_rep[intent], dim=0)
        mu_intent[intent] = torch.mean(train_rep[intent], dim=0, dtype=torch.double)
        diff = train_rep[intent] - mu_intent[intent]
        index = 0  # avoid out of memory 
        while index < diff.shape[0]:
            tmp_diff = torch.unsqueeze(diff[index: index + 200], axis=2)
            tmp_diff_T = torch.transpose(tmp_diff, 2, 1)
            sigma += torch.sum(tmp_diff * tmp_diff_T, axis=0)
            train_num += tmp_diff.shape[0]
            index += 200

    sigma = sigma / train_num
    sigma_I = torch.tensor(np.linalg.inv(sigma.cpu().numpy()), dtype=torch.double).to(sigma.device)
    
    test_rep = []
    all_intent_num_pred = []
    all_intent_num_y = []
    all_is_ood = []

    for x, y, is_ood in test_loader:
        rep, intent_num = mdl(x)
        test_rep.append(rep)
        intent_num_pred = [round(pred) for pred in intent_num.squeeze(1).tolist()]
        
        all_intent_num_pred += intent_num_pred
        all_intent_num_y += [len(y[i]) for i in range(len(y))]
        all_is_ood += is_ood
    
    intent_num_acc = sum([all_intent_num_y[i] == all_intent_num_pred[i] for i in range(len(all_intent_num_y))])  / len(all_intent_num_y)

    test_rep = torch.cat(test_rep, dim=0)  # [bsz, intent, hidden]

    all_m = []
    for intent in train_rep.keys():
        if mu_intent.get(intent, None) is not None:
            diff = test_rep[:, intent, :] - mu_intent[intent]
            m = torch.sum(torch.mm(diff, sigma_I) * diff, axis=1).tolist()
            all_m.append(m)
    all_m = np.array(all_m)

    output = np.sort(all_m.T, axis=-1)  # [bsz, intent]; from low to high
    
    if use_intent_num:
        ind_intent_num = output.shape[1]
        intent_num_pred_index = [min(max(all_intent_num_pred[i] - 1, 0), ind_intent_num - 1) for i in range(len(all_intent_num_pred))]
        score = output[range(len(output)), intent_num_pred_index].tolist()
    else:
        score = output[range(len(output)), [0] * len(output)].tolist()

    return score, all_is_ood, intent_num_acc

