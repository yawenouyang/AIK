from train import Train
import torch.optim as optim
import numpy as np
import fitlog
import torch
from util.ood_method import get_auc, ith_logit, max_logit, lof, energy
import torch.nn as nn


class TrainBCE(Train):

    def __init__(self, config):
        super().__init__(config=config)
        
        self.bce_criterion = torch.nn.BCEWithLogitsLoss()
        self.intent_num_criterion = torch.nn.MSELoss()
        self.sigmoid = nn.Sigmoid()

        other_params = list(set(self.mdl.parameters()) - set(self.mdl.encoder.parameters()))
        self.other_optimizer = optim.Adam(other_params, lr=self.args.lr)

    def train_epoch(self):
        self.current_epoch += 1
        
        all_bce_loss = []
        all_pred = []
        all_y = []
        for x, y, _ in self.train_loader:
            y_one_hot = torch.zeros(len(y), self.args.ind_intent_num).to(self.args.device)
            for index, one_ys in enumerate(y):
                y_one_hot[index, one_ys] = 1
            golden_intent_num = torch.Tensor([[len(y[i])] for i in range(len(y))]).to(self.args.device)

            self.mdl.zero_grad()
            logit, pred_intent_num = self.mdl(x)  # for LOF, pred_intent_num is None

            bce_loss = self.bce_criterion(logit, y_one_hot)
            if self.args.l3:
                intent_num_loss = self.intent_num_criterion(pred_intent_num, golden_intent_num)
                loss = bce_loss * self.args.l1 + intent_num_loss * self.args.l3
            else:
                loss = bce_loss * self.args.l1

            loss.backward()
            self.other_optimizer.step()
            self.bert_optimizer.step()
            
            all_bce_loss.append(bce_loss.item())

            for one_predicted in logit.detach().cpu().numpy():
                all_pred.append((one_predicted > 0.5).nonzero()[0].tolist())
            all_y += y

            self.step += 1

        bce_loss = np.mean(all_bce_loss)
        acc = sum([all_y[i] == all_pred[i] for i in range(len(all_y))]) / (len(all_y))
        print(f"[Epoch {self.current_epoch}] Train BCE Loss={bce_loss}, intent acc={acc}")     
        fitlog.add_loss(bce_loss, name="bce_loss", step=self.step, epoch=self.current_epoch)
        
    @torch.no_grad()
    def evaluate(self, loader):
        if self.args.ood_method == 'lof':
            all_is_ood, score = lof(self.train_loader, self.test_loader, self.mdl)
        else:  # for energy, logit
            all_intent_num_pred = []
            all_logit = []
            all_is_ood = []

            for x, y, is_ood in loader:
                logit, pred_intent_num = self.mdl(x)
                all_logit += logit.tolist()
                all_intent_num_pred += [round(pred) for pred in pred_intent_num.squeeze(1).tolist()]
                all_is_ood += is_ood
                
            all_logit = np.array(all_logit)

            if self.args.ood_method == 'logit':
                if self.args.l3 != 0:
                    score = ith_logit(all_logit, all_intent_num_pred)  # Section 5.6
                else:
                    score = max_logit(all_logit)  # Main results
            elif self.args.ood_method == 'energy':
                score = energy(all_logit)
                
        auroc, fpr95, aupr_out, aupr_in = get_auc(all_is_ood, score)
        print(f'split index {self.args.split_index} auroc: {auroc}, fpr95: {fpr95}, aupr out: {aupr_out}, aupr in: {aupr_in}')

        if loader == self.valid_loader:
            fitlog.add_metric({"valid": {"auroc": auroc, "fpr95": fpr95, "aupr_out": aupr_out, "aupr_in": aupr_in}}, step=self.step, epoch=self.current_epoch)
        else:
            fitlog.add_best_metric({"test": {"auroc": auroc, "fpr95": fpr95, "aupr_out": aupr_out, "aupr_in": aupr_in}})
            self.save_score(score, auroc)
        
        return auroc

if __name__ == '__main__':
    exp = TrainBCE('configs/bce.yaml')
    exp.train()
    exp.test()
    fitlog.finish()