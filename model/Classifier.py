import torch
import torch.nn as nn

from model.BERT import BERT


class Classifier(nn.Module):

    def __init__(self, args):
        super(Classifier, self).__init__()
        self.args = args
        self.encoder = BERT(args)
        self.softmax = nn.Softmax(dim=1)
        self.mlp_intent_num = nn.Linear(self.args.hidden_size, 1)
        self.query = nn.Linear(self.args.hidden_size, self.args.ind_intent_num, bias=False)
        if self.args.method == 'bce':
            self.mlp_logit = nn.Parameter(torch.rand(self.args.ind_intent_num, self.args.hidden_size, requires_grad=True).unsqueeze(0))

    @staticmethod
    def masked_softmax(x, m=None, axis=-1):
        if len(m.size()) == 2:
            m = m.unsqueeze(1)
        if m is not None:
            m = m.float()
            x = x * m
        e_x = torch.exp(x - torch.max(x, dim=axis, keepdim=True)[0])
        if m is not None:
            e_x = e_x * m
        softmax = e_x / (torch.sum(e_x, dim=axis, keepdim=True) + 1e-6)
        return softmax

    def forward(self, sens):
        pooled_output, output, mask = self.encoder(sens)
        intent_num = self.mlp_intent_num(pooled_output)
        
        weight = self.query(output)  # [batch, seq, intent]
        weight = torch.transpose(weight, 1, 2)  # [batch, intent, seq]
        weight = self.masked_softmax(weight, mask)  # [batch, intent, seq]
        rep = torch.bmm(weight, output)  # [batch, intent, hidden]

        if self.args.method == 'bce':
            logit = torch.sum(self.mlp_logit * rep, dim=2)
            return logit, intent_num
        else:
            return rep, intent_num
