from transformers import BertModel, BertTokenizer
import torch.nn as nn

class BERT(nn.Module):
    
    def __init__(self, args):
        super(BERT, self).__init__()
        self.args = args
        self.tokenizer = BertTokenizer.from_pretrained(self.args.bert_path)
        self.encoder = BertModel.from_pretrained(self.args.bert_path, return_dict=True)

    def forward(self, sens):
        encoding = self.tokenizer(sens, return_tensors='pt', padding=True, truncation=True, max_length=512)
        input_ids = encoding['input_ids'].to(self.args.device)
        attention_mask = encoding['attention_mask'].to(self.args.device)
        outputs = self.encoder(input_ids, attention_mask)
        last_hidden_output = outputs[0]  # [batch, seq, hidden], for attention
        pooled_output = outputs[1]  # [batch, hidden], for intent number prediction
        return pooled_output, last_hidden_output, attention_mask