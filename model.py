import pdb
from copy import deepcopy
from transformers import BertForMaskedLM
import torch.nn as nn
import torch
from torch.nn import functional as F
from torch.autograd import Variable
from transformers import BertForMaskedLM

def variable(t: torch.Tensor, use_cuda=True, **kwargs):
    if torch.cuda.is_available() and use_cuda:
        t = t.cuda()
    return Variable(t, **kwargs)
    
class BERT_debias(nn.Module):
    def __init__(self, model_name, cfg, args, dataloader, lines, labels, tokenizer):
        super().__init__()
        self.bert_mlm = BertForMaskedLM.from_pretrained(pretrained_model_name_or_path=model_name, config=cfg)
        self.biased_model = BertForMaskedLM.from_pretrained('bert-base-uncased')
        self.biased_model.cuda()
        self.biased_params = {n: p for n, p in self.biased_model.named_parameters() if p.requires_grad}
        self._biased_means = {}
        self.data_loader = dataloader
        self.lines = lines
        self.labels = labels
        self.tokenizer = tokenizer
        self.device = args.device
        self._precision_matrices = self._diag_fisher()
        self.args = args

        for n, p in deepcopy(self.biased_params).items():
            self._biased_means[n] = variable(p.data)
        
    def forward(self,
                args=None,
                input_ids=None,
                attention_mask=None,
                labels=None,
                token_type_ids=None,
                debias_label_ids=None,
                gender_vector=None
                ):

        output = self.bert_mlm(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                              labels=labels)

        # output.hidden_states[-1].shape (8, 164, 768)
        if debias_label_ids is not None:
            final_hidden = output.hidden_states[-1]

            targets = debias_label_ids.unsqueeze(2) * final_hidden

            # absolute ver
            if self.args.orth_loss_ver == "abs":
                orthogonal_loss = torch.sum(torch.abs(torch.matmul(targets, gender_vector)))

            # squared ver
            elif self.args.orth_loss_ver == "square":
                orthogonal_loss = torch.sum(torch.square(torch.matmul(targets, gender_vector)))

            return output.loss, orthogonal_loss

        if debias_label_ids is None:
            return output

    def save_pretrained(self, output_dir):
        self.bert_mlm.save_pretrained(output_dir)

    def _diag_fisher(self):
        precision_matrices = {}
        for n, p in deepcopy(self.biased_params).items():
            p.data.zero_()
            precision_matrices[n] = variable(p.data)

        self.biased_model.eval()

        for idx, line in enumerate(self.lines):

            line_output = []
            # read the line and its label
            line = self.lines[idx]
            label = self.labels[idx][0]
            label_anti = self.labels[idx][1]

            if label.lower() not in ('she', 'her'):
                male_label = label
                female_label = label_anti
                g_index = 1
            else:
                male_label = label_anti
                female_label = label
                g_index = 0

            comparison_labels = [male_label, female_label]

            comparison_indices = self.tokenizer.convert_tokens_to_ids(comparison_labels)

            # tokenise the line
            input_ids = torch.tensor(self.tokenizer.encode(line)).unsqueeze(0)  # Batch size 1
            input_ids = input_ids.to(self.device)

            outputs = self.biased_model(input_ids, labels=input_ids)
            loss = outputs[0]
            loss.backward()

            for n, p in self.biased_model.named_parameters():
                precision_matrices[n].data += p.grad.data ** 2 / len(self.lines)

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self):
        loss = 0
        for n, p in self.bert_mlm.named_parameters():
            _loss = self._precision_matrices[n] * (p - self._biased_means[n]) ** 2
            loss += _loss.sum()
        return loss