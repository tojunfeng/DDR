import os
import torch
import logging
from dataclasses import dataclass
from transformers import AutoModel
from transformers.modeling_outputs import ModelOutput
from transformers.hf_argparser import HfArgumentParser
from utils import select_grouped_indices


@dataclass
class BiencoderModelOutput(ModelOutput):
    loss: torch.FloatTensor = None
    scores: torch.FloatTensor = None
    labels: torch.LongTensor = None
    q_embedding: torch.FloatTensor = None
    p_embedding: torch.FloatTensor = None


class QueryModel(torch.nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.Cast_nn = torch.nn.Linear(model.config.hidden_size, model.config.hidden_size)

        torch.nn.init.xavier_normal_(self.Cast_nn.weight)
    
    def forward(self, inputs):
        M_embedding :torch.Tensor = self.model(**inputs).last_hidden_state[:, 0]
        M_embedding = M_embedding.double()

        first = torch.ones(inputs['input_ids'].size(0), 1, dtype=M_embedding.dtype).to(inputs['input_ids'].device)
        second = M_embedding*M_embedding
        third = M_embedding
        forth = torch.ones_like(first, dtype=M_embedding.dtype).to(inputs['input_ids'].device)

        embedding = torch.cat((first, second, third, forth), dim=-1)
        return {
            'q_embedding': embedding
        }


class PassageModel(torch.nn.Module):

    def __init__(self, model, beta=1):
        super().__init__()
        self.model = model
        self.beta = torch.tensor(beta, dtype=torch.float, device=model.device)
        self.var_hidden_size = model.config.hidden_size

        self.Cast_nn_mean = torch.nn.Linear(model.config.hidden_size, self.var_hidden_size)

        self.Q = torch.nn.Linear(model.config.hidden_size, model.config.hidden_size)
        self.K = torch.nn.Linear(model.config.hidden_size, model.config.hidden_size)
        self.V = torch.nn.Linear(model.config.hidden_size, self.var_hidden_size)

        torch.nn.init.xavier_normal_(self.Q.weight)
        torch.nn.init.xavier_normal_(self.K.weight)
        torch.nn.init.xavier_normal_(self.V.weight)
        torch.nn.init.xavier_normal_(self.Cast_nn_mean.weight)
        
    def get_Var(self, embedding, attention_mask):
        # embedding: [batch_size, seq_len, hidden_size]
        # attention_mask: [batch_size, seq_len]
        # return: [batch_size, hidden_size]
        q_embedding = self.Q(embedding[:, 0]) # [batch_size, hidden_size]
        k_embedding = self.K(embedding)
        v_embedding = self.V(embedding)
        attention = torch.einsum('bh,bsh->bs', q_embedding, k_embedding) / (self.model.config.hidden_size ** 0.5)
        attention = attention.masked_fill(~attention_mask, float('-inf'))
        attention = torch.nn.functional.softmax(attention, dim=-1)
        var_embedding = torch.einsum('bs,bsh->bh', attention, v_embedding)
        var_embedding = var_embedding.double()
        mask = var_embedding < 10
        var_embedding[mask] = torch.log1p(torch.exp(self.beta*var_embedding[mask]))/self.beta

        return var_embedding


    def forward(self, inputs):
        embedding = self.model(**inputs).last_hidden_state
        M_embedding = embedding[:, 0].double()
        # Pad token is ignored
        attention_mask = inputs['input_ids'].ne(self.model.config.pad_token_id)
        Var_embedding = self.get_Var(embedding, attention_mask)

        first = -1/2*torch.sum(torch.log(Var_embedding), dim=-1).reshape(-1, 1)
        second = -1/2/Var_embedding
        third =  M_embedding/Var_embedding
        fourth = -1/2*torch.sum(M_embedding*M_embedding/Var_embedding, dim=-1).reshape(-1, 1)

        embedding = torch.cat((first, second, third, fourth), dim=-1)
        return {
            'p_embedding': embedding
        }
    

class BiencodeModel(torch.nn.Module):
    '''Biencoder model'''

    def __init__(self, args: HfArgumentParser, q_model, p_model):
        super().__init__()
        self.args = args
        self.q_model = q_model
        self.p_model = p_model
        self.kd_loss_fn = torch.nn.KLDivLoss(reduction='batchmean', log_target=True)
        

    def forward(self, query, passage, kl_label):
        '''Forward pass'''
        q_embedding = self.q_model(query)['q_embedding']
        p_embedding = self.p_model(passage)['p_embedding']
        scores = torch.matmul(q_embedding, p_embedding.t())
        labels = torch.arange(scores.size(0)).to(scores.device) * (self.args.num_negatives + 1)

        group_index = select_grouped_indices(scores, self.args.num_negatives + 1, start=0)
        group_score = torch.gather(scores, dim=1, index=group_index)

        group_score = torch.log_softmax(group_score, dim=1)
        target_score = torch.log_softmax(kl_label, dim=1)
        loss = self.kd_loss_fn(input=group_score, target=target_score)
            
        return BiencoderModelOutput(
            loss=loss,
            scores=scores,
            labels=labels,
            q_embedding=q_embedding,
            p_embedding=p_embedding
        )
    
    
    @classmethod
    def build(cls, args: HfArgumentParser, from_checkpoint=False):
        q_model = AutoModel.from_pretrained(args.model_path)
        p_model = AutoModel.from_pretrained(args.model_path)
        q_model = QueryModel(q_model)
        p_model = PassageModel(p_model)
        ret = cls(args, q_model, p_model)
        checkpoint_file_name = os.path.join(args.checkpoint, 'pytorch_model.bin')
        if from_checkpoint:
            logging.info(f'Loading checkpoint from {checkpoint_file_name}')
            ret.load_state_dict(torch.load(checkpoint_file_name))
        else:
            try:
                ret.load_state_dict(torch.load(checkpoint_file_name))
                logging.info(f'Loading checkpoint from {checkpoint_file_name}')
            except:
                logging.info(f'init model from {args.model_path}')
        return ret


