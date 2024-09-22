from typing import Optional
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F

def average_pool(last_hidden_states: Tensor,
                 attention_mask: Optional[torch.Tensor] = None) -> Tensor:
    if attention_mask is None:
        bz, seqlen = last_hidden_states.size()[:2]
        attention_mask = torch.ones(bz, seqlen, device=last_hidden_states.device)
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

class mDPRBase(nn.Module):
    def __init__(self, base_encoder, args):
        super(mDPRBase, self).__init__()
        self.base_encoder = base_encoder
        self.args = args
        self.is_ance = args.base_model_name == "castorini/ance-msmarco-passage"
        self.labels = torch.arange(args.batch_size, dtype=torch.long, device=args.device)
        self.use_pooler = args.use_pooler
        self.normalize = args.normalize
        self.temperature = args.temperature
        self.aggregation = args.aggregation
        self.loss_fct = nn.CrossEntropyLoss()

    def match(self, q_ids, q_mask, d_ids, d_mask):
        q_reps = self.query(q_ids, q_mask)
        d_reps = self.doc(d_ids, d_mask)
        scores = self.score(q_reps, d_reps)
        loss = self.loss_fct(scores, self.labels[:scores.size(0)])
        return loss
    
    def feature(self, input_ids, attention_mask=None, return_pooler=True, aggregation='cls'):
        enc_reps = self.base_encoder(input_ids, attention_mask=attention_mask)
        if self.is_ance:
            enc_output = enc_reps
        elif return_pooler:
            enc_output = enc_reps.pooler_output
        elif aggregation == 'avg_pool':
            enc_output = average_pool(enc_reps.last_hidden_state, attention_mask)
        elif aggregation == 'cls':
            enc_output = enc_reps.last_hidden_state[:, 0, :]
        else:
            raise ValueError(f"unknown aggregation method {aggregation}")
        return enc_output
    
    def post_process(self, encoder_output):
        return nn.functional.normalize(encoder_output, dim=-1) * self.temperature if self.normalize else encoder_output * self.temperature
    
    def query(self, input_ids, attention_mask):
        enc_output = self.feature(input_ids, attention_mask=attention_mask, return_pooler=self.use_pooler, aggregation=self.aggregation)
        return self.post_process(enc_output)

    def doc(self, input_ids, attention_mask):
        return self.query(input_ids, attention_mask)

    def score(self, query_reps, document_reps):
        return query_reps.mm(document_reps.t())

    def forward(self, q_ids, q_mask, d_ids, d_mask):
        return self.match(q_ids, q_mask, d_ids, d_mask)

    def save(self, path):
        state = self.state_dict()
        torch.save(state, path)

    def load(self, path, map_location=None):
        checkpoint = torch.load(path, map_location=map_location)
        if type(checkpoint) is dict:
            self.load_state_dict(checkpoint['model_state_dict'], strict=True)
        else:
            self.load_state_dict(checkpoint, strict=True)

# model wrapper
class mDPR(nn.Module):
    def __init__(self, base_encoder, args):
        super(mDPR, self).__init__()
        self.base_encoder = base_encoder
        self.hidden_size = base_encoder.config.hidden_size
        self.args = args
        self.num_langs = args.num_langs
        
        self.ranking_labels = torch.zeros(self.args.batch_size, 2 * self.args.batch_size * args.num_langs**2, dtype=torch.long, device=args.device, requires_grad=False)
        for i in range(self.args.batch_size):
            self.ranking_labels[i, i*args.num_langs**2:(i+1)*args.num_langs**2] = 1
        
        self.score_dist = torch.ones(2 * self.args.batch_size**2, args.num_langs**2, dtype=torch.float, device=args.device, requires_grad=False)
        self.score_dist = F.softmax(self.score_dist, dim=1)

        self.rank_loss_fct = nn.MultiLabelSoftMarginLoss(reduction='mean')
        self.score_loss_fct = nn.KLDivLoss(reduction='batchmean')

    def match(self, q_ids, q_mask, pos_ids, pos_mask, neg_ids, neg_mask):
        q_reps = self.query(q_ids, q_mask)
        pos_reps = self.doc(pos_ids, pos_mask)
        neg_reps = self.doc(neg_ids, neg_mask)
        pos_mm = self.score(q_reps, pos_reps).unfold(0, self.num_langs, self.num_langs).unfold(1, self.num_langs, self.num_langs)
        neg_mm = self.score(q_reps, neg_reps).unfold(0, self.num_langs, self.num_langs).unfold(1, self.num_langs, self.num_langs)

        score_blocks = torch.cat([pos_mm.flatten(0,1).flatten(1,2), neg_mm.flatten(0,1).flatten(1,2)], dim=0)
        score_blocks = F.log_softmax(score_blocks, dim=1)
        score_loss = self.score_loss_fct(score_blocks, self.score_dist)

        batch_scores = torch.cat([pos_mm.flatten(1,3), neg_mm.flatten(1,3)], dim=1)
        ranking_loss = self.rank_loss_fct(batch_scores, self.ranking_labels)
        
        return ranking_loss + score_loss, ranking_loss.item(), score_loss.item()

    def query(self, input_ids, attention_mask):
        enc_reps = self.base_encoder(input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        return enc_reps

    def doc(self, input_ids, attention_mask):
        return self.query(input_ids, attention_mask)

    def score(self, query_reps, document_reps):
        return query_reps.mm(document_reps.t())
        

    def forward(self, q_ids, q_mask, pos_ids, pos_mask, neg_ids, neg_mask):
        return self.match(q_ids, q_mask, pos_ids, pos_mask, neg_ids, neg_mask)

    def save(self, path):
        state = self.state_dict()
        torch.save(state, path)

    def load(self, path, map_location=None):
        checkpoint = torch.load(path, map_location=map_location)
        if type(checkpoint) is dict:
            self.load_state_dict(checkpoint['model_state_dict'], strict=True)
        else:
            self.load_state_dict(checkpoint, strict=True)


#######################################################################################################################################
# model wrapper
def split_matrix(matrix):
    diagonal = torch.diagonal(matrix).permute(2, 0, 1).flatten(1, 2) # Get the diagonal elements
    non_diagonal = matrix[~torch.eye(matrix.size(0), dtype=bool)] # Get the non-diagonal elements
    return diagonal, non_diagonal

class m2mDPR(nn.Module):
    def __init__(self, base_encoder, args):
        super(m2mDPR, self).__init__()
        self.base_encoder = base_encoder
        self.hidden_size = base_encoder.config.hidden_size
        self.args = args
        self.num_langs = args.num_langs
        self.use_pooler = args.use_pooler
        self.normalize = args.normalize
        self.temperature = args.temperature
        
        self.ranking_labels = torch.zeros(args.batch_size * args.num_langs**2, dtype=torch.long, device=args.device, requires_grad=False)
        
        self.score_dist = torch.ones(args.batch_size**2 + args.batch_size**2 * (args.train_n_passages-1), args.num_langs**2, dtype=torch.float, device=args.device, requires_grad=False)
        self.score_dist = F.softmax(self.score_dist, dim=1)

        self.rank_loss_fct = nn.CrossEntropyLoss()
        self.score_loss_fct = nn.KLDivLoss(reduction='batchmean')
        
    def match(self, q_ids, q_mask, pos_ids, pos_mask, neg_ids, neg_mask):
        q_reps = self.query(q_ids, q_mask)
        pos_reps = self.doc(pos_ids, pos_mask)
        neg_reps = self.doc(neg_ids, neg_mask)
        pos_mm = self.score(q_reps, pos_reps).unfold(0, self.num_langs, self.num_langs).unfold(1, self.num_langs, self.num_langs)
        neg_mm = self.score(q_reps, neg_reps).unfold(0, self.num_langs, self.num_langs).unfold(1, self.num_langs, self.num_langs)

        score_blocks = torch.cat([pos_mm.flatten(0,1).flatten(1,2), neg_mm.flatten(0,1).flatten(1,2)], dim=0)
        score_blocks = F.log_softmax(score_blocks, dim=1)
        score_loss = self.score_loss_fct(score_blocks, self.score_dist)

        pos_diag, pos_nondiag = split_matrix(pos_mm)

        pos_nondiag_flat = pos_nondiag.view(pos_mm.size(0), -1)
        pos_scores = pos_diag.view(self.args.batch_size*self.num_langs**2, 1)
        neg_scores = torch.cat([pos_nondiag_flat, neg_mm.flatten(1,3)], dim=1).unsqueeze(1).repeat(1, self.num_langs**2, 1).flatten(0, 1)
        batch_scores = torch.cat([pos_scores, neg_scores], dim=1)
        ranking_loss = self.rank_loss_fct(batch_scores, self.ranking_labels[:batch_scores.size(0)])

        return ranking_loss + 0.1 * score_loss, ranking_loss.item(), score_loss.item()

    def query(self, input_ids, attention_mask):
        enc_reps = self.base_encoder(input_ids, attention_mask=attention_mask)
        enc_output = enc_reps.pooler_output if self.use_pooler else enc_reps.last_hidden_state[:, 0, :]
        return nn.functional.normalize(enc_output, dim=-1) * self.temperature if self.normalize else enc_output * self.temperature
    
    def doc(self, input_ids, attention_mask):
        return self.query(input_ids, attention_mask)

    def score(self, query_reps, document_reps):
        return query_reps.mm(document_reps.t())

    def forward(self, q_ids, q_mask, pos_ids, pos_mask, neg_ids, neg_mask):
        return self.match(q_ids, q_mask, pos_ids, pos_mask, neg_ids, neg_mask)

    def save(self, path):
        state = self.state_dict()
        torch.save(state, path)

    def load(self, path, map_location=None):
        checkpoint = torch.load(path, map_location=map_location)
        if type(checkpoint) is dict:
            self.load_state_dict(checkpoint['model_state_dict'], strict=True)
        else:
            self.load_state_dict(checkpoint, strict=True)


class m2mDPRBase(nn.Module):
    def __init__(self, base_encoder, args):
        super(m2mDPRBase, self).__init__()
        self.base_encoder = base_encoder
        self.hidden_size = base_encoder.config.hidden_size
        self.args = args
        self.num_langs = args.num_langs
        self.use_pooler = args.use_pooler
        
        self.ranking_labels = torch.zeros(self.args.batch_size * args.num_langs**2, dtype=torch.long, device=args.device, requires_grad=False)
        self.rank_loss_fct = nn.CrossEntropyLoss()

    def match(self, q_ids, q_mask, pos_ids, pos_mask, neg_ids, neg_mask):
        q_reps = self.query(q_ids, q_mask)
        pos_reps = self.doc(pos_ids, pos_mask)
        neg_reps = self.doc(neg_ids, neg_mask)
        pos_mm = self.score(q_reps, pos_reps).unfold(0, self.num_langs, self.num_langs).unfold(1, self.num_langs, self.num_langs)
        neg_mm = self.score(q_reps, neg_reps).unfold(0, self.num_langs, self.num_langs).unfold(1, self.num_langs, self.num_langs)

        pos_diag, pos_nondiag = split_matrix(pos_mm)
        pos_scores = pos_diag.view(self.args.batch_size*self.num_langs**2, 1)
        neg_scores = torch.cat([pos_nondiag, neg_mm.flatten(1,3)], dim=1).unsqueeze(1).repeat(1, self.num_langs**2, 1).flatten(0, 1)
        batch_scores = torch.cat([pos_scores, neg_scores], dim=1)
        ranking_loss = self.rank_loss_fct(batch_scores, self.ranking_labels)

        return ranking_loss

    def query(self, input_ids, attention_mask):
        enc_reps = self.base_encoder(input_ids, attention_mask=attention_mask)
        return enc_reps.pooler_output if self.use_pooler else enc_reps.last_hidden_state[:, 0, :]

    def doc(self, input_ids, attention_mask):
        return self.query(input_ids, attention_mask)

    def score(self, query_reps, document_reps):
        return query_reps.mm(document_reps.t())
        

    def forward(self, q_ids, q_mask, pos_ids, pos_mask, neg_ids, neg_mask):
        return self.match(q_ids, q_mask, pos_ids, pos_mask, neg_ids, neg_mask)

    def save(self, path):
        state = self.state_dict()
        torch.save(state, path)

    def load(self, path, map_location=None):
        checkpoint = torch.load(path, map_location=map_location)
        if type(checkpoint) is dict:
            self.load_state_dict(checkpoint['model_state_dict'], strict=True)
        else:
            self.load_state_dict(checkpoint, strict=True)

