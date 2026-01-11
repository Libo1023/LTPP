import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.decoder import TimeConvTransE


class RecurrentRGCN(nn.Module):
    def __init__(self, decoder_name, encoder_name, num_ents, num_rels, num_static_rels, num_words, num_times, time_interval, h_dim, opn, history_rate, sequence_len, num_bases=-1, num_basis=-1,
                 num_hidden_layers=1, dropout=0, self_loop=False, skip_connect=False, layer_norm=False, input_dropout=0,
                 hidden_dropout=0, feat_dropout=0, aggregation='cat', weight=1, discount=0, angle=0, use_static=False,
                 entity_prediction=False, relation_prediction=False, use_cuda=False,
                 gpu = 0, analysis=False):
        super(RecurrentRGCN, self).__init__()

        self.num_rels = num_rels
        self.num_ents = num_ents
        self.num_times = num_times
        self.time_interval = time_interval
        self.h_dim = h_dim
        self.layer_norm = layer_norm
        self.use_cuda = use_cuda
        
        # Set proper device
        if torch.backends.mps.is_available():
            print("MPS detected, but using CPU/CUDA instead.")
            if torch.cuda.is_available():
                self.gpu = torch.device("cuda")
            else:
                self.gpu = torch.device("cpu")
        elif torch.cuda.is_available():
            self.gpu = torch.device("cuda")
        else:
            self.gpu = torch.device("cpu")

        # Randomly initialized embeddings (no GNN/RNN processing)
        self.entity_embeddings = torch.nn.Parameter(torch.Tensor(num_ents, h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.entity_embeddings)

        self.relation_embeddings = torch.nn.Parameter(torch.Tensor(num_rels * 2, h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.relation_embeddings)

        # Time embedding parameters
        self.weight_t1 = nn.parameter.Parameter(torch.randn(1, h_dim))
        self.bias_t1 = nn.parameter.Parameter(torch.randn(1, h_dim))
        self.weight_t2 = nn.parameter.Parameter(torch.randn(1, h_dim))
        self.bias_t2 = nn.parameter.Parameter(torch.randn(1, h_dim))

        # Only TimeConvTransE decoder for entity prediction
        if decoder_name == "timeconvtranse":
            self.decoder = TimeConvTransE(num_ents, h_dim, input_dropout, hidden_dropout, feat_dropout)
        else:
            raise NotImplementedError("Only timeconvtranse decoder is supported")

        self.loss_e = torch.nn.CrossEntropyLoss()

    def get_init_time(self, quadrupleList):
        """Generate time embeddings"""
        device = self.weight_t1.device
        T_idx = quadrupleList[:, 3] // self.time_interval
        T_idx = T_idx.unsqueeze(1).float().to(device)
        t1 = self.weight_t1 * T_idx + self.bias_t1
        t2 = torch.sin(self.weight_t2 * T_idx + self.bias_t2)
        return t1, t2

    def predict(self, test_graph, num_rels, static_graph, test_triplets, entity_history_vocabulary, rel_history_vocabulary, use_cuda):
        """Prediction using only randomly initialized embeddings"""
        self.use_cuda = use_cuda
        with torch.no_grad():
            inverse_test_triplets = test_triplets[:, [2, 1, 0, 3]]
            inverse_test_triplets[:, 1] = inverse_test_triplets[:, 1] + num_rels
            all_triples = torch.cat((test_triplets, inverse_test_triplets))
            
            # Use random embeddings directly (no GNN/RNN processing)
            embedding = F.normalize(self.entity_embeddings) if self.layer_norm else self.entity_embeddings
            time_embs = self.get_init_time(all_triples)

            # Entity prediction only
            scores_ob = self.decoder.forward(embedding, self.relation_embeddings, time_embs, all_triples).view(-1, self.num_ents)
            score = F.softmax(scores_ob, dim=1)
            score = torch.log(score)

            # Return dummy relation scores (not used)
            dummy_rel_score = torch.zeros_like(score)
            return all_triples, score, dummy_rel_score

    def get_loss(self, glist, triples, static_graph, entity_history_vocabulary, rel_history_vocabulary, use_cuda):
        """Compute loss using only randomly initialized embeddings"""
        self.use_cuda = use_cuda
        loss_ent = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)

        inverse_triples = triples[:, [2, 1, 0, 3]]
        inverse_triples[:, 1] = inverse_triples[:, 1] + self.num_rels
        all_triples = torch.cat([triples, inverse_triples])
        all_triples = all_triples.to(self.gpu)

        # Use random embeddings directly (no GNN/RNN processing)
        embedding = F.normalize(self.entity_embeddings) if self.layer_norm else self.entity_embeddings
        time_embs = self.get_init_time(all_triples)

        # Entity prediction only
        scores_ob = self.decoder.forward(embedding, self.relation_embeddings, time_embs, all_triples).view(-1, self.num_ents)
        scores_en = torch.log(F.softmax(scores_ob, dim=1))
        loss_ent = F.nll_loss(scores_en, all_triples[:, 2])

        # Return zero losses for unused components
        loss_rel = torch.zeros(1).to(self.gpu) if use_cuda else torch.zeros(1)
        loss_static = torch.zeros(1).to(self.gpu) if use_cuda else torch.zeros(1)
        
        return loss_ent, loss_rel, loss_static