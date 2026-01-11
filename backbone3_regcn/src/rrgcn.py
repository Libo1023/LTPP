import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from rgcn.layers import UnionRGCNLayer, RGCNBlockLayer
from src.model import BaseRGCN
from src.decoder import ConvTransE


class RGCNCell(BaseRGCN):
    def build_hidden_layer(self, idx):
        act = F.rrelu
        if idx:
            self.num_basis = 0
        print("activate function: {}".format(act))
        if self.skip_connect:
            sc = False if idx == 0 else True
        else:
            sc = False
        if self.encoder_name == "uvrgcn":
            return UnionRGCNLayer(self.h_dim, self.h_dim, self.num_rels, self.num_bases,
                             activation=act, dropout=self.dropout, self_loop=self.self_loop, 
                             skip_connect=sc, rel_emb=self.rel_emb)
        else:
            raise NotImplementedError

    def forward(self, g, init_ent_emb, init_rel_emb):
        if self.encoder_name == "uvrgcn":
            node_id = g.ndata['id'].squeeze()
            g.ndata['h'] = init_ent_emb[node_id]
            x, r = init_ent_emb, init_rel_emb
            for i, layer in enumerate(self.layers):
                layer(g, [], r[i])
            return g.ndata.pop('h')
        else:
            if self.features is not None:
                print("----------------Feature is not None, Attention ------------")
                g.ndata['id'] = self.features
            node_id = g.ndata['id'].squeeze()
            g.ndata['h'] = init_ent_emb[node_id]
            if self.skip_connect:
                prev_h = []
                for layer in self.layers:
                    prev_h = layer(g, prev_h)
            else:
                for layer in self.layers:
                    layer(g, [])
            return g.ndata.pop('h')


class RecurrentRGCN(nn.Module):
    """RE-GCN: Recurrent Evolution GCN for Temporal Knowledge Graph Reasoning"""
    def __init__(self, decoder_name, encoder_name, num_ents, num_rels, num_static_rels, 
                 num_words, h_dim, opn, sequence_len, num_bases=-1, num_basis=-1,
                 num_hidden_layers=1, dropout=0, self_loop=False, skip_connect=False, 
                 layer_norm=False, input_dropout=0, hidden_dropout=0, feat_dropout=0, 
                 aggregation='cat', weight=1, discount=0, angle=0, use_static=False,
                 entity_prediction=False, relation_prediction=False, use_cuda=False,
                 gpu=0, analysis=False):
        super(RecurrentRGCN, self).__init__()

        self.decoder_name = decoder_name
        self.encoder_name = encoder_name
        self.num_rels = num_rels
        self.num_ents = num_ents
        self.opn = opn
        self.num_words = num_words
        self.num_static_rels = num_static_rels
        self.sequence_len = sequence_len
        self.h_dim = h_dim
        self.layer_norm = layer_norm
        self.h = None
        self.run_analysis = analysis
        self.aggregation = aggregation
        self.weight = weight
        self.discount = discount
        self.use_static = use_static
        self.angle = angle
        self.entity_prediction = entity_prediction
        self.emb_rel = None
        self.gpu = gpu

        # Initialize relation embeddings
        self.emb_rel = torch.nn.Parameter(torch.Tensor(self.num_rels * 2, self.h_dim), 
                                          requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.emb_rel)

        # Initialize entity embeddings
        self.dynamic_emb = torch.nn.Parameter(torch.Tensor(num_ents, h_dim), 
                                              requires_grad=True).float()
        torch.nn.init.normal_(self.dynamic_emb)

        # Static graph components
        if self.use_static:
            self.words_emb = torch.nn.Parameter(torch.Tensor(self.num_words, h_dim), 
                                                requires_grad=True).float()
            torch.nn.init.xavier_normal_(self.words_emb)
            self.statci_rgcn_layer = RGCNBlockLayer(self.h_dim, self.h_dim, 
                                                    self.num_static_rels*2, num_bases,
                                                    activation=F.rrelu, dropout=dropout, 
                                                    self_loop=False, skip_connect=False)
            self.static_loss = torch.nn.MSELoss()

        self.loss_e = torch.nn.CrossEntropyLoss()

        # RE-GCN cell
        self.rgcn = RGCNCell(num_ents,
                             h_dim,
                             h_dim,
                             num_rels * 2,
                             num_bases,
                             num_basis,
                             num_hidden_layers,
                             dropout,
                             self_loop,
                             skip_connect,
                             encoder_name,
                             self.opn,
                             self.emb_rel,
                             use_cuda,
                             analysis)

        # Time gate for entity evolution (Equation 4 in RE-GCN paper)
        self.time_gate_weight = nn.Parameter(torch.Tensor(h_dim, h_dim))
        nn.init.xavier_uniform_(self.time_gate_weight, gain=nn.init.calculate_gain('relu'))
        self.time_gate_bias = nn.Parameter(torch.Tensor(h_dim))
        nn.init.zeros_(self.time_gate_bias)

        # GRU for relation evolution
        self.relation_cell_1 = nn.GRUCell(self.h_dim * 2, self.h_dim)

        # Decoder
        if decoder_name == "convtranse":
            self.decoder_ob = ConvTransE(num_ents, h_dim, input_dropout, 
                                        hidden_dropout, feat_dropout)
        else:
            raise NotImplementedError

    def forward(self, g_list, static_graph, use_cuda):
        # Determine device
        if use_cuda and torch.cuda.is_available():
            device = torch.device('cuda:{}'.format(self.gpu))
        else:
            device = torch.device('cpu')

        # Initialize entity embeddings
        if self.use_static:
            static_graph = static_graph.to(device)
            static_graph.ndata['h'] = torch.cat((self.dynamic_emb, self.words_emb), dim=0)
            self.statci_rgcn_layer(static_graph, [])
            static_emb = static_graph.ndata.pop('h')[:self.num_ents, :]
            static_emb = F.normalize(static_emb) if self.layer_norm else static_emb
            self.h = static_emb
        else:
            self.h = F.normalize(self.dynamic_emb) if self.layer_norm else self.dynamic_emb[:, :]
            static_emb = None

        history_embs = []

        # Process each graph in the sequence
        for i, g in enumerate(g_list):
            g = g.to(device)
            
            # Prepare relation input for GRU
            temp_e = self.h[g.r_to_e]
            x_input = torch.zeros(self.num_rels * 2, self.h_dim).float().to(device)
            
            for span, r_idx in zip(g.r_len, g.uniq_r):
                x = temp_e[span[0]:span[1], :]
                x_mean = torch.mean(x, dim=0, keepdim=True)
                x_input[r_idx] = x_mean
            
            # Update relation embeddings with GRU
            if i == 0:
                x_input = torch.cat((self.emb_rel, x_input), dim=1)
                self.h_0 = self.relation_cell_1(x_input, self.emb_rel)
                self.h_0 = F.normalize(self.h_0) if self.layer_norm else self.h_0
            else:
                x_input = torch.cat((self.emb_rel, x_input), dim=1)
                self.h_0 = self.relation_cell_1(x_input, self.h_0)
                self.h_0 = F.normalize(self.h_0) if self.layer_norm else self.h_0
            
            # Apply relation-aware GCN
            current_h = self.rgcn.forward(g, self.h, [self.h_0, self.h_0])
            current_h = F.normalize(current_h) if self.layer_norm else current_h
            
            # Apply time gate (Equation 4 in RE-GCN paper)
            time_weight = F.sigmoid(torch.mm(self.h, self.time_gate_weight) + self.time_gate_bias)
            self.h = time_weight * current_h + (1 - time_weight) * self.h
            
            history_embs.append(self.h)
        
        return history_embs, static_emb, self.h_0

    def predict(self, test_graph, num_rels, static_graph, test_triplets, use_cuda):
        with torch.no_grad():
            # Add inverse relations
            inverse_test_triplets = test_triplets[:, [2, 1, 0, 3]]
            inverse_test_triplets[:, 1] = inverse_test_triplets[:, 1] + num_rels
            all_triples = torch.cat((test_triplets, inverse_test_triplets))
            
            # Forward pass
            evolve_embs, _, r_emb = self.forward(test_graph, static_graph, use_cuda)
            embedding = F.normalize(evolve_embs[-1]) if self.layer_norm else evolve_embs[-1]

            # Decode
            score = self.decoder_ob.forward(embedding, r_emb, all_triples, mode="test")
            
            return all_triples, score

    def get_loss(self, glist, triples, static_graph, use_cuda):
        # Determine device
        if use_cuda and torch.cuda.is_available():
            device = torch.device('cuda:{}'.format(self.gpu))
        else:
            device = torch.device('cpu')
            
        loss_ent = torch.zeros(1).to(device)
        loss_static = torch.zeros(1).to(device)

        # Add inverse triples
        inverse_triples = triples[:, [2, 1, 0, 3]]
        inverse_triples[:, 1] = inverse_triples[:, 1] + self.num_rels
        all_triples = torch.cat([triples, inverse_triples])
        all_triples = all_triples.to(device)

        # Forward pass
        evolve_embs, static_emb, r_emb = self.forward(glist, static_graph, use_cuda)
        pre_emb = F.normalize(evolve_embs[-1]) if self.layer_norm else evolve_embs[-1]

        # Entity prediction loss
        if self.entity_prediction:
            scores_ob = self.decoder_ob.forward(pre_emb, r_emb, all_triples).view(-1, self.num_ents)
            loss_ent += self.loss_e(scores_ob, all_triples[:, 2])

        # Static graph constraint loss
        if self.use_static:
            if self.discount == 1:
                for time_step, evolve_emb in enumerate(evolve_embs):
                    step = (self.angle * math.pi / 180) * (time_step + 1)
                    if self.layer_norm:
                        sim_matrix = torch.sum(static_emb * F.normalize(evolve_emb), dim=1)
                    else:
                        sim_matrix = torch.sum(static_emb * evolve_emb, dim=1)
                        c = torch.norm(static_emb, p=2, dim=1) * torch.norm(evolve_emb, p=2, dim=1)
                        sim_matrix = sim_matrix / c
                    mask = (math.cos(step) - sim_matrix) > 0
                    loss_static += self.weight * torch.sum(
                        torch.masked_select(math.cos(step) - sim_matrix, mask))
            elif self.discount == 0:
                for time_step, evolve_emb in enumerate(evolve_embs):
                    step = (self.angle * math.pi / 180)
                    if self.layer_norm:
                        sim_matrix = torch.sum(static_emb * F.normalize(evolve_emb), dim=1)
                    else:
                        sim_matrix = torch.sum(static_emb * evolve_emb, dim=1)
                        c = torch.norm(static_emb, p=2, dim=1) * torch.norm(evolve_emb, p=2, dim=1)
                        sim_matrix = sim_matrix / c
                    mask = (math.cos(step) - sim_matrix) > 0
                    loss_static += self.weight * torch.sum(
                        torch.masked_select(math.cos(step) - sim_matrix, mask))
        
        # No relation prediction in RE-GCN
        loss_rel = torch.zeros(1).to(device)
        
        return loss_ent, loss_rel, loss_static