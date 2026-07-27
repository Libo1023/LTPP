from __future__ import absolute_import, print_function
import argparse
import math
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from collections import defaultdict
from datetime import datetime
from tqdm import tqdm
import dgl
import dgl.function as fn
from scipy import stats

np.random.seed(123)
torch.manual_seed(123)

# ============================================================================
# PART 1: DATA LOADING
# ============================================================================

class RGCNLinkDataset:
    def __init__(self, name, dir=None):
        self.name = name
        self.dir = os.path.join(dir, self.name) if dir else self.name

    def load(self, load_time=True):
        entity_path = os.path.join(self.dir, 'entity2id.txt')
        relation_path = os.path.join(self.dir, 'relation2id.txt')
        train_path = os.path.join(self.dir, 'train.txt')
        valid_path = os.path.join(self.dir, 'valid.txt')
        test_path = os.path.join(self.dir, 'test.txt')
        entity_dict = self._read_dictionary(entity_path)
        relation_dict = self._read_dictionary(relation_path)
        self.train = np.array(self._read_triplets_as_list(train_path, load_time))
        self.valid = np.array(self._read_triplets_as_list(valid_path, load_time))
        self.test = np.array(self._read_triplets_as_list(test_path, load_time))
        self.num_nodes = len(entity_dict)
        self.num_rels = len(relation_dict)
        print(f"Entities: {self.num_nodes}, Relations: {self.num_rels}")
        print(f"Train: {len(self.train)}, Valid: {len(self.valid)}, Test: {len(self.test)}")

    def _read_dictionary(self, filename):
        d = {}
        with open(filename, 'r+', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                d[int(parts[1])] = parts[0]
        return d

    def _read_triplets_as_list(self, filename, load_time):
        l = []
        with open(filename, 'r+', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                s, r, o = int(parts[0]), int(parts[1]), int(parts[2])
                if load_time:
                    l.append([s, r, o, int(parts[3])])
                else:
                    l.append([s, r, o])
        return l


def load_data(dataset):
    data = RGCNLinkDataset(dataset, "./data")
    data.load()
    return data


def split_by_time(data):
    snapshot_list, snapshot, latest_t = [], [], 0
    for i in range(len(data)):
        t = data[i][3]
        if latest_t != t:
            latest_t = t
            if len(snapshot):
                snapshot_list.append(np.array(snapshot).copy())
            snapshot = []
        snapshot.append(data[i][:])
    if len(snapshot) > 0:
        snapshot_list.append(np.array(snapshot).copy())
    times = sorted(set(triple[3] for triple in data))
    return snapshot_list, np.asarray(times)


def r2e(triplets, num_rels):
    src, rel, dst = triplets.transpose()
    uniq_r = np.unique(rel)
    uniq_r = np.concatenate((uniq_r, uniq_r + num_rels))
    r_to_e = defaultdict(set)
    for j, (s, r, d) in enumerate(triplets):
        r_to_e[r].add(s)
        r_to_e[r + num_rels].add(s)
    r_len, e_idx, idx = [], [], 0
    for r in uniq_r:
        r_len.append((idx, idx + len(r_to_e[r])))
        e_idx.extend(list(r_to_e[r]))
        idx += len(r_to_e[r])
    return uniq_r, r_len, e_idx


def build_sub_graph(num_nodes, num_rels, triples):
    def comp_deg_norm(g):
        in_deg = g.in_degrees(range(g.number_of_nodes())).float()
        in_deg[torch.nonzero(in_deg == 0).view(-1)] = 1
        return 1.0 / in_deg
    triples = triples[:, :3]
    src, rel, dst = triples.transpose()
    src, dst = np.concatenate((src, dst)), np.concatenate((dst, src))
    rel = np.concatenate((rel, rel + num_rels))
    g = dgl.DGLGraph()
    g.add_nodes(num_nodes)
    g.add_edges(src, dst)
    norm = comp_deg_norm(g)
    node_id = torch.arange(0, num_nodes, dtype=torch.long).view(-1, 1)
    g.ndata.update({'id': node_id, 'norm': norm.view(-1, 1)})
    g.apply_edges(lambda edges: {'norm': edges.dst['norm'] * edges.src['norm']})
    g.edata['type'] = torch.LongTensor(rel)
    uniq_r, r_len, r_to_e = r2e(triples, num_rels)
    g.uniq_r, g.r_to_e, g.r_len = uniq_r, torch.from_numpy(np.array(r_to_e)).long(), r_len
    return g


def load_all_answers_for_filter(total_data, num_rel):
    def add_subject(e1, e2, r, d, num_rel):
        if e2 not in d: d[e2] = {}
        if r + num_rel not in d[e2]: d[e2][r + num_rel] = set()
        d[e2][r + num_rel].add(e1)
    def add_object(e1, e2, r, d):
        if e1 not in d: d[e1] = {}
        if r not in d[e1]: d[e1][r] = set()
        d[e1][r].add(e2)
    all_ans = {}
    for line in total_data:
        s, r, o = line[:3]
        add_subject(s, o, r, all_ans, num_rel=num_rel)
        add_object(s, o, r, all_ans)
    return all_ans


def load_all_answers_for_time_filter(total_data, num_rels, num_nodes):
    all_ans_list = []
    all_snap, _ = split_by_time(total_data)
    for snap in all_snap:
        all_ans_list.append(load_all_answers_for_filter(snap, num_rels))
    return all_ans_list


# ============================================================================
# PART 2: MODEL LAYERS
# ============================================================================

class RGCNLayer(nn.Module):
    def __init__(self, in_feat, out_feat, bias=None, activation=None, self_loop=False,
                 skip_connect=False, dropout=0.0, layer_norm=False):
        super(RGCNLayer, self).__init__()
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop
        self.skip_connect = skip_connect
        self.layer_norm = layer_norm
        if self.bias:
            self.bias = nn.Parameter(torch.Tensor(out_feat))
            nn.init.xavier_uniform_(self.bias, gain=nn.init.calculate_gain('relu'))
        if self.self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight, gain=nn.init.calculate_gain('relu'))
        if self.skip_connect:
            self.skip_connect_weight = nn.Parameter(torch.Tensor(out_feat, out_feat))
            nn.init.xavier_uniform_(self.skip_connect_weight, gain=nn.init.calculate_gain('relu'))
            self.skip_connect_bias = nn.Parameter(torch.Tensor(out_feat))
            nn.init.zeros_(self.skip_connect_bias)
        self.dropout = nn.Dropout(dropout) if dropout else None
        if self.layer_norm:
            self.normalization_layer = nn.LayerNorm(out_feat, elementwise_affine=False)

    def propagate(self, g): raise NotImplementedError

    def forward(self, g, prev_h=[]):
        if self.self_loop:
            loop_message = torch.mm(g.ndata['h'], self.loop_weight)
            if self.dropout: loop_message = self.dropout(loop_message)
        if len(prev_h) != 0 and self.skip_connect:
            skip_weight = torch.sigmoid(torch.mm(prev_h, self.skip_connect_weight) + self.skip_connect_bias)
        self.propagate(g)
        node_repr = g.ndata['h']
        if self.bias: node_repr = node_repr + self.bias
        if len(prev_h) != 0 and self.skip_connect:
            prev_repr = (1 - skip_weight) * prev_h
            if self.activation: node_repr = self.activation(node_repr)
            if self.self_loop:
                loop_message = skip_weight * (self.activation(loop_message) if self.activation else loop_message)
                node_repr = node_repr + loop_message
            node_repr = node_repr + prev_repr
        else:
            if self.self_loop: node_repr = node_repr + loop_message
            if self.layer_norm: node_repr = self.normalization_layer(node_repr)
            if self.activation: node_repr = self.activation(node_repr)
        g.ndata['h'] = node_repr
        return node_repr


class RGCNBlockLayer(RGCNLayer):
    def __init__(self, in_feat, out_feat, num_rels, num_bases, bias=None, activation=None,
                 self_loop=False, dropout=0.0, skip_connect=False, layer_norm=False):
        super(RGCNBlockLayer, self).__init__(in_feat, out_feat, bias, activation,
                                             self_loop=self_loop, skip_connect=skip_connect, dropout=dropout)
        self.num_rels, self.num_bases = num_rels, num_bases
        assert self.num_bases > 0
        self.out_feat = out_feat
        self.submat_in = in_feat // self.num_bases
        self.submat_out = out_feat // self.num_bases
        self.weight = nn.Parameter(torch.Tensor(self.num_rels, self.num_bases * self.submat_in * self.submat_out))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

    def msg_func(self, edges):
        weight = self.weight.index_select(0, edges.data['type']).view(-1, self.submat_in, self.submat_out)
        node = edges.src['h'].view(-1, 1, self.submat_in)
        return {'msg': torch.bmm(node, weight).view(-1, self.out_feat)}

    def propagate(self, g):
        g.update_all(self.msg_func, fn.sum(msg='msg', out='h'), self.apply_func)

    def apply_func(self, nodes):
        return {'h': nodes.data['h'] * nodes.data['norm']}


class UnionRGCNLayer(nn.Module):
    def __init__(self, in_feat, out_feat, num_rels, num_bases=-1, bias=None, activation=None,
                 self_loop=False, dropout=0.0, skip_connect=False, rel_emb=None):
        super(UnionRGCNLayer, self).__init__()
        self.in_feat, self.out_feat = in_feat, out_feat
        self.bias, self.activation = bias, activation
        self.self_loop, self.skip_connect = self_loop, skip_connect
        self.num_rels, self.rel_emb = num_rels, None
        self.weight_neighbor = nn.Parameter(torch.Tensor(self.in_feat, self.out_feat))
        nn.init.xavier_uniform_(self.weight_neighbor, gain=nn.init.calculate_gain('relu'))
        if self.self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight, gain=nn.init.calculate_gain('relu'))
            self.evolve_loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.evolve_loop_weight, gain=nn.init.calculate_gain('relu'))
        if self.skip_connect:
            self.skip_connect_weight = nn.Parameter(torch.Tensor(out_feat, out_feat))
            nn.init.xavier_uniform_(self.skip_connect_weight, gain=nn.init.calculate_gain('relu'))
            self.skip_connect_bias = nn.Parameter(torch.Tensor(out_feat))
            nn.init.zeros_(self.skip_connect_bias)
        self.dropout = nn.Dropout(dropout) if dropout else None

    def propagate(self, g):
        g.update_all(lambda x: self.msg_func(x), fn.sum(msg='msg', out='h'), self.apply_func)

    def forward(self, g, prev_h, emb_rel):
        device = torch.device('cpu')
        self.rel_emb = emb_rel
        if self.self_loop:
            masked_index = torch.masked_select(
                torch.arange(0, g.number_of_nodes(), dtype=torch.long, device=device),
                (g.in_degrees(range(g.number_of_nodes())) > 0))
            loop_message = torch.mm(g.ndata['h'], self.evolve_loop_weight)
            loop_message[masked_index, :] = torch.mm(g.ndata['h'], self.loop_weight)[masked_index, :]
        if len(prev_h) != 0 and self.skip_connect:
            skip_weight = torch.sigmoid(torch.mm(prev_h, self.skip_connect_weight) + self.skip_connect_bias)
        self.propagate(g)
        node_repr = g.ndata['h']
        if len(prev_h) != 0 and self.skip_connect:
            if self.self_loop: node_repr = node_repr + loop_message
            node_repr = skip_weight * node_repr + (1 - skip_weight) * prev_h
        else:
            if self.self_loop: node_repr = node_repr + loop_message
        if self.activation: node_repr = self.activation(node_repr)
        if self.dropout: node_repr = self.dropout(node_repr)
        g.ndata['h'] = node_repr
        return node_repr

    def msg_func(self, edges):
        relation = self.rel_emb.index_select(0, edges.data['type']).view(-1, self.out_feat)
        node = edges.src['h'].view(-1, self.out_feat)
        return {'msg': torch.mm(node + relation, self.weight_neighbor)}

    def apply_func(self, nodes):
        return {'h': nodes.data['h'] * nodes.data['norm']}


# ============================================================================
# PART 3: DECODER
# ============================================================================

class ConvTransE(nn.Module):
    def __init__(self, num_entities, embedding_dim, input_dropout=0, hidden_dropout=0,
                 feature_map_dropout=0, channels=50, kernel_size=3):
        super(ConvTransE, self).__init__()
        self.inp_drop = nn.Dropout(input_dropout)
        self.hidden_drop = nn.Dropout(hidden_dropout)
        self.feature_map_drop = nn.Dropout(feature_map_dropout)
        self.conv1 = nn.Conv1d(2, channels, kernel_size, stride=1, padding=int(math.floor(kernel_size / 2)))
        self.bn0 = nn.BatchNorm1d(2)
        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(embedding_dim)
        self.register_parameter('b', Parameter(torch.zeros(num_entities)))
        self.fc = nn.Linear(embedding_dim * channels, embedding_dim)

    def forward(self, embedding, emb_rel, triplets, mode="train"):
        e1_embedded_all = torch.tanh(embedding)
        device = e1_embedded_all.device
        triplets = triplets.to(device)
        batch_size = len(triplets)
        e1_embedded = e1_embedded_all[triplets[:, 0]].unsqueeze(1)
        rel_embedded = emb_rel[triplets[:, 1]].unsqueeze(1)
        stacked_inputs = torch.cat([e1_embedded, rel_embedded], 1)
        stacked_inputs = self.bn0(stacked_inputs)
        x = self.inp_drop(stacked_inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        if batch_size > 1: x = self.bn2(x)
        x = F.relu(x)
        return torch.mm(x, e1_embedded_all.transpose(1, 0))


# ============================================================================
# PART 4: MODEL
# ============================================================================

class BaseRGCN(nn.Module):
    def __init__(self, num_nodes, h_dim, out_dim, num_rels, num_bases=-1, num_basis=-1,
                 num_hidden_layers=1, dropout=0, self_loop=False, skip_connect=False,
                 encoder_name="", opn="sub", rel_emb=None, use_cuda=False, analysis=False):
        super(BaseRGCN, self).__init__()
        self.num_nodes, self.h_dim, self.out_dim = num_nodes, h_dim, out_dim
        self.num_rels, self.num_bases, self.num_basis = num_rels, num_bases, num_basis
        self.num_hidden_layers, self.dropout = num_hidden_layers, dropout
        self.skip_connect, self.self_loop = skip_connect, self_loop
        self.encoder_name, self.use_cuda = encoder_name, use_cuda
        self.rel_emb, self.opn = rel_emb, opn
        self.build_model()
        self.features = self.create_features()

    def build_model(self):
        self.layers = nn.ModuleList()
        i2h = self.build_input_layer()
        if i2h: self.layers.append(i2h)
        for idx in range(self.num_hidden_layers):
            self.layers.append(self.build_hidden_layer(idx))
        h2o = self.build_output_layer()
        if h2o: self.layers.append(h2o)

    def create_features(self): return None
    def build_input_layer(self): return None
    def build_hidden_layer(self, idx): raise NotImplementedError
    def build_output_layer(self): return None


class RGCNCell(BaseRGCN):
    def build_hidden_layer(self, idx):
        act = F.rrelu
        if idx: self.num_basis = 0
        sc = False if idx == 0 or not self.skip_connect else True
        if self.encoder_name == "uvrgcn":
            return UnionRGCNLayer(self.h_dim, self.h_dim, self.num_rels, self.num_bases,
                                  activation=act, dropout=self.dropout, self_loop=self.self_loop,
                                  skip_connect=sc, rel_emb=self.rel_emb)
        raise NotImplementedError

    def forward(self, g, init_ent_emb, init_rel_emb):
        if self.encoder_name == "uvrgcn":
            node_id = g.ndata['id'].squeeze()
            g.ndata['h'] = init_ent_emb[node_id]
            for i, layer in enumerate(self.layers):
                layer(g, [], init_rel_emb[i])
            return g.ndata.pop('h')
        raise NotImplementedError


class RecurrentRGCN(nn.Module):
    def __init__(self, decoder_name, encoder_name, num_ents, num_rels, num_static_rels,
                 num_words, h_dim, opn, sequence_len, num_bases=-1, num_basis=-1,
                 num_hidden_layers=1, dropout=0, self_loop=False, skip_connect=False,
                 layer_norm=False, input_dropout=0, hidden_dropout=0, feat_dropout=0,
                 aggregation='cat', weight=1, discount=0, angle=0, use_static=False,
                 entity_prediction=False, relation_prediction=False, use_cuda=False,
                 gpu=0, analysis=False):
        super(RecurrentRGCN, self).__init__()
        self.decoder_name, self.encoder_name = decoder_name, encoder_name
        self.num_rels, self.num_ents = num_rels, num_ents
        self.opn, self.num_words, self.num_static_rels = opn, num_words, num_static_rels
        self.sequence_len, self.h_dim = sequence_len, h_dim
        self.layer_norm, self.h = layer_norm, None
        self.aggregation, self.weight, self.discount = aggregation, weight, discount
        self.use_static, self.angle = use_static, angle
        self.entity_prediction, self.emb_rel, self.gpu = entity_prediction, None, gpu

        self.emb_rel = nn.Parameter(torch.Tensor(self.num_rels * 2, self.h_dim), requires_grad=True).float()
        nn.init.xavier_normal_(self.emb_rel)
        self.dynamic_emb = nn.Parameter(torch.Tensor(num_ents, h_dim), requires_grad=True).float()
        nn.init.normal_(self.dynamic_emb)

        if self.use_static:
            self.words_emb = nn.Parameter(torch.Tensor(self.num_words, h_dim), requires_grad=True).float()
            nn.init.xavier_normal_(self.words_emb)
            self.statci_rgcn_layer = RGCNBlockLayer(self.h_dim, self.h_dim, self.num_static_rels * 2,
                                                    num_bases, activation=F.rrelu, dropout=dropout)

        self.loss_e = nn.CrossEntropyLoss()
        self.rgcn = RGCNCell(num_ents, h_dim, h_dim, num_rels * 2, num_bases, num_basis,
                             num_hidden_layers, dropout, self_loop, skip_connect, encoder_name,
                             self.opn, self.emb_rel, use_cuda, analysis)
        self.time_gate_weight = nn.Parameter(torch.Tensor(h_dim, h_dim))
        nn.init.xavier_uniform_(self.time_gate_weight, gain=nn.init.calculate_gain('relu'))
        self.time_gate_bias = nn.Parameter(torch.Tensor(h_dim))
        nn.init.zeros_(self.time_gate_bias)
        self.relation_cell_1 = nn.GRUCell(self.h_dim * 2, self.h_dim)

        if decoder_name == "convtranse":
            self.decoder_ob = ConvTransE(num_ents, h_dim, input_dropout, hidden_dropout, feat_dropout)
        else:
            raise NotImplementedError

    def forward(self, g_list, static_graph, use_cuda):
        device = torch.device('cpu')
        if self.use_static:
            static_graph = static_graph.to(device)
            static_graph.ndata['h'] = torch.cat((self.dynamic_emb, self.words_emb), dim=0)
            self.statci_rgcn_layer(static_graph, [])
            static_emb = static_graph.ndata.pop('h')[:self.num_ents, :]
            self.h = F.normalize(static_emb) if self.layer_norm else static_emb
        else:
            self.h = F.normalize(self.dynamic_emb) if self.layer_norm else self.dynamic_emb[:, :]
            static_emb = None
        history_embs = []
        for i, g in enumerate(g_list):
            g = g.to(device)
            temp_e = self.h[g.r_to_e]
            x_input = torch.zeros(self.num_rels * 2, self.h_dim).float().to(device)
            for span, r_idx in zip(g.r_len, g.uniq_r):
                x_input[r_idx] = torch.mean(temp_e[span[0]:span[1], :], dim=0, keepdim=True)
            if i == 0:
                x_input = torch.cat((self.emb_rel, x_input), dim=1)
                self.h_0 = self.relation_cell_1(x_input, self.emb_rel)
                self.h_0 = F.normalize(self.h_0) if self.layer_norm else self.h_0
            else:
                x_input = torch.cat((self.emb_rel, x_input), dim=1)
                self.h_0 = self.relation_cell_1(x_input, self.h_0)
                self.h_0 = F.normalize(self.h_0) if self.layer_norm else self.h_0
            current_h = self.rgcn.forward(g, self.h, [self.h_0, self.h_0])
            current_h = F.normalize(current_h) if self.layer_norm else current_h
            time_weight = torch.sigmoid(torch.mm(self.h, self.time_gate_weight) + self.time_gate_bias)
            self.h = time_weight * current_h + (1 - time_weight) * self.h
            history_embs.append(self.h)
        return history_embs, static_emb, self.h_0

    def predict(self, test_graph, num_rels, static_graph, test_triplets, use_cuda):
        with torch.no_grad():
            inverse_test_triplets = test_triplets[:, [2, 1, 0, 3]]
            inverse_test_triplets[:, 1] = inverse_test_triplets[:, 1] + num_rels
            all_triples = torch.cat((test_triplets, inverse_test_triplets))
            evolve_embs, _, r_emb = self.forward(test_graph, static_graph, use_cuda)
            embedding = F.normalize(evolve_embs[-1]) if self.layer_norm else evolve_embs[-1]
            return all_triples, self.decoder_ob.forward(embedding, r_emb, all_triples, mode="test")


# ============================================================================
# PART 5: EVALUATION UTILITIES
# ============================================================================

def sort_and_rank(score, target):
    _, indices = torch.sort(score, dim=1, descending=True)
    indices = torch.nonzero(indices == target.view(-1, 1), as_tuple=False)
    return indices[:, 1].view(-1)


def filter_score(test_triples, score, all_ans):
    if all_ans is None: return score
    test_triples = test_triples.cpu()
    for idx, triple in enumerate(test_triples):
        h, r, t, _ = triple
        ans = list(all_ans[h.item()][r.item()])
        ans.remove(t.item())
        score[idx][torch.LongTensor(ans)] = -10000000
    return score


def get_total_rank(test_triples, score, all_ans, eval_bz=1000):
    num_triples = len(test_triples)
    n_batch = (num_triples + eval_bz - 1) // eval_bz
    rank, filter_rank = [], []
    for idx in range(n_batch):
        batch_start, batch_end = idx * eval_bz, min(num_triples, (idx + 1) * eval_bz)
        triples_batch = test_triples[batch_start:batch_end, :]
        score_batch = score[batch_start:batch_end, :]
        target = test_triples[batch_start:batch_end, 2]
        rank.append(sort_and_rank(score_batch, target))
        filter_rank.append(sort_and_rank(filter_score(triples_batch, score_batch.clone(), all_ans), target))
    rank = torch.cat(rank) + 1
    filter_rank = torch.cat(filter_rank) + 1
    return torch.mean(1.0 / filter_rank.float()).item(), torch.mean(1.0 / rank.float()).item(), rank, filter_rank


def compute_metrics_from_ranks(rank_list):
    hits = [1, 3, 10]
    total_rank = torch.cat(rank_list)
    mrr = torch.mean(1.0 / total_rank.float())
    hit_results = [torch.mean((total_rank <= hit).float()).item() for hit in hits]
    return mrr.item(), hit_results


def count_entity_frequencies(train_data, num_rels):
    entity_freq = defaultdict(int)
    for quad in train_data:
        s, r, o, t = quad
        entity_freq[s] += 2
        entity_freq[o] += 2
    return entity_freq


def identify_longtail_entities(entity_freq, threshold, num_entities):
    return set(eid for eid in range(num_entities) if entity_freq.get(eid, 0) <= threshold)


# ============================================================================
# PART 6: HARD MODE CALIBRATOR
# ============================================================================

class HardModeCalibrator:
    """HARD mode calibrator - uses ground-truth object (DATA LEAKAGE)."""
    
    def __init__(self, longtail_entities, num_entities, num_rels):
        self.longtail_entities = longtail_entities
        self.longtail_entities_set = set(longtail_entities)
        self.num_entities, self.num_rels = num_entities, num_rels
        self.entity_freq, self.entity_neighbors, self.entity_2hop_neighbors = None, None, None
        self.mult_both, self.mult_one = 1.0, 1.0
        self.adaptive_scale_type, self.adaptive_scale_factor = 'none', 1.0
        self.longtail_entity_boost, self.relation_lt_boost = None, None
        self.neighbor_weight_1hop, self.neighbor_weight_2hop, self.boost_scale = 0.0, 0.0, 1.0
        self.entity_avg_ranks, self.relation_avg_ranks = {}, {}
        self.lt_mask, self.rank_change_stats = None, None

    def _get_true_query_type(self, subject, obj):
        s_lt = subject in self.longtail_entities_set
        o_lt = obj in self.longtail_entities_set
        if s_lt and o_lt: return 'both_lt'
        elif s_lt or o_lt: return 'one_lt'
        return 'neither'

    def _compute_score_adaptive_scale(self, score_row, adaptive_type, adaptive_factor):
        if adaptive_type == 'none': return 1.0
        score_np = score_row.detach().cpu().numpy()
        if adaptive_type == 'std':
            std = np.std(score_np)
            return std * adaptive_factor if std > 1e-8 else adaptive_factor * 0.1
        elif adaptive_type == 'range':
            score_range = np.max(score_np) - np.min(score_np)
            return score_range * adaptive_factor if score_range > 1e-8 else adaptive_factor * 0.1
        elif adaptive_type == 'iqr':
            q75, q25 = np.percentile(score_np, [75, 25])
            iqr = q75 - q25
            return iqr * adaptive_factor if iqr > 1e-8 else adaptive_factor * 0.1
        return 1.0

    def learn_from_training(self, train_data, entity_freq_dict):
        print("\n[Calibrator-HARD] Learning from training set...")
        self.entity_freq = torch.zeros(self.num_entities)
        for eid, freq in entity_freq_dict.items():
            if eid < self.num_entities: self.entity_freq[eid] = freq
        self.entity_neighbors = defaultdict(set)
        for quad in train_data:
            s, r, o = int(quad[0]), int(quad[1]), int(quad[2])
            self.entity_neighbors[s].add(o)
            self.entity_neighbors[o].add(s)
        self.entity_neighbors = dict(self.entity_neighbors)
        print("  - Building 2-hop neighbor graph...")
        self.entity_2hop_neighbors = {}
        for lt_ent in tqdm(self.longtail_entities_set, desc="  2-hop graph", leave=False):
            if lt_ent in self.entity_neighbors:
                two_hop = set()
                for neighbor in self.entity_neighbors[lt_ent]:
                    if neighbor in self.entity_neighbors:
                        two_hop.update(self.entity_neighbors[neighbor])
                two_hop.discard(lt_ent)
                two_hop = two_hop & self.longtail_entities_set
                if two_hop: self.entity_2hop_neighbors[lt_ent] = two_hop
        print(f"  - Entity frequency range: [{self.entity_freq.min():.0f}, {self.entity_freq.max():.0f}]")
        print(f"  - Entities with 1-hop neighbors: {len(self.entity_neighbors)}")
        print(f"  - LT entities with 2-hop LT neighbors: {len(self.entity_2hop_neighbors)}")

    def _compute_entity_ranks(self, val_triples, val_scores, val_targets):
        val_triples_np = val_triples.cpu().numpy()
        val_scores_cpu, val_targets_cpu = val_scores.cpu(), val_targets.cpu()
        entity_ranks, relation_ranks = defaultdict(list), defaultdict(list)
        for i, triple in enumerate(val_triples_np):
            s, r, o = int(triple[0]), int(triple[1]), int(triple[2])
            target = int(val_targets_cpu[i])
            if self._get_true_query_type(s, o) == 'neither': continue
            score_row = val_scores_cpu[i]
            rank = (score_row > score_row[target]).sum().item() + 1
            if target in self.longtail_entities_set:
                entity_ranks[target].append(rank)
                relation_ranks[r].append(rank)
        self.entity_avg_ranks = {e: np.mean(r) for e, r in entity_ranks.items()}
        self.relation_avg_ranks = {r: np.mean(ranks) for r, ranks in relation_ranks.items()}

    def _create_boost_tensors(self, boost_scale, rt_high, rt_mid, rt_low):
        entity_boost = torch.zeros(self.num_entities)
        relation_boost = torch.zeros(self.num_rels * 2)
        for eid, avg_rank in self.entity_avg_ranks.items():
            if avg_rank > rt_high:
                boost = min(2.0, np.log10(avg_rank) / 2.0) * boost_scale
            elif avg_rank > rt_mid:
                boost = min(1.0, np.log10(avg_rank) / 2.5) * boost_scale
            elif avg_rank > rt_low:
                boost = min(0.5, np.log10(avg_rank) / 3.0) * boost_scale
            else:
                boost = 0.0
            entity_boost[eid] = boost
        for rid, avg_rank in self.relation_avg_ranks.items():
            if avg_rank > rt_mid:
                relation_boost[rid] = min(0.5, np.log10(avg_rank) / 4.0) * boost_scale
        return entity_boost, relation_boost

    def _apply_calibration_hard(self, triples, scores, entity_boost, relation_boost,
                                 neighbor_w1, neighbor_w2, mult_both, mult_one,
                                 adaptive_type, adaptive_factor, track_changes=False, targets=None):
        triples_np = triples.cpu().numpy()
        device = scores.device
        calibrated = scores.clone()
        lt_mask = torch.zeros(self.num_entities, dtype=torch.bool)
        for lt_ent in self.longtail_entities_set:
            if lt_ent < self.num_entities: lt_mask[lt_ent] = True
        lt_mask = lt_mask.to(device)
        entity_boost_dev = entity_boost.to(device)
        relation_boost_dev = relation_boost.to(device)
        rank_changes = {'boosted': 0, 'lowered': 0, 'unchanged': 0, 'skipped': 0, 'total': len(triples_np)}
        for i, triple in enumerate(triples_np):
            s, r, o = int(triple[0]), int(triple[1]), int(triple[2])
            query_type = self._get_true_query_type(s, o)
            if query_type == 'neither':
                rank_changes['skipped'] += 1
                continue
            mult = mult_both if query_type == 'both_lt' else mult_one
            adaptive_scale = self._compute_score_adaptive_scale(scores[i], adaptive_type, adaptive_factor)
            effective_mult = mult * adaptive_scale
            if track_changes and targets is not None:
                target = int(targets[i].item()) if torch.is_tensor(targets[i]) else int(targets[i])
                rank_before = (scores[i] > scores[i][target]).sum().item() + 1
            calibrated[i] = calibrated[i] + entity_boost_dev * effective_mult
            if r < len(relation_boost_dev) and relation_boost_dev[r] > 0:
                calibrated[i][lt_mask] += relation_boost_dev[r].item() * effective_mult
            if neighbor_w1 > 0 and s in self.entity_neighbors:
                lt_neighbors = [n for n in self.entity_neighbors[s] if n in self.longtail_entities_set and n < self.num_entities]
                if lt_neighbors:
                    boost = neighbor_w1 * effective_mult / np.sqrt(len(lt_neighbors))
                    for n in lt_neighbors: calibrated[i, n] += boost
            if neighbor_w2 > 0 and s in self.entity_2hop_neighbors:
                neighbors_2hop = self.entity_2hop_neighbors[s]
                if neighbors_2hop:
                    boost = neighbor_w2 * effective_mult / np.sqrt(len(neighbors_2hop))
                    for n in neighbors_2hop:
                        if n < self.num_entities: calibrated[i, n] += boost
            if track_changes and targets is not None:
                rank_after = (calibrated[i] > calibrated[i][target]).sum().item() + 1
                if rank_after < rank_before: rank_changes['boosted'] += 1
                elif rank_after > rank_before: rank_changes['lowered'] += 1
                else: rank_changes['unchanged'] += 1
        return calibrated, rank_changes

    def _compute_lt_mrr(self, triples, calibrated, targets):
        triples_np, targets_cpu, calibrated_cpu = triples.cpu().numpy(), targets.cpu(), calibrated.cpu()
        reciprocal_ranks = []
        for i, triple in enumerate(triples_np):
            s, r, o = int(triple[0]), int(triple[1]), int(triple[2])
            if self._get_true_query_type(s, o) == 'neither': continue
            target = int(targets_cpu[i])
            rank = (calibrated_cpu[i] > calibrated_cpu[i][target]).sum().item() + 1
            reciprocal_ranks.append(1.0 / rank)
        return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0

    def _evaluate_calibration_hard(self, triples, scores, targets, entity_boost, relation_boost,
                                    neighbor_w1, neighbor_w2, mult_both, mult_one,
                                    adaptive_type, adaptive_factor):
        calibrated, _ = self._apply_calibration_hard(triples, scores, entity_boost, relation_boost,
                                                      neighbor_w1, neighbor_w2, mult_both, mult_one,
                                                      adaptive_type, adaptive_factor)
        return self._compute_lt_mrr(triples, calibrated, targets)

    def learn_from_validation(self, val_triples, val_scores, val_targets, early_stop_patience=0.4):
        print("\n[Calibrator-HARD] Learning calibration parameters...")
        print("  - Computing entity/relation ranks...")
        self._compute_entity_ranks(val_triples, val_scores, val_targets)
        print(f"  - LT entities with rank data: {len(self.entity_avg_ranks)}")
        print(f"  - Relations with rank data: {len(self.relation_avg_ranks)}")
        baseline_mrr = self._compute_lt_mrr(val_triples, val_scores, val_targets)
        print(f"  - Baseline LT MRR: {baseline_mrr:.6f}")

        # Grid search parameters
        boost_scales = [0.1, 0.5, 1.5, 3.5]
        rank_thresholds = [(100, 50, 20), (50, 25, 10)]
        neighbor_w1_opts = [0.2, 0.5]
        neighbor_w2_opts = [0.2, 0.5]
        adaptive_type_opts = ["range", "iqr"]
        adaptive_factor_opts = [0.1, 0.5, 1.5]
        mult_both_opts = [0.2, 0.5, 1.5]
        mult_one_opts = [0.2, 0.5, 1.5]

        print("  - Pre-computing boost tensors...")
        boost_cache = {}
        for bs in boost_scales:
            for rt in rank_thresholds:
                boost_cache[(bs, rt)] = self._create_boost_tensors(bs, rt[0], rt[1], rt[2])

        total_combos = (len(boost_scales) * len(rank_thresholds) * len(neighbor_w1_opts) *
                       len(neighbor_w2_opts) * len(adaptive_type_opts) * len(adaptive_factor_opts) *
                       len(mult_both_opts) * len(mult_one_opts))
        patience_limit = int(total_combos * early_stop_patience)
        print(f"  - Grid search over {total_combos} combinations")
        print(f"  - Early stopping patience: {patience_limit} ({early_stop_patience*100:.0f}%)")

        best_mrr, best_params, combos_no_improve, combo_count = baseline_mrr, None, 0, 0
        with tqdm(total=total_combos, desc="  Grid search") as pbar:
            for bs in boost_scales:
                for rt in rank_thresholds:
                    ent_boost, rel_boost = boost_cache[(bs, rt)]
                    for nw1 in neighbor_w1_opts:
                        for nw2 in neighbor_w2_opts:
                            for at in adaptive_type_opts:
                                for af in adaptive_factor_opts:
                                    for mb in mult_both_opts:
                                        for mo in mult_one_opts:
                                            mrr = self._evaluate_calibration_hard(
                                                val_triples, val_scores, val_targets,
                                                ent_boost, rel_boost, nw1, nw2, mb, mo, at, af)
                                            combo_count += 1
                                            if mrr > best_mrr:
                                                best_mrr, combos_no_improve = mrr, 0
                                                best_params = {
                                                    'boost_scale': bs, 'rank_thresholds': rt,
                                                    'neighbor_w1': nw1, 'neighbor_w2': nw2,
                                                    'adaptive_type': at, 'adaptive_factor': af,
                                                    'mult_both': mb, 'mult_one': mo,
                                                    'entity_boost': ent_boost.clone(),
                                                    'relation_boost': rel_boost.clone(),
                                                    'best_mrr': best_mrr}
                                            else:
                                                combos_no_improve += 1
                                            pbar.update(1)
                                            pbar.set_postfix({'best': f'{best_mrr:.4f}', 'no_imp': combos_no_improve})
                                            if combos_no_improve >= patience_limit:
                                                print(f"\n  - Early stop at {combo_count}/{total_combos}")
                                                break
                                        if combos_no_improve >= patience_limit: break
                                    if combos_no_improve >= patience_limit: break
                                if combos_no_improve >= patience_limit: break
                            if combos_no_improve >= patience_limit: break
                        if combos_no_improve >= patience_limit: break
                    if combos_no_improve >= patience_limit: break
                if combos_no_improve >= patience_limit: break

        if best_params is None:
            print("  - No improvement, disabling calibration")
            self._set_no_calibration()
        else:
            self._apply_best_params(best_params, baseline_mrr)
        self._precompute_masks()

    def _set_no_calibration(self):
        self.longtail_entity_boost = torch.zeros(self.num_entities)
        self.relation_lt_boost = torch.zeros(self.num_rels * 2)
        self.neighbor_weight_1hop = self.neighbor_weight_2hop = 0.0
        self.mult_both = self.mult_one = 1.0
        self.adaptive_scale_type, self.adaptive_scale_factor = 'none', 1.0

    def _apply_best_params(self, bp, baseline_mrr):
        self.longtail_entity_boost = bp['entity_boost']
        self.relation_lt_boost = bp['relation_boost']
        self.neighbor_weight_1hop, self.neighbor_weight_2hop = bp['neighbor_w1'], bp['neighbor_w2']
        self.boost_scale = bp['boost_scale']
        self.adaptive_scale_type, self.adaptive_scale_factor = bp['adaptive_type'], bp['adaptive_factor']
        self.mult_both, self.mult_one = bp['mult_both'], bp['mult_one']
        improv = 100 * (bp['best_mrr'] - baseline_mrr) / baseline_mrr if baseline_mrr > 0 else 0
        print(f"\n  - Best LT MRR: {bp['best_mrr']:.6f} (+{improv:.2f}%)")
        rt = bp['rank_thresholds']
        print(f"  - Params: mult_both={bp['mult_both']}, mult_one={bp['mult_one']}, "
              f"adaptive={bp['adaptive_type']}/{bp['adaptive_factor']}, "
              f"boost_scale={bp['boost_scale']}, rt={rt}, nw1={bp['neighbor_w1']}, nw2={bp['neighbor_w2']}")

    def _precompute_masks(self):
        self.lt_mask = torch.zeros(self.num_entities, dtype=torch.bool)
        for lt_ent in self.longtail_entities_set:
            if lt_ent < self.num_entities: self.lt_mask[lt_ent] = True

    def calibrate_scores(self, triples, scores, num_rels, targets=None, track_changes=False):
        if self.longtail_entity_boost is None:
            if track_changes:
                self.rank_change_stats = {'boosted': 0, 'lowered': 0, 'unchanged': 0, 'skipped': len(triples), 'total': len(triples)}
            return scores
        calibrated, rank_changes = self._apply_calibration_hard(
            triples, scores, self.longtail_entity_boost, self.relation_lt_boost,
            self.neighbor_weight_1hop, self.neighbor_weight_2hop,
            self.mult_both, self.mult_one,
            self.adaptive_scale_type, self.adaptive_scale_factor,
            track_changes=track_changes, targets=targets)
        if track_changes: self.rank_change_stats = rank_changes
        return calibrated

    def get_rank_change_stats(self): return self.rank_change_stats


# ============================================================================
# PART 7: WILCOXON SIGNED-RANK TEST
# ============================================================================

def compute_wilcoxon_test(ranks_before, ranks_after, name=""):
    """Perform Wilcoxon signed-rank test comparing ranks before and after."""
    ranks_before_np = ranks_before.cpu().numpy().astype(float)
    ranks_after_np = ranks_after.cpu().numpy().astype(float)
    differences = ranks_before_np - ranks_after_np
    n_improved = np.sum(differences > 0)
    n_worsened = np.sum(differences < 0)
    n_unchanged = np.sum(differences == 0)
    median_before, median_after = np.median(ranks_before_np), np.median(ranks_after_np)
    mean_change = np.mean(differences)
    non_zero_diff = differences[differences != 0]
    
    if len(non_zero_diff) < 10:
        return {
            'name': name, 'n_samples': len(ranks_before_np),
            'n_improved': int(n_improved), 'n_worsened': int(n_worsened), 'n_unchanged': int(n_unchanged),
            'median_before': median_before, 'median_after': median_after, 'mean_change': mean_change,
            'statistic': None, 'p_value': None, 'effect_size': None, 'significant': None,
            'interpretation': "Insufficient non-zero differences (n < 10)"
        }
    try:
        statistic, p_value = stats.wilcoxon(ranks_before_np, ranks_after_np, alternative='two-sided')
        n = len(non_zero_diff)
        z_score = stats.norm.ppf(1 - p_value / 2)
        effect_size = abs(z_score) / np.sqrt(n)
        effect_interp = "small" if effect_size < 0.3 else ("medium" if effect_size < 0.5 else "large")
        significant = p_value < 0.05
        if significant:
            direction = "improvement" if mean_change > 0 else "degradation"
            interpretation = f"Significant {direction} (p={p_value:.4e}, {effect_interp} effect)"
        else:
            interpretation = f"No significant difference (p={p_value:.4e})"
        return {
            'name': name, 'n_samples': len(ranks_before_np),
            'n_improved': int(n_improved), 'n_worsened': int(n_worsened), 'n_unchanged': int(n_unchanged),
            'median_before': median_before, 'median_after': median_after, 'mean_change': mean_change,
            'statistic': statistic, 'p_value': p_value, 'effect_size': effect_size,
            'significant': significant, 'interpretation': interpretation
        }
    except Exception as e:
        return {
            'name': name, 'n_samples': len(ranks_before_np),
            'n_improved': int(n_improved), 'n_worsened': int(n_worsened), 'n_unchanged': int(n_unchanged),
            'median_before': median_before, 'median_after': median_after, 'mean_change': mean_change,
            'statistic': None, 'p_value': None, 'effect_size': None, 'significant': None,
            'interpretation': f"Test failed: {str(e)}"
        }


def print_wilcoxon_results(results):
    print("\n" + "=" * 80)
    print(f"WILCOXON SIGNED-RANK TEST: {results['name']}")
    print("=" * 80)
    print(f"\n{'Sample Statistics':<40}")
    print("-" * 80)
    print(f"  Total samples:              {results['n_samples']:>10}")
    print(f"  Improved (rank decreased):  {results['n_improved']:>10} ({100*results['n_improved']/results['n_samples']:.2f}%)")
    print(f"  Worsened (rank increased):  {results['n_worsened']:>10} ({100*results['n_worsened']/results['n_samples']:.2f}%)")
    print(f"  Unchanged:                  {results['n_unchanged']:>10} ({100*results['n_unchanged']/results['n_samples']:.2f}%)")
    print(f"\n{'Rank Statistics':<40}")
    print("-" * 80)
    print(f"  Median rank (before):       {results['median_before']:>10.2f}")
    print(f"  Median rank (after):        {results['median_after']:>10.2f}")
    print(f"  Mean rank change:           {results['mean_change']:>10.4f}")
    print(f"\n{'Test Results':<40}")
    print("-" * 80)
    if results['statistic'] is not None:
        print(f"  Test statistic (W):         {results['statistic']:>10.2f}")
        print(f"  P-value:                    {results['p_value']:>10.4e}")
        print(f"  Effect size (r):            {results['effect_size']:>10.4f}")
        print(f"  Significant (p < 0.05):     {'Yes' if results['significant'] else 'No':>10}")
    print(f"\n{'Interpretation':<40}")
    print("-" * 80)
    print(f"  {results['interpretation']}")
    print("=" * 80 + "\n")


# ============================================================================
# PART 8: EVALUATION PIPELINE
# ============================================================================

def evaluate_model(model, history_list, test_list, num_rels, num_nodes,
                   all_ans_list, static_graph, longtail_entities, calibrator=None,
                   apply_calibration=False, dataset_name="Test", args=None,
                   track_rank_changes=False, collect_ranks=False):
    ranks_raw, ranks_filter = [], []
    ranks_raw_lt, ranks_filter_lt = [], []
    all_test_triples, all_final_scores, all_longtail_masks, all_ans_collected = [], [], [], []
    total_rank_changes = {'boosted': 0, 'lowered': 0, 'unchanged': 0, 'skipped': 0, 'total': 0}
    all_ranks_raw, all_ranks_filter = [], []
    all_ranks_raw_lt, all_ranks_filter_lt = [], []
    device = torch.device('cpu')
    model.eval()
    input_list = [snap for snap in history_list[-args.test_history_len:]]

    for time_idx, test_snap in enumerate(tqdm(test_list, desc=f"Evaluating {dataset_name}")):
        history_glist = [build_sub_graph(num_nodes, num_rels, g) for g in input_list]
        test_triples_input = torch.LongTensor(test_snap).to(device)
        test_triples, final_score = model.predict(history_glist, num_rels, static_graph, test_triples_input, False)
        
        all_test_triples.append(test_triples.clone())
        all_final_scores.append(final_score.clone())
        all_ans_collected.append(all_ans_list[time_idx])

        longtail_mask = torch.tensor([
            (triple[0].item() in longtail_entities) or (triple[2].item() in longtail_entities)
            for triple in test_triples.cpu()
        ])
        all_longtail_masks.append(longtail_mask)

        if apply_calibration and calibrator is not None:
            eval_score = calibrator.calibrate_scores(test_triples, final_score, num_rels,
                                                     targets=test_triples[:, 2], track_changes=track_rank_changes)
            if track_rank_changes:
                snap_stats = calibrator.get_rank_change_stats()
                if snap_stats:
                    for k in total_rank_changes: total_rank_changes[k] += snap_stats.get(k, 0)
        else:
            eval_score = final_score

        _, _, rank_raw, rank_filter = get_total_rank(test_triples, eval_score, all_ans_list[time_idx])
        ranks_raw.append(rank_raw)
        ranks_filter.append(rank_filter)
        if collect_ranks:
            all_ranks_raw.append(rank_raw)
            all_ranks_filter.append(rank_filter)

        if longtail_mask.any():
            lt_test_triples = test_triples[longtail_mask]
            lt_eval_score = eval_score[longtail_mask]
            lt_all_ans = {}
            for i, is_lt in enumerate(longtail_mask):
                if is_lt:
                    triple = test_triples[i].cpu()
                    h, r = triple[0].item(), triple[1].item()
                    if h in all_ans_list[time_idx] and r in all_ans_list[time_idx][h]:
                        if h not in lt_all_ans: lt_all_ans[h] = {}
                        lt_all_ans[h][r] = all_ans_list[time_idx][h][r]
            _, _, rank_raw_lt, rank_filter_lt = get_total_rank(lt_test_triples, lt_eval_score, lt_all_ans)
            ranks_raw_lt.append(rank_raw_lt)
            ranks_filter_lt.append(rank_filter_lt)
            if collect_ranks:
                all_ranks_raw_lt.append(rank_raw_lt)
                all_ranks_filter_lt.append(rank_filter_lt)

        input_list.pop(0)
        input_list.append(test_snap)

    mrr_raw, hits_raw = compute_metrics_from_ranks(ranks_raw)
    mrr_filter, hits_filter = compute_metrics_from_ranks(ranks_filter)
    if ranks_raw_lt:
        mrr_raw_lt, hits_raw_lt = compute_metrics_from_ranks(ranks_raw_lt)
        mrr_filter_lt, hits_filter_lt = compute_metrics_from_ranks(ranks_filter_lt)
    else:
        mrr_raw_lt = mrr_filter_lt = 0.0
        hits_raw_lt = hits_filter_lt = [0.0, 0.0, 0.0]

    results = {
        'mrr': mrr_filter, 'mrr_raw': mrr_raw, 'hits': hits_filter, 'hits_raw': hits_raw,
        'mrr_lt': mrr_filter_lt, 'mrr_lt_raw': mrr_raw_lt, 'hits_lt': hits_filter_lt, 'hits_lt_raw': hits_raw_lt,
        'total_queries': sum(len(t) for t in all_test_triples),
        'lt_queries': sum(m.sum().item() for m in all_longtail_masks),
        'all_triples': all_test_triples, 'all_scores': all_final_scores,
        'all_masks': all_longtail_masks, 'all_ans': all_ans_collected,
        'rank_changes': total_rank_changes if track_rank_changes else None
    }
    if collect_ranks:
        results['all_ranks_raw'] = torch.cat(all_ranks_raw) if all_ranks_raw else torch.tensor([])
        results['all_ranks_filter'] = torch.cat(all_ranks_filter) if all_ranks_filter else torch.tensor([])
        results['all_ranks_raw_lt'] = torch.cat(all_ranks_raw_lt) if all_ranks_raw_lt else torch.tensor([])
        results['all_ranks_filter_lt'] = torch.cat(all_ranks_filter_lt) if all_ranks_filter_lt else torch.tensor([])
    return results


# ============================================================================
# PART 9: PRINTING AND SAVING
# ============================================================================

def print_metrics_table(title, raw_mrr, raw_hits, filter_mrr, filter_hits, num_queries=None):
    print("\n" + "=" * 70)
    print(title)
    if num_queries: print(f"Number of queries: {num_queries}")
    print("=" * 70)
    print(f"{'Metric':<20} {'Raw':>20} {'Filtered':>20}")
    print("-" * 70)
    print(f"{'MRR':<20} {raw_mrr:>20.6f} {filter_mrr:>20.6f}")
    print(f"{'Hits@1':<20} {raw_hits[0]:>20.6f} {filter_hits[0]:>20.6f}")
    print(f"{'Hits@3':<20} {raw_hits[1]:>20.6f} {filter_hits[1]:>20.6f}")
    print(f"{'Hits@10':<20} {raw_hits[2]:>20.6f} {filter_hits[2]:>20.6f}")
    print("=" * 70)


def print_comparison_table(title, before, after):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)
    for section, prefix in [("OVERALL", ""), ("LONG-TAIL", "_lt")]:
        print(f"\n>>> {section} ENTITY PREDICTION")
        print("-" * 90)
        print(f"{'Metric':<15} {'Before (Raw)':>15} {'Before (Filt)':>15} {'After (Raw)':>15} {'After (Filt)':>15}")
        print("-" * 90)
        for i, m in enumerate(['MRR', 'Hits@1', 'Hits@3', 'Hits@10']):
            if m == 'MRR':
                b_raw, b_filt = before[f'mrr{prefix}_raw'], before[f'mrr{prefix}']
                a_raw, a_filt = after[f'mrr{prefix}_raw'], after[f'mrr{prefix}']
            else:
                idx = i - 1
                b_raw, b_filt = before[f'hits{prefix}_raw'][idx], before[f'hits{prefix}'][idx]
                a_raw, a_filt = after[f'hits{prefix}_raw'][idx], after[f'hits{prefix}'][idx]
            print(f"{m:<15} {b_raw:>15.6f} {b_filt:>15.6f} {a_raw:>15.6f} {a_filt:>15.6f}")
    print("=" * 90 + "\n")


def print_rank_change_stats(rc, total):
    print("\n" + "=" * 70)
    print("RANK CHANGE STATISTICS")
    print("=" * 70)
    modified = rc['boosted'] + rc['lowered'] + rc['unchanged']
    print(f"Total: {total}, Skipped: {rc['skipped']} ({100*rc['skipped']/total:.1f}%), Modified: {modified} ({100*modified/total:.1f}%)")
    if modified > 0:
        print(f"  Boosted: {rc['boosted']} ({100*rc['boosted']/modified:.1f}%), "
              f"Lowered: {rc['lowered']} ({100*rc['lowered']/modified:.1f}%), "
              f"Unchanged: {rc['unchanged']} ({100*rc['unchanged']/modified:.1f}%)")
    print("=" * 70 + "\n")


def save_results_to_file(args, before, after, calib_params, lt_stats, wilcoxon_results):
    filename = f"3_regcn_results_{args.dataset}_hard_gpu{args.gpu}.txt"
    filepath = os.path.join("./", filename)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(filepath, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("RE-GCN EVALUATION WITH HARD MODE POST-PROCESSING\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Timestamp: {timestamp}\n\n")
        f.write("-" * 80 + "\nCONFIGURATION\n" + "-" * 80 + "\n")
        f.write(f"Dataset: {args.dataset}, GPU: {args.gpu}\n")
        f.write(f"Long-tail threshold: {args.longtail_threshold}\n")
        f.write(f"Test history length: {args.test_history_len}\n")
        f.write(f"Layer norm: {args.layer_norm}\n")
        f.write(f"Early stop patience: {args.early_stop_patience*100:.0f}%\n\n")
        f.write("-" * 80 + "\nLONG-TAIL STATISTICS\n" + "-" * 80 + "\n")
        f.write(f"LT entities: {lt_stats['num_lt_entities']} ({lt_stats['lt_percentage']:.2f}%)\n\n")
        
        for label, res in [("BEFORE", before), ("AFTER (HARD)", after)]:
            f.write("-" * 80 + f"\nTEST RESULTS - {label}\n" + "-" * 80 + "\n")
            for section, prefix in [("Overall", ""), ("Long-tail", "_lt")]:
                f.write(f"\n>>> {section} Entity Prediction\n")
                f.write(f"{'Metric':<15} {'Raw':>15} {'Filtered':>15}\n" + "-" * 45 + "\n")
                f.write(f"{'MRR':<15} {res[f'mrr{prefix}_raw']:>15.6f} {res[f'mrr{prefix}']:>15.6f}\n")
                for i, h in enumerate(['Hits@1', 'Hits@3', 'Hits@10']):
                    f.write(f"{h:<15} {res[f'hits{prefix}_raw'][i]:>15.6f} {res[f'hits{prefix}'][i]:>15.6f}\n")
            f.write(f"Total queries: {res['total_queries']}, LT queries: {res['lt_queries']}\n\n")

        f.write("-" * 80 + "\nIMPROVEMENT SUMMARY\n" + "-" * 80 + "\n")
        for section, prefix in [("Overall", ""), ("Long-tail", "_lt")]:
            f.write(f"\n>>> {section}\n")
            f.write(f"{'Metric':<15} {'Raw Î”':>15} {'Filt Î”':>15}\n" + "-" * 45 + "\n")
            f.write(f"{'MRR':<15} {after[f'mrr{prefix}_raw']-before[f'mrr{prefix}_raw']:>+15.6f} {after[f'mrr{prefix}']-before[f'mrr{prefix}']:>+15.6f}\n")
            for i, h in enumerate(['Hits@1', 'Hits@3', 'Hits@10']):
                f.write(f"{h:<15} {after[f'hits{prefix}_raw'][i]-before[f'hits{prefix}_raw'][i]:>+15.6f} {after[f'hits{prefix}'][i]-before[f'hits{prefix}'][i]:>+15.6f}\n")

        if after.get('rank_changes'):
            rc = after['rank_changes']
            f.write("\n" + "-" * 80 + "\nRANK CHANGE STATISTICS\n" + "-" * 80 + "\n")
            f.write(f"Total: {rc['total']}, Skipped: {rc['skipped']}, Boosted: {rc['boosted']}, Lowered: {rc['lowered']}, Unchanged: {rc['unchanged']}\n")

        f.write("\n" + "-" * 80 + "\nWILCOXON SIGNED-RANK TEST RESULTS\n" + "-" * 80 + "\n")
        for w in wilcoxon_results:
            f.write(f"\n>>> {w['name']}\n")
            f.write(f"  Samples: {w['n_samples']}, Improved: {w['n_improved']}, Worsened: {w['n_worsened']}, Unchanged: {w['n_unchanged']}\n")
            f.write(f"  Median before: {w['median_before']:.2f}, Median after: {w['median_after']:.2f}, Mean change: {w['mean_change']:.4f}\n")
            if w['statistic'] is not None:
                f.write(f"  W={w['statistic']:.2f}, p={w['p_value']:.4e}, r={w['effect_size']:.4f}, Significant: {'Yes' if w['significant'] else 'No'}\n")
            f.write(f"  {w['interpretation']}\n")

        if calib_params:
            f.write("\n" + "-" * 80 + "\nCALIBRATOR PARAMETERS\n" + "-" * 80 + "\n")
            for k, v in calib_params.items():
                if not k.startswith('entity') and not k.startswith('relation'):
                    f.write(f"{k}: {v}\n")
        f.write("\n" + "=" * 80 + "\nEND OF RESULTS\n" + "=" * 80 + "\n")
    print(f"\n[INFO] Results saved to: {filepath}")
    return filepath


# ============================================================================
# PART 10: MAIN EXPERIMENT
# ============================================================================

def run_experiment(args):
    print("\n" + "=" * 70)
    print("LOADING DATASET")
    print("=" * 70)
    data = load_data(args.dataset)
    train_list, _ = split_by_time(data.train)
    valid_list, _ = split_by_time(data.valid)
    test_list, _ = split_by_time(data.test)
    num_nodes, num_rels = data.num_nodes, data.num_rels

    entity_freq = count_entity_frequencies(data.train, num_rels)
    longtail_entities = identify_longtail_entities(entity_freq, args.longtail_threshold, num_nodes)
    lt_stats = {'num_lt_entities': len(longtail_entities), 'num_entities': num_nodes,
                'lt_percentage': 100.0 * len(longtail_entities) / num_nodes}
    print(f"\nLong-tail: {len(longtail_entities)} entities ({lt_stats['lt_percentage']:.2f}%)")

    all_ans_test = load_all_answers_for_time_filter(data.test, num_rels, num_nodes)
    all_ans_valid = load_all_answers_for_time_filter(data.valid, num_rels, num_nodes)

    model_name = f"{args.dataset}-uvrgcn-convtranse-ly2-his12-angle10-dp0.2_0.2_0.2-gpu{args.gpu}"
    model_state_file = os.path.join('./models/', model_name)
    print(f"Model checkpoint: {model_state_file}")
    if not os.path.exists(model_state_file):
        print(f"ERROR: Model not found: {model_state_file}")
        return None, None

    num_static_rels, num_words, static_graph = 0, 0, None
    if args.add_static_graph:
        static_path = f"./data/{args.dataset}/e-w-graph.txt"
        if os.path.exists(static_path):
            static_triples = []
            with open(static_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    static_triples.append([int(parts[0]), int(parts[1]), int(parts[2])])
            static_triples = np.array(static_triples)
            num_static_rels = len(np.unique(static_triples[:, 1]))
            num_words = len(np.unique(static_triples[:, 2]))
            static_triples[:, 2] += num_nodes
            static_graph = build_sub_graph(num_words + num_nodes, num_static_rels, static_triples)
        else:
            args.add_static_graph = False

    model = RecurrentRGCN(
        decoder_name="convtranse", encoder_name="uvrgcn", num_ents=num_nodes,
        num_rels=num_rels, num_static_rels=num_static_rels, num_words=num_words,
        h_dim=args.n_hidden, opn="sub", sequence_len=args.test_history_len,
        num_bases=args.n_bases, num_basis=args.n_basis, num_hidden_layers=args.n_layers,
        dropout=args.dropout, self_loop=True, skip_connect=False, layer_norm=args.layer_norm,
        input_dropout=args.input_dropout, hidden_dropout=args.hidden_dropout,
        feat_dropout=args.feat_dropout, aggregation="none", weight=0.5, discount=1,
        angle=10, use_static=args.add_static_graph, entity_prediction=True,
        relation_prediction=False, use_cuda=False, gpu=args.gpu, analysis=False)

    ckpt = torch.load(model_state_file, map_location=torch.device('cpu'))
    print(f"Loaded from epoch: {ckpt.get('epoch', 'unknown')}")
    model.load_state_dict(ckpt['state_dict'])

    calibrator = HardModeCalibrator(longtail_entities, num_nodes, num_rels)
    calibrator.learn_from_training(data.train, entity_freq)

    # STEP 1: Validation
    print("\n" + "=" * 70 + "\nSTEP 1: VALIDATION SET\n" + "=" * 70)
    val_results = evaluate_model(model, train_list, valid_list, num_rels, num_nodes,
                                  all_ans_valid, static_graph, longtail_entities,
                                  dataset_name="Validation", args=args)
    print_metrics_table("VALIDATION - Overall", val_results['mrr_raw'], val_results['hits_raw'],
                        val_results['mrr'], val_results['hits'], val_results['total_queries'])
    if val_results['mrr_lt'] > 0:
        print_metrics_table("VALIDATION - Long-tail", val_results['mrr_lt_raw'], val_results['hits_lt_raw'],
                            val_results['mrr_lt'], val_results['hits_lt'], val_results['lt_queries'])

    val_triples = torch.cat(val_results['all_triples'], dim=0)
    val_scores = torch.cat(val_results['all_scores'], dim=0)
    calibrator.learn_from_validation(val_triples, val_scores, val_triples[:, 2], args.early_stop_patience)
    calib_params = {
        'mode': 'hard', 'boost_scale': calibrator.boost_scale,
        'neighbor_w1': calibrator.neighbor_weight_1hop, 'neighbor_w2': calibrator.neighbor_weight_2hop,
        'adaptive_type': calibrator.adaptive_scale_type, 'adaptive_factor': calibrator.adaptive_scale_factor,
        'mult_both': calibrator.mult_both, 'mult_one': calibrator.mult_one}

    # STEP 2: Test BEFORE
    print("\n" + "=" * 70 + "\nSTEP 2: TEST (BEFORE POST-PROCESSING)\n" + "=" * 70)
    test_before = evaluate_model(model, train_list + valid_list, test_list, num_rels, num_nodes,
                                  all_ans_test, static_graph, longtail_entities,
                                  dataset_name="Test (Before)", args=args, collect_ranks=True)
    print_metrics_table("TEST (BEFORE) - Overall", test_before['mrr_raw'], test_before['hits_raw'],
                        test_before['mrr'], test_before['hits'], test_before['total_queries'])
    if test_before['mrr_lt'] > 0:
        print_metrics_table("TEST (BEFORE) - Long-tail", test_before['mrr_lt_raw'], test_before['hits_lt_raw'],
                            test_before['mrr_lt'], test_before['hits_lt'], test_before['lt_queries'])

    # STEP 3: Test AFTER
    print("\n" + "=" * 70 + "\nSTEP 3: TEST (AFTER POST-PROCESSING - HARD)\n" + "=" * 70)
    test_after = evaluate_model(model, train_list + valid_list, test_list, num_rels, num_nodes,
                                 all_ans_test, static_graph, longtail_entities,
                                 calibrator=calibrator, apply_calibration=True,
                                 dataset_name="Test (HARD)", args=args,
                                 track_rank_changes=True, collect_ranks=True)
    print_metrics_table("TEST (AFTER) - Overall", test_after['mrr_raw'], test_after['hits_raw'],
                        test_after['mrr'], test_after['hits'], test_after['total_queries'])
    if test_after['mrr_lt'] > 0:
        print_metrics_table("TEST (AFTER) - Long-tail", test_after['mrr_lt_raw'], test_after['hits_lt_raw'],
                            test_after['mrr_lt'], test_after['hits_lt'], test_after['lt_queries'])

    print_comparison_table("TEST: BEFORE vs AFTER (HARD MODE)", test_before, test_after)
    if test_after['rank_changes']:
        print_rank_change_stats(test_after['rank_changes'], test_after['total_queries'])

    # STEP 4: Wilcoxon tests
    print("\n" + "=" * 70 + "\nSTEP 4: WILCOXON SIGNED-RANK TESTS\n" + "=" * 70)
    wilcoxon_results = []
    w_overall_raw = compute_wilcoxon_test(test_before['all_ranks_raw'], test_after['all_ranks_raw'],
                                          "Overall Entity Prediction (Raw)")
    wilcoxon_results.append(w_overall_raw)
    print_wilcoxon_results(w_overall_raw)
    
    w_overall_filter = compute_wilcoxon_test(test_before['all_ranks_filter'], test_after['all_ranks_filter'],
                                              "Overall Entity Prediction (Filtered)")
    wilcoxon_results.append(w_overall_filter)
    print_wilcoxon_results(w_overall_filter)
    
    if len(test_before['all_ranks_raw_lt']) > 0:
        w_lt_raw = compute_wilcoxon_test(test_before['all_ranks_raw_lt'], test_after['all_ranks_raw_lt'],
                                          "Long-tail Entity Prediction (Raw)")
        wilcoxon_results.append(w_lt_raw)
        print_wilcoxon_results(w_lt_raw)
        
        w_lt_filter = compute_wilcoxon_test(test_before['all_ranks_filter_lt'], test_after['all_ranks_filter_lt'],
                                             "Long-tail Entity Prediction (Filtered)")
        wilcoxon_results.append(w_lt_filter)
        print_wilcoxon_results(w_lt_filter)

    save_results_to_file(args, test_before, test_after, calib_params, lt_stats, wilcoxon_results)
    return test_before, test_after


# ============================================================================
# PART 11: MAIN ENTRY
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='RE-GCN Evaluation with HARD Mode Post-Processing')
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID in checkpoint name")
    parser.add_argument("-d", "--dataset", type=str, default='ICEWS14',
                        choices=['ICEWS18', 'ICEWS14', 'ICEWS05-15'])
    parser.add_argument("--longtail-threshold", type=int, default=5)
    parser.add_argument("--test-history-len", type=int, default=12)
    parser.add_argument("--add-static-graph", action='store_true', default=True)
    parser.add_argument("--n-hidden", type=int, default=200)
    parser.add_argument("--n-bases", type=int, default=100)
    parser.add_argument("--n-basis", type=int, default=100)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--input-dropout", type=float, default=0.2)
    parser.add_argument("--hidden-dropout", type=float, default=0.2)
    parser.add_argument("--feat-dropout", type=float, default=0.2)
    parser.add_argument("--layer-norm", action='store_true', default=True)
    parser.add_argument("--early-stop-patience", type=float, default=0.5,
                        help="Early stop patience ratio (0-1, default 0.4)")
    return parser.parse_args()


def main():
    import warnings
    warnings.filterwarnings("ignore")
    args = parse_args()
    if args.early_stop_patience <= 0 or args.early_stop_patience > 1:
        print("ERROR: early_stop_patience must be in (0, 1]")
        return
    print("\n" + "=" * 70)
    print("RE-GCN EVALUATION WITH HARD MODE POST-PROCESSING")
    print("+ Wilcoxon Signed-Rank Test for Statistical Significance")
    print("=" * 70)
    print("\nConfiguration:")
    for arg, value in sorted(vars(args).items()):
        print(f"  {arg}: {value}")
    print("=" * 70 + "\n")
    run_experiment(args)


if __name__ == '__main__':
    main()