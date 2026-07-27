from __future__ import absolute_import, print_function
import argparse
import gzip
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime

import dgl
import dgl.function as fn
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from tqdm import tqdm
from scipy import stats

np.random.seed(123)
torch.manual_seed(123)


# ============================================================================
# PART 1: KNOWLEDGE GRAPH DATA LOADING
# ============================================================================

class RGCNLinkDataset(object):
    def __init__(self, name, dir=None):
        self.name = name
        if dir:
            self.dir = os.path.join(dir, self.name)
        else:
            raise ValueError("Directory must be provided")
        print("Dataset directory:", self.dir)

    def load(self, load_time=True):
        entity_path = os.path.join(self.dir, 'entity2id.txt')
        relation_path = os.path.join(self.dir, 'relation2id.txt')
        train_path = os.path.join(self.dir, 'train.txt')
        valid_path = os.path.join(self.dir, 'valid.txt')
        test_path = os.path.join(self.dir, 'test.txt')
        entity_dict = _read_dictionary(entity_path)
        relation_dict = _read_dictionary(relation_path)
        self.train = np.array(_read_triplets_as_list(train_path, entity_dict, relation_dict, load_time))
        self.valid = np.array(_read_triplets_as_list(valid_path, entity_dict, relation_dict, load_time))
        self.test = np.array(_read_triplets_as_list(test_path, entity_dict, relation_dict, load_time))
        self.num_nodes = len(entity_dict)
        print("# Entities: {}".format(self.num_nodes))
        self.num_rels = len(relation_dict)
        self.relation_dict = relation_dict
        self.entity_dict = entity_dict
        print("# Relations: {}".format(self.num_rels))
        print("# Training edges: {}".format(len(self.train)))


def _read_dictionary(filename):
    d = {}
    with open(filename, 'r+', encoding='utf-8') as f:
        for line in f:
            line = line.strip().split('\t')
            d[int(line[1])] = line[0]
    return d


def _read_triplets(filename):
    with open(filename, 'r+', encoding='utf-8') as f:
        for line in f:
            processed_line = line.strip().split('\t')
            yield processed_line


def _read_triplets_as_list(filename, entity_dict, relation_dict, load_time):
    l = []
    for triplet in _read_triplets(filename):
        s = int(triplet[0])
        r = int(triplet[1])
        o = int(triplet[2])
        if load_time:
            st = int(triplet[3])
            l.append([s, r, o, st])
        else:
            l.append([s, r, o])
    return l


def load_data(dataset):
    if dataset in ['ICEWS18', 'ICEWS14', 'GDELT', 'ICEWS14s', 'ICEWS05-15', 'YAGO', 'WIKI']:
        data = RGCNLinkDataset(dataset, "./data")
        data.load()
        return data
    else:
        raise ValueError('Unknown dataset: {}'.format(dataset))


# ============================================================================
# PART 2: DATA SPLITTING AND PREPROCESSING
# ============================================================================

def split_by_time(data):
    snapshot_list, snapshot = [], []
    latest_t = 0
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
    nodes, rels = [], []
    for snap in snapshot_list:
        uniq_v, _ = np.unique((snap[:, 0], snap[:, 2]), return_inverse=True)
        uniq_r = np.unique(snap[:, 1])
        nodes.append(len(uniq_v))
        rels.append(len(uniq_r) * 2)
    times = sorted(set(triple[3] for triple in data))
    print("# Ave nodes: {:.2f}, Ave rels: {:.2f}, Snapshots: {}".format(
        np.average(nodes), np.average(rels), len(snapshot_list)))
    return snapshot_list, np.asarray(times)


def add_subject(e1, e2, r, d, num_rel):
    if e2 not in d:
        d[e2] = {}
    if r + num_rel not in d[e2]:
        d[e2][r + num_rel] = set()
    d[e2][r + num_rel].add(e1)


def add_object(e1, e2, r, d, num_rel):
    if e1 not in d:
        d[e1] = {}
    if r not in d[e1]:
        d[e1][r] = set()
    d[e1][r].add(e2)


def load_all_answers_for_filter(total_data, num_rel, rel_p=False):
    def add_relation(e1, e2, r, d):
        if e1 not in d:
            d[e1] = {}
        if e2 not in d[e1]:
            d[e1][e2] = set()
        d[e1][e2].add(r)
    all_ans = {}
    for line in total_data:
        s, r, o = line[:3]
        if rel_p:
            add_relation(s, o, r, all_ans)
            add_relation(o, s, r + num_rel, all_ans)
        else:
            add_subject(s, o, r, all_ans, num_rel=num_rel)
            add_object(s, o, r, all_ans, num_rel=0)
    return all_ans


def load_all_answers_for_time_filter(total_data, num_rels, num_nodes, rel_p=False):
    all_ans_list = []
    all_snap, _ = split_by_time(total_data)
    for snap in all_snap:
        all_ans_list.append(load_all_answers_for_filter(snap, num_rels, rel_p))
    return all_ans_list


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
# PART 3: UTILITY FUNCTIONS
# ============================================================================

def sort_and_rank(score, target):
    device = score.device
    target = target.to(device)
    _, indices = torch.sort(score, dim=1, descending=True)
    indices = torch.nonzero(indices == target.view(-1, 1), as_tuple=False)
    return indices[:, 1].view(-1)


def filter_score(test_triples, score, all_ans):
    if all_ans is None:
        return score
    test_triples = test_triples.cpu()
    for idx, triple in enumerate(test_triples):
        h, r, t, _ = triple
        ans = list(all_ans[h.item()][r.item()])
        ans.remove(t.item())
        score[idx][torch.LongTensor(ans)] = -10000000
    return score


def r2e(triplets, num_rels):
    src, rel, dst = triplets.transpose()
    uniq_r = np.concatenate((np.unique(rel), np.unique(rel) + num_rels))
    r_to_e = defaultdict(set)
    for s, r, d in triplets:
        r_to_e[r].add(s)
        r_to_e[r + num_rels].add(s)
    r_len, e_idx, idx = [], [], 0
    for r in uniq_r:
        r_len.append((idx, idx + len(r_to_e[r])))
        e_idx.extend(list(r_to_e[r]))
        idx += len(r_to_e[r])
    return uniq_r, r_len, e_idx


def build_sub_graph(num_nodes, num_rels, triples, use_cuda, gpu):
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
    g.uniq_r = uniq_r
    g.r_to_e = r_to_e
    g.r_len = r_len
    if use_cuda:
        g.to(gpu)
        g.r_to_e = torch.from_numpy(np.array(r_to_e)).long()
    return g


# ============================================================================
# PART 4: NEURAL NETWORK LAYERS
# ============================================================================

class RGCNLayer(nn.Module):
    def __init__(self, in_feat, out_feat, bias=None, activation=None,
                 self_loop=False, skip_connect=False, dropout=0.0, layer_norm=False):
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

    def propagate(self, g):
        raise NotImplementedError

    def forward(self, g, prev_h=[]):
        if self.self_loop:
            loop_message = torch.mm(g.ndata['h'], self.loop_weight)
            if self.dropout:
                loop_message = self.dropout(loop_message)
        if len(prev_h) != 0 and self.skip_connect:
            skip_weight = torch.sigmoid(torch.mm(prev_h, self.skip_connect_weight) + self.skip_connect_bias)
        self.propagate(g)
        node_repr = g.ndata['h']
        if self.bias is not None:
            node_repr = node_repr + self.bias
        if len(prev_h) != 0 and self.skip_connect:
            previous_node_repr = (1 - skip_weight) * prev_h
            if self.activation:
                node_repr = self.activation(node_repr)
            if self.self_loop:
                loop_message = skip_weight * (self.activation(loop_message) if self.activation else loop_message)
                node_repr = node_repr + loop_message
            node_repr = node_repr + previous_node_repr
        else:
            if self.self_loop:
                node_repr = node_repr + loop_message
            if self.layer_norm:
                node_repr = self.normalization_layer(node_repr)
            if self.activation:
                node_repr = self.activation(node_repr)
        g.ndata['h'] = node_repr
        return node_repr


class RGCNBlockLayer(RGCNLayer):
    def __init__(self, in_feat, out_feat, num_rels, num_bases, bias=None,
                 activation=None, self_loop=False, dropout=0.0, skip_connect=False, layer_norm=False):
        super(RGCNBlockLayer, self).__init__(in_feat, out_feat, bias, activation,
                                             self_loop=self_loop, skip_connect=skip_connect, dropout=dropout)
        self.num_rels = num_rels
        self.num_bases = num_bases
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
    def __init__(self, in_feat, out_feat, num_rels, num_bases=-1, bias=None,
                 activation=None, self_loop=False, dropout=0.0, skip_connect=False, rel_emb=None):
        super(UnionRGCNLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop
        self.num_rels = num_rels
        self.rel_emb = None
        self.skip_connect = skip_connect
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
        self.conv1d = nn.Conv1d(2, 1, 1, stride=1, padding=0)
        self.fc = nn.Linear(self.out_feat, self.out_feat)
        self.gnn_bn0 = nn.BatchNorm1d(2)
        self.gnn_bn1 = nn.BatchNorm1d(1)
        self.gnn_bn2 = nn.BatchNorm1d(self.out_feat)

    def propagate(self, g):
        g.update_all(lambda x: self.msg_func(x), fn.sum(msg='msg', out='h'), self.apply_func)

    def forward(self, g, prev_h, emb_rel):
        device = torch.device("cpu")
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
            if self.self_loop:
                node_repr = node_repr + loop_message
            node_repr = skip_weight * node_repr + (1 - skip_weight) * prev_h
        else:
            if self.self_loop:
                node_repr = node_repr + loop_message
        if self.activation:
            node_repr = self.activation(node_repr)
        if self.dropout:
            node_repr = self.dropout(node_repr)
        g.ndata['h'] = node_repr
        return node_repr

    def msg_func(self, edges):
        relation = self.rel_emb.index_select(0, edges.data['type']).view(-1, self.out_feat)
        node = edges.src['h'].view(-1, self.out_feat)
        batch_size = node.shape[0]
        stacked_inputs = torch.cat([node.unsqueeze(1), relation.unsqueeze(1)], 1)
        stacked_inputs = self.gnn_bn0(stacked_inputs)
        msg = self.conv1d(stacked_inputs).view(batch_size, -1)
        return {'msg': torch.mm(msg, self.weight_neighbor)}

    def apply_func(self, nodes):
        return {'h': nodes.data['h'] * nodes.data['norm']}


# ============================================================================
# PART 5: DECODERS
# ============================================================================

class TimeConvTransR(nn.Module):
    def __init__(self, num_relations, embedding_dim, input_dropout=0, hidden_dropout=0,
                 feature_map_dropout=0, channels=50, kernel_size=3):
        super(TimeConvTransR, self).__init__()
        self.inp_drop = nn.Dropout(input_dropout)
        self.hidden_drop = nn.Dropout(hidden_dropout)
        self.feature_map_drop = nn.Dropout(feature_map_dropout)
        self.conv1 = nn.Conv1d(4, channels, kernel_size, stride=1, padding=int(math.floor(kernel_size / 2)))
        self.bn0 = nn.BatchNorm1d(4)
        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(embedding_dim)
        self.register_parameter('b', Parameter(torch.zeros(num_relations * 2)))
        self.fc = nn.Linear(embedding_dim * channels, embedding_dim)

    def forward(self, embedding, emb_rel, emb_time, triplets, partial_embeding=None):
        e1_embedded_all = torch.tanh(embedding)
        device = e1_embedded_all.device
        triplets = triplets.to(device)
        batch_size = len(triplets)
        e1_embedded = e1_embedded_all[triplets[:, 0]].unsqueeze(1)
        e2_embedded = e1_embedded_all[triplets[:, 2]].unsqueeze(1)
        emb_time_1, emb_time_2 = emb_time
        stacked_inputs = torch.cat([e1_embedded, e2_embedded, emb_time_1.unsqueeze(1), emb_time_2.unsqueeze(1)], 1)
        stacked_inputs = self.bn0(stacked_inputs)
        x = self.inp_drop(stacked_inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, emb_rel.transpose(1, 0))
        if partial_embeding is not None:
            x = torch.mul(x, partial_embeding)
        return x


class TimeConvTransE(nn.Module):
    def __init__(self, num_entities, embedding_dim, input_dropout=0, hidden_dropout=0,
                 feature_map_dropout=0, channels=50, kernel_size=3):
        super(TimeConvTransE, self).__init__()
        self.num_entities = num_entities
        self.inp_drop = nn.Dropout(input_dropout)
        self.hidden_drop = nn.Dropout(hidden_dropout)
        self.feature_map_drop = nn.Dropout(feature_map_dropout)
        self.conv1 = nn.Conv1d(4, channels, kernel_size, stride=1, padding=int(math.floor(kernel_size / 2)))
        self.bn0 = nn.BatchNorm1d(4)
        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(embedding_dim)
        self.register_parameter('b', Parameter(torch.zeros(num_entities)))
        self.fc = nn.Linear(embedding_dim * channels, embedding_dim)

    def forward(self, embedding, emb_rel, emb_time, triplets, partial_embeding=None):
        e1_embedded_all = torch.tanh(embedding)
        device = e1_embedded_all.device
        triplets = triplets.to(device)
        batch_size = len(triplets)
        e1_embedded = e1_embedded_all[triplets[:, 0]].unsqueeze(1)
        rel_embedded = emb_rel[triplets[:, 1]].unsqueeze(1)
        emb_time_1, emb_time_2 = emb_time
        stacked_inputs = torch.cat([e1_embedded, rel_embedded, emb_time_1.unsqueeze(1), emb_time_2.unsqueeze(1)], 1)
        stacked_inputs = self.bn0(stacked_inputs)
        x = self.inp_drop(stacked_inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        if batch_size > 1:
            x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, e1_embedded_all.transpose(1, 0))
        if partial_embeding is not None:
            x = torch.mul(x, partial_embeding)
        return x


# ============================================================================
# PART 6: BASE RGCN MODEL
# ============================================================================

class BaseRGCN(nn.Module):
    def __init__(self, num_nodes, h_dim, out_dim, num_rels, num_bases=-1, num_basis=-1,
                 num_hidden_layers=1, dropout=0, self_loop=False, skip_connect=False,
                 encoder_name="", opn="sub", rel_emb=None, use_cuda=False, analysis=False):
        super(BaseRGCN, self).__init__()
        self.num_nodes = num_nodes
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.num_basis = num_basis
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.skip_connect = skip_connect
        self.self_loop = self_loop
        self.encoder_name = encoder_name
        self.use_cuda = use_cuda
        self.run_analysis = analysis
        self.rel_emb = rel_emb
        self.opn = opn
        self.build_model()
        self.features = self.create_features()

    def build_model(self):
        self.layers = nn.ModuleList()
        i2h = self.build_input_layer()
        if i2h is not None:
            self.layers.append(i2h)
        for idx in range(self.num_hidden_layers):
            self.layers.append(self.build_hidden_layer(idx))
        h2o = self.build_output_layer()
        if h2o is not None:
            self.layers.append(h2o)

    def create_features(self):
        return None

    def build_input_layer(self):
        return None

    def build_hidden_layer(self, idx):
        raise NotImplementedError

    def build_output_layer(self):
        return None


# ============================================================================
# PART 7: RECURRENT RGCN MODEL
# ============================================================================

class RGCNCell(BaseRGCN):
    def build_hidden_layer(self, idx):
        act = F.rrelu
        if idx:
            self.num_basis = 0
        sc = False if idx == 0 or not self.skip_connect else True
        if self.encoder_name == "convgcn":
            return UnionRGCNLayer(self.h_dim, self.h_dim, self.num_rels, self.num_bases,
                                  activation=act, dropout=self.dropout, self_loop=self.self_loop,
                                  skip_connect=sc, rel_emb=self.rel_emb)
        raise NotImplementedError

    def forward(self, g, init_ent_emb, init_rel_emb):
        if self.encoder_name == "convgcn":
            node_id = g.ndata['id'].squeeze()
            g.ndata['h'] = init_ent_emb[node_id]
            for i, layer in enumerate(self.layers):
                layer(g, [], init_rel_emb[i])
            return g.ndata.pop('h')
        if self.features is not None:
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
    def __init__(self, decoder_name, encoder_name, num_ents, num_rels, num_static_rels, num_words, num_times,
                 time_interval, h_dim, opn, history_rate, sequence_len, num_bases=-1, num_basis=-1,
                 num_hidden_layers=1, dropout=0, self_loop=False, skip_connect=False, layer_norm=False,
                 input_dropout=0, hidden_dropout=0, feat_dropout=0, aggregation='cat', weight=1, discount=0,
                 angle=0, use_static=False, entity_prediction=False, relation_prediction=False, use_cuda=False,
                 gpu=0, analysis=False):
        super(RecurrentRGCN, self).__init__()
        self.decoder_name = decoder_name
        self.encoder_name = encoder_name
        self.num_rels = num_rels
        self.num_ents = num_ents
        self.opn = opn
        self.history_rate = history_rate
        self.num_words = num_words
        self.num_static_rels = num_static_rels
        self.num_times = num_times
        self.time_interval = time_interval
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
        self.relation_prediction = relation_prediction
        self.entity_prediction = entity_prediction
        self.gpu = torch.device("cpu")
        self.emb_rel = nn.Parameter(torch.Tensor(self.num_rels * 2, self.h_dim))
        nn.init.xavier_normal_(self.emb_rel)
        self.dynamic_emb = nn.Parameter(torch.Tensor(num_ents, h_dim))
        nn.init.normal_(self.dynamic_emb)
        self.weight_t1 = nn.Parameter(torch.randn(1, h_dim))
        self.bias_t1 = nn.Parameter(torch.randn(1, h_dim))
        self.weight_t2 = nn.Parameter(torch.randn(1, h_dim))
        self.bias_t2 = nn.Parameter(torch.randn(1, h_dim))
        if self.use_static:
            self.words_emb = nn.Parameter(torch.Tensor(self.num_words, h_dim))
            nn.init.xavier_normal_(self.words_emb)
            self.statci_rgcn_layer = RGCNBlockLayer(self.h_dim, self.h_dim, self.num_static_rels * 2, num_bases,
                                                    activation=F.rrelu, dropout=dropout, self_loop=False, skip_connect=False)
        self.rgcn = RGCNCell(num_ents, h_dim, h_dim, num_rels * 2, num_bases, num_basis,
                             num_hidden_layers, dropout, self_loop, skip_connect, encoder_name,
                             opn, self.emb_rel, use_cuda, analysis)
        self.relation_cell_1 = nn.GRUCell(self.h_dim * 2, self.h_dim)
        self.entity_cell_1 = nn.GRUCell(self.h_dim, self.h_dim)
        if decoder_name == "timeconvtranse":
            self.decoder_ob1 = TimeConvTransE(num_ents, h_dim, input_dropout, hidden_dropout, feat_dropout)
            self.decoder_ob2 = TimeConvTransE(num_ents, h_dim, input_dropout, hidden_dropout, feat_dropout)
            self.rdecoder_re1 = TimeConvTransR(num_rels, h_dim, input_dropout, hidden_dropout, feat_dropout)
            self.rdecoder_re2 = TimeConvTransR(num_rels, h_dim, input_dropout, hidden_dropout, feat_dropout)
        else:
            raise NotImplementedError

    def forward(self, g_list, static_graph, use_cuda):
        if self.use_static:
            static_graph = static_graph.to(self.gpu)
            static_graph.ndata['h'] = torch.cat((self.dynamic_emb, self.words_emb), dim=0)
            self.statci_rgcn_layer(static_graph, [])
            static_emb = static_graph.ndata.pop('h')[:self.num_ents, :]
            static_emb = F.normalize(static_emb) if self.layer_norm else static_emb
            self.h = static_emb
        else:
            self.h = F.normalize(self.dynamic_emb) if self.layer_norm else self.dynamic_emb[:, :]
            static_emb = None
        history_embs = []
        for i, g in enumerate(g_list):
            g = g.to(self.gpu)
            temp_e = self.h[g.r_to_e]
            x_input = torch.zeros(self.num_rels * 2, self.h_dim).float()
            for span, r_idx in zip(g.r_len, g.uniq_r):
                x_input[r_idx] = torch.mean(temp_e[span[0]:span[1], :], dim=0, keepdim=True)
            x_input = torch.cat((self.emb_rel, x_input), dim=1)
            if i == 0:
                self.h_0 = self.relation_cell_1(x_input, self.emb_rel)
            else:
                self.h_0 = self.relation_cell_1(x_input, self.h_0)
            self.h_0 = F.normalize(self.h_0) if self.layer_norm else self.h_0
            current_h = self.rgcn.forward(g, self.h, [self.h_0, self.h_0])
            current_h = F.normalize(current_h) if self.layer_norm else current_h
            self.h = self.entity_cell_1(current_h, self.h)
            self.h = F.normalize(self.h) if self.layer_norm else self.h
            history_embs.append(self.h)
        return history_embs, static_emb, self.h_0

    def predict(self, test_graph, num_rels, static_graph, test_triplets, entity_history_vocabulary,
                rel_history_vocabulary, use_cuda):
        with torch.no_grad():
            inverse_test_triplets = test_triplets[:, [2, 1, 0, 3]]
            inverse_test_triplets[:, 1] = inverse_test_triplets[:, 1] + num_rels
            all_triples = torch.cat((test_triplets, inverse_test_triplets))
            evolve_embs, _, r_emb = self.forward(test_graph, static_graph, use_cuda)
            embedding = F.normalize(evolve_embs[-1]) if self.layer_norm else evolve_embs[-1]
            time_embs = self.get_init_time(all_triples)
            logits_r = self.decoder_ob1.forward(embedding, r_emb, time_embs, all_triples).view(-1, self.num_ents)
            logits_h = self.decoder_ob2.forward(embedding, r_emb, time_embs, all_triples,
                                                 partial_embeding=entity_history_vocabulary).view(-1, self.num_ents)
            combined_logits = self.history_rate * logits_h + (1 - self.history_rate) * logits_r
            score_rel_r = self.rel_raw_mode(embedding, r_emb, time_embs, all_triples)
            score_rel_h = self.rel_history_mode(embedding, r_emb, time_embs, all_triples, rel_history_vocabulary)
            score_rel = self.history_rate * score_rel_h + (1 - self.history_rate) * score_rel_r
            score_rel = torch.log(score_rel)
            return all_triples, combined_logits, score_rel

    def get_init_time(self, quadrupleList):
        device = self.weight_t1.device
        T_idx = (quadrupleList[:, 3] // self.time_interval).unsqueeze(1).float().to(device)
        t1 = self.weight_t1 * T_idx + self.bias_t1
        t2 = torch.sin(self.weight_t2 * T_idx + self.bias_t2)
        return t1, t2

    def rel_raw_mode(self, pre_emb, r_emb, time_embs, all_triples):
        scores_re = self.rdecoder_re1.forward(pre_emb, r_emb, time_embs, all_triples).view(-1, 2 * self.num_rels)
        return F.softmax(scores_re, dim=1)

    def rel_history_mode(self, pre_emb, r_emb, time_embs, all_triples, history_vocabulary):
        global_index = torch.Tensor(np.array(history_vocabulary.cpu(), dtype=float))
        score_global = self.rdecoder_re2.forward(pre_emb, r_emb, time_embs, all_triples, partial_embeding=global_index)
        return F.softmax(score_global, dim=1)


# ============================================================================
# PART 8: HARD MODE POST-PROCESSING CALIBRATOR
# ============================================================================

class HardModeCalibrator:
    def __init__(self, longtail_entities, num_entities, num_rels):
        self.longtail_entities = longtail_entities
        self.longtail_entities_set = set(longtail_entities)
        self.num_entities = num_entities
        self.num_rels = num_rels
        self.entity_freq = None
        self.entity_neighbors = None
        self.entity_2hop_neighbors = None
        self.mult_both = 1.0
        self.mult_one = 1.0
        self.adaptive_scale_type = 'none'
        self.adaptive_scale_factor = 1.0
        self.longtail_entity_boost = None
        self.relation_lt_boost = None
        self.neighbor_weight_1hop = 0.0
        self.neighbor_weight_2hop = 0.0
        self.boost_scale = 1.0
        self.entity_avg_ranks = {}
        self.relation_avg_ranks = {}
        self.lt_mask = None
        self.rank_change_stats = None

    def _get_true_query_type(self, subject, obj):
        s_lt = subject in self.longtail_entities_set
        o_lt = obj in self.longtail_entities_set
        if s_lt and o_lt:
            return 'both_lt'
        elif s_lt or o_lt:
            return 'one_lt'
        return 'neither'

    def _compute_score_adaptive_scale(self, score_row, adaptive_type, adaptive_factor):
        if adaptive_type == 'none':
            return 1.0
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
        for ent_id, freq in entity_freq_dict.items():
            if ent_id < self.num_entities:
                self.entity_freq[ent_id] = freq
        self.entity_neighbors = defaultdict(set)
        for quad in train_data:
            s, r, o = int(quad[0]), int(quad[1]), int(quad[2])
            self.entity_neighbors[s].add(o)
            self.entity_neighbors[o].add(s)
        self.entity_neighbors = dict(self.entity_neighbors)
        print("  - Building 2-hop neighbor graph...")
        self.entity_2hop_neighbors = {}
        for lt_ent in tqdm(self.longtail_entities_set, desc="  Building 2-hop graph", leave=False):
            if lt_ent in self.entity_neighbors:
                two_hop = set()
                for neighbor in self.entity_neighbors[lt_ent]:
                    if neighbor in self.entity_neighbors:
                        two_hop.update(self.entity_neighbors[neighbor])
                two_hop.discard(lt_ent)
                two_hop = two_hop & self.longtail_entities_set
                if two_hop:
                    self.entity_2hop_neighbors[lt_ent] = two_hop
        print(f"  - Entity frequency range: [{self.entity_freq.min():.0f}, {self.entity_freq.max():.0f}]")
        print(f"  - Entities with 1-hop neighbors: {len(self.entity_neighbors)}")
        print(f"  - LT entities with 2-hop LT neighbors: {len(self.entity_2hop_neighbors)}")

    def _compute_entity_ranks(self, val_triples, val_scores, val_targets):
        val_triples_np = val_triples.cpu().numpy() if torch.is_tensor(val_triples) else val_triples
        val_scores_cpu = val_scores.cpu() if torch.is_tensor(val_scores) else val_scores
        val_targets_cpu = val_targets.cpu() if torch.is_tensor(val_targets) else val_targets
        entity_ranks = defaultdict(list)
        relation_ranks = defaultdict(list)
        for i, triple in enumerate(val_triples_np):
            s, r, o = int(triple[0]), int(triple[1]), int(triple[2])
            target = int(val_targets_cpu[i])
            if self._get_true_query_type(s, o) == 'neither':
                continue
            score_row = val_scores_cpu[i]
            target_score = score_row[target].item()
            rank = (score_row > target_score).sum().item() + 1
            if target in self.longtail_entities_set:
                entity_ranks[target].append(rank)
                relation_ranks[r].append(rank)
        self.entity_avg_ranks = {e: np.mean(ranks) for e, ranks in entity_ranks.items()}
        self.relation_avg_ranks = {r: np.mean(ranks) for r, ranks in relation_ranks.items()}

    def _create_boost_tensors(self, boost_scale, rank_threshold_high, rank_threshold_mid, rank_threshold_low):
        entity_boost = torch.zeros(self.num_entities)
        relation_boost = torch.zeros(self.num_rels * 2)
        for ent_id, avg_rank in self.entity_avg_ranks.items():
            if avg_rank > rank_threshold_high:
                boost = min(2.0, np.log10(avg_rank) / 2.0) * boost_scale
            elif avg_rank > rank_threshold_mid:
                boost = min(1.0, np.log10(avg_rank) / 2.5) * boost_scale
            elif avg_rank > rank_threshold_low:
                boost = min(0.5, np.log10(avg_rank) / 3.0) * boost_scale
            else:
                boost = 0.0
            entity_boost[ent_id] = boost
        for rel_id, avg_rank in self.relation_avg_ranks.items():
            if avg_rank > rank_threshold_mid:
                relation_boost[rel_id] = min(0.5, np.log10(avg_rank) / 4.0) * boost_scale
        return entity_boost, relation_boost

    def _apply_calibration_hard(self, triples, scores, entity_boost, relation_boost,
                                 neighbor_w1, neighbor_w2, mult_both, mult_one,
                                 adaptive_type, adaptive_factor,
                                 track_changes=False, targets=None):
        triples_np = triples.cpu().numpy() if torch.is_tensor(triples) else triples
        device = scores.device
        calibrated = scores.clone()
        lt_mask = torch.zeros(self.num_entities, dtype=torch.bool, device=device)
        for lt_ent in self.longtail_entities_set:
            if lt_ent < self.num_entities:
                lt_mask[lt_ent] = True
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
                    for n in lt_neighbors:
                        calibrated[i, n] += boost
            if neighbor_w2 > 0 and s in self.entity_2hop_neighbors:
                neighbors_2hop = self.entity_2hop_neighbors[s]
                if neighbors_2hop:
                    boost = neighbor_w2 * effective_mult / np.sqrt(len(neighbors_2hop))
                    for n in neighbors_2hop:
                        if n < self.num_entities:
                            calibrated[i, n] += boost
            if track_changes and targets is not None:
                rank_after = (calibrated[i] > calibrated[i][target]).sum().item() + 1
                if rank_after < rank_before:
                    rank_changes['boosted'] += 1
                elif rank_after > rank_before:
                    rank_changes['lowered'] += 1
                else:
                    rank_changes['unchanged'] += 1
        return calibrated, rank_changes

    def _evaluate_calibration_hard(self, triples, scores, targets, entity_boost, relation_boost,
                                    neighbor_w1, neighbor_w2, mult_both, mult_one,
                                    adaptive_type, adaptive_factor):
        calibrated, _ = self._apply_calibration_hard(
            triples, scores, entity_boost, relation_boost,
            neighbor_w1, neighbor_w2, mult_both, mult_one,
            adaptive_type, adaptive_factor
        )
        return self._compute_lt_mrr(triples, calibrated, targets)

    def _compute_lt_mrr(self, triples, calibrated, targets):
        triples_np = triples.cpu().numpy() if torch.is_tensor(triples) else triples
        targets_cpu = targets.cpu() if torch.is_tensor(targets) else targets
        calibrated_cpu = calibrated.cpu()
        reciprocal_ranks = []
        for i, triple in enumerate(triples_np):
            s, r, o = int(triple[0]), int(triple[1]), int(triple[2])
            if self._get_true_query_type(s, o) == 'neither':
                continue
            target = int(targets_cpu[i])
            score_row = calibrated_cpu[i]
            target_score = score_row[target].item()
            rank = (score_row > target_score).sum().item() + 1
            reciprocal_ranks.append(1.0 / rank)
        return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0

    def learn_from_validation(self, val_triples, val_scores, val_targets, early_stop_ratio=0.4):
        print("\n[Calibrator-HARD] Learning calibration parameters...")
        print("  - Computing entity/relation ranks...")
        self._compute_entity_ranks(val_triples, val_scores, val_targets)
        print(f"  - LT entities with rank data: {len(self.entity_avg_ranks)}")
        print(f"  - Relations with rank data: {len(self.relation_avg_ranks)}")
        baseline_mrr = self._compute_lt_mrr(val_triples, val_scores, val_targets)
        print(f"  - Baseline LT MRR (no calibration): {baseline_mrr:.6f}")
        # Grid search parameters (user-specified)
        boost_scales = [0.1, 0.5, 1.5, 3.5]
        rank_thresholds = [(100, 50, 20), (50, 25, 10)]
        neighbor_w1_opts = [0.2, 0.5]
        neighbor_w2_opts = [0.2, 0.5]
        adaptive_type_opts = ["range", "iqr"]
        adaptive_factor_opts = [0.1, 0.5, 1.5]
        mult_both_opts = [0.2, 0.5, 1.5]
        mult_one_opts = [0.2, 0.5, 1.5]
        print("  - Pre-computing boost tensors...")
        boost_tensor_cache = {}
        for boost_scale in boost_scales:
            for rt in rank_thresholds:
                boost_tensor_cache[(boost_scale, rt)] = self._create_boost_tensors(boost_scale, rt[0], rt[1], rt[2])
        total_combos = (len(boost_scales) * len(rank_thresholds) * len(neighbor_w1_opts) *
                        len(neighbor_w2_opts) * len(adaptive_type_opts) * len(adaptive_factor_opts) *
                        len(mult_both_opts) * len(mult_one_opts))
        print(f"  - Grid search over {total_combos} parameter combinations (HARD mode)...")
        early_stop_threshold = int(total_combos * early_stop_ratio)
        print(f"  - Early stopping after {early_stop_threshold} iterations without improvement ({early_stop_ratio*100:.0f}%)")
        best_mrr = baseline_mrr
        best_params = None
        iterations_since_improvement = 0
        current_iteration = 0
        early_stopped = False
        with tqdm(total=total_combos, desc="  Grid search") as pbar:
            for boost_scale in boost_scales:
                if early_stopped:
                    break
                for rt in rank_thresholds:
                    if early_stopped:
                        break
                    entity_boost, relation_boost = boost_tensor_cache[(boost_scale, rt)]
                    for neighbor_w1 in neighbor_w1_opts:
                        if early_stopped:
                            break
                        for neighbor_w2 in neighbor_w2_opts:
                            if early_stopped:
                                break
                            for adaptive_type in adaptive_type_opts:
                                if early_stopped:
                                    break
                                for adaptive_factor in adaptive_factor_opts:
                                    if early_stopped:
                                        break
                                    for mult_both in mult_both_opts:
                                        if early_stopped:
                                            break
                                        for mult_one in mult_one_opts:
                                            mrr = self._evaluate_calibration_hard(
                                                val_triples, val_scores, val_targets,
                                                entity_boost, relation_boost,
                                                neighbor_w1, neighbor_w2,
                                                mult_both, mult_one,
                                                adaptive_type, adaptive_factor
                                            )
                                            current_iteration += 1
                                            if mrr > best_mrr:
                                                best_mrr = mrr
                                                iterations_since_improvement = 0
                                                best_params = {
                                                    'boost_scale': boost_scale,
                                                    'rank_thresholds': rt,
                                                    'neighbor_w1': neighbor_w1,
                                                    'neighbor_w2': neighbor_w2,
                                                    'adaptive_type': adaptive_type,
                                                    'adaptive_factor': adaptive_factor,
                                                    'mult_both': mult_both,
                                                    'mult_one': mult_one,
                                                    'entity_boost': entity_boost.clone(),
                                                    'relation_boost': relation_boost.clone(),
                                                    'best_mrr': best_mrr
                                                }
                                            else:
                                                iterations_since_improvement += 1
                                            pbar.update(1)
                                            pbar.set_postfix({'best_MRR': f'{best_mrr:.4f}', 'no_improv': iterations_since_improvement})
                                            if iterations_since_improvement >= early_stop_threshold:
                                                print(f"\n  - Early stopping at iteration {current_iteration}/{total_combos}")
                                                early_stopped = True
                                                break
        if best_params is None:
            print("  - No improvement found, disabling calibration")
            self._set_no_calibration()
        else:
            self._apply_best_params(best_params, baseline_mrr)
        self._precompute_masks()

    def _set_no_calibration(self):
        self.longtail_entity_boost = torch.zeros(self.num_entities)
        self.relation_lt_boost = torch.zeros(self.num_rels * 2)
        self.neighbor_weight_1hop = 0.0
        self.neighbor_weight_2hop = 0.0
        self.mult_both = 1.0
        self.mult_one = 1.0
        self.adaptive_scale_type = 'none'
        self.adaptive_scale_factor = 1.0

    def _apply_best_params(self, best_params, baseline_mrr):
        self.longtail_entity_boost = best_params['entity_boost']
        self.relation_lt_boost = best_params['relation_boost']
        self.neighbor_weight_1hop = best_params['neighbor_w1']
        self.neighbor_weight_2hop = best_params['neighbor_w2']
        self.boost_scale = best_params['boost_scale']
        self.adaptive_scale_type = best_params['adaptive_type']
        self.adaptive_scale_factor = best_params['adaptive_factor']
        self.mult_both = best_params['mult_both']
        self.mult_one = best_params['mult_one']
        best_mrr = best_params['best_mrr']
        improvement_pct = 100 * (best_mrr - baseline_mrr) / baseline_mrr if baseline_mrr > 0 else 0
        print(f"\n  - Best LT MRR: {best_mrr:.6f} (improvement: +{improvement_pct:.2f}%)")
        self._print_optimal_params(best_params, baseline_mrr, best_mrr)

    def _print_optimal_params(self, best_params, baseline_mrr, best_mrr):
        rt = best_params['rank_thresholds']
        print("\n" + "=" * 70)
        print("OPTIMAL CALIBRATION PARAMETERS (HARD MODE)")
        print("=" * 70)
        print(f"Query type multipliers (DATA LEAKAGE WARNING):")
        print(f"  both_lt multiplier: {best_params['mult_both']:.2f}")
        print(f"  one_lt multiplier: {best_params['mult_one']:.2f}")
        print(f"\nScore-adaptive scaling:")
        print(f"  Type: {best_params['adaptive_type']}")
        print(f"  Factor: {best_params['adaptive_factor']}")
        print(f"\nBoost parameters:")
        print(f"  Boost scale: {best_params['boost_scale']}")
        print(f"  Rank thresholds: high={rt[0]}, mid={rt[1]}, low={rt[2]}")
        print(f"  Neighbor weight (1-hop): {best_params['neighbor_w1']}")
        print(f"  Neighbor weight (2-hop): {best_params['neighbor_w2']}")
        print(f"\nValidation LT MRR: {baseline_mrr:.4f} -> {best_mrr:.4f}")
        print("=" * 70 + "\n")

    def _precompute_masks(self):
        self.lt_mask = torch.zeros(self.num_entities, dtype=torch.bool)
        for lt_ent in self.longtail_entities_set:
            if lt_ent < self.num_entities:
                self.lt_mask[lt_ent] = True

    def calibrate_scores(self, triples, scores, num_rels, targets=None, track_changes=False):
        if self.longtail_entity_boost is None:
            if track_changes:
                self.rank_change_stats = {'boosted': 0, 'lowered': 0, 'unchanged': 0,
                                          'skipped': len(triples), 'total': len(triples)}
            return scores
        calibrated, rank_changes = self._apply_calibration_hard(
            triples, scores, self.longtail_entity_boost, self.relation_lt_boost,
            self.neighbor_weight_1hop, self.neighbor_weight_2hop,
            self.mult_both, self.mult_one,
            self.adaptive_scale_type, self.adaptive_scale_factor,
            track_changes=track_changes, targets=targets
        )
        if track_changes:
            self.rank_change_stats = rank_changes
        return calibrated

    def get_rank_change_stats(self):
        return self.rank_change_stats


# ============================================================================
# PART 9: EVALUATION FUNCTIONS
# ============================================================================

def get_total_rank(test_triples, score, all_ans, eval_bz):
    num_triples = len(test_triples)
    n_batch = (num_triples + eval_bz - 1) // eval_bz
    rank, filter_rank = [], []
    for idx in range(n_batch):
        batch_start = idx * eval_bz
        batch_end = min(num_triples, (idx + 1) * eval_bz)
        triples_batch = test_triples[batch_start:batch_end, :]
        score_batch = score[batch_start:batch_end, :]
        target = test_triples[batch_start:batch_end, 2]
        rank.append(sort_and_rank(score_batch, target))
        filter_score_batch = filter_score(triples_batch, score_batch.clone(), all_ans)
        filter_rank.append(sort_and_rank(filter_score_batch, target))
    rank = torch.cat(rank) + 1
    filter_rank = torch.cat(filter_rank) + 1
    mrr = torch.mean(1.0 / rank.float())
    filter_mrr = torch.mean(1.0 / filter_rank.float())
    return filter_mrr.item(), mrr.item(), rank, filter_rank


def compute_metrics_from_ranks(rank_list):
    hits = [1, 3, 10]
    total_rank = torch.cat(rank_list)
    mrr = torch.mean(1.0 / total_rank.float())
    hit_results = [torch.mean((total_rank <= hit).float()).item() for hit in hits]
    return mrr.item(), hit_results


def print_metrics_table(title, raw_mrr, raw_hits, filter_mrr, filter_hits, num_queries=None):
    print("\n" + "=" * 70)
    print(title)
    if num_queries is not None:
        print(f"Number of queries: {num_queries}")
    print("=" * 70)
    print(f"{'Metric':<20} {'Raw':>20} {'Filtered':>20}")
    print("-" * 70)
    print(f"{'MRR':<20} {raw_mrr:>20.6f} {filter_mrr:>20.6f}")
    print(f"{'Hits@1':<20} {raw_hits[0]:>20.6f} {filter_hits[0]:>20.6f}")
    print(f"{'Hits@3':<20} {raw_hits[1]:>20.6f} {filter_hits[1]:>20.6f}")
    print(f"{'Hits@10':<20} {raw_hits[2]:>20.6f} {filter_hits[2]:>20.6f}")
    print("=" * 70)


def print_comparison_table(title, before_results, after_results):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)
    print("\n>>> OVERALL ENTITY PREDICTION")
    print("-" * 90)
    print(f"{'Metric':<15} {'Before (Raw)':>15} {'Before (Filt)':>15} {'After (Raw)':>15} {'After (Filt)':>15}")
    print("-" * 90)
    metrics = ['MRR', 'Hits@1', 'Hits@3', 'Hits@10']
    for i, m in enumerate(metrics):
        if m == 'MRR':
            b_raw, b_filt = before_results['mrr_raw'], before_results['mrr']
            a_raw, a_filt = after_results['mrr_raw'], after_results['mrr']
        else:
            idx = i - 1
            b_raw, b_filt = before_results['hits_raw'][idx], before_results['hits'][idx]
            a_raw, a_filt = after_results['hits_raw'][idx], after_results['hits'][idx]
        print(f"{m:<15} {b_raw:>15.6f} {b_filt:>15.6f} {a_raw:>15.6f} {a_filt:>15.6f}")
    print("\n>>> LONG-TAIL ENTITY PREDICTION")
    print("-" * 90)
    print(f"{'Metric':<15} {'Before (Raw)':>15} {'Before (Filt)':>15} {'After (Raw)':>15} {'After (Filt)':>15}")
    print("-" * 90)
    if before_results.get('mrr_lt', 0) > 0:
        for i, m in enumerate(metrics):
            if m == 'MRR':
                b_raw, b_filt = before_results['mrr_lt_raw'], before_results['mrr_lt']
                a_raw, a_filt = after_results['mrr_lt_raw'], after_results['mrr_lt']
            else:
                idx = i - 1
                b_raw, b_filt = before_results['hits_lt_raw'][idx], before_results['hits_lt'][idx]
                a_raw, a_filt = after_results['hits_lt_raw'][idx], after_results['hits_lt'][idx]
            print(f"{m:<15} {b_raw:>15.6f} {b_filt:>15.6f} {a_raw:>15.6f} {a_filt:>15.6f}")
    else:
        print("No long-tail queries found.")
    print("=" * 90 + "\n")


def print_rank_change_stats(rank_stats, total_queries):
    print("\n" + "=" * 70)
    print("POST-PROCESSING RANK CHANGE STATISTICS")
    print("=" * 70)
    boosted = rank_stats.get('boosted', 0)
    lowered = rank_stats.get('lowered', 0)
    unchanged = rank_stats.get('unchanged', 0)
    skipped = rank_stats.get('skipped', 0)
    modified = boosted + lowered + unchanged
    print(f"\nTotal test queries: {total_queries}")
    print(f"\n--- Queries by calibration status ---")
    print(f"  Skipped (not calibrated):     {skipped:>8} ({100.0*skipped/total_queries:>6.2f}%)")
    print(f"  Modified (calibrated):        {modified:>8} ({100.0*modified/total_queries:>6.2f}%)")
    print(f"\n--- Effect on modified queries ---")
    if modified > 0:
        print(f"  True object BOOSTED UP:       {boosted:>8} ({100.0*boosted/modified:>6.2f}% of modified)")
        print(f"  True object LOWERED DOWN:     {lowered:>8} ({100.0*lowered/modified:>6.2f}% of modified)")
        print(f"  True object UNCHANGED:        {unchanged:>8} ({100.0*unchanged/modified:>6.2f}% of modified)")
    else:
        print("  No queries were modified")
    print("=" * 70 + "\n")


# ============================================================================
# PART 10: WILCOXON SIGNED-RANK TEST
# ============================================================================

def compute_wilcoxon_test(ranks_before, ranks_after, name=""):
    """
    Perform Wilcoxon signed-rank test comparing ranks before and after post-processing.
    Returns test results including statistic, p-value, and effect size.
    """
    ranks_before_np = ranks_before.cpu().numpy().astype(float)
    ranks_after_np = ranks_after.cpu().numpy().astype(float)
    differences = ranks_before_np - ranks_after_np
    n_improved = np.sum(differences > 0)
    n_worsened = np.sum(differences < 0)
    n_unchanged = np.sum(differences == 0)
    median_before = np.median(ranks_before_np)
    median_after = np.median(ranks_after_np)
    mean_change = np.mean(differences)
    non_zero_diff = differences[differences != 0]
    if len(non_zero_diff) < 10:
        return {
            'name': name,
            'n_samples': len(ranks_before_np),
            'n_improved': int(n_improved),
            'n_worsened': int(n_worsened),
            'n_unchanged': int(n_unchanged),
            'median_before': median_before,
            'median_after': median_after,
            'mean_change': mean_change,
            'statistic': None,
            'p_value': None,
            'effect_size': None,
            'significant': None,
            'interpretation': "Insufficient non-zero differences for test (n < 10)"
        }
    try:
        statistic, p_value = stats.wilcoxon(ranks_before_np, ranks_after_np, alternative='two-sided')
        n = len(non_zero_diff)
        z_score = stats.norm.ppf(1 - p_value / 2)
        effect_size = abs(z_score) / np.sqrt(n)
        if effect_size < 0.3:
            effect_interpretation = "small"
        elif effect_size < 0.5:
            effect_interpretation = "medium"
        else:
            effect_interpretation = "large"
        significant = p_value < 0.05
        if significant:
            if mean_change > 0:
                interpretation = f"Significant improvement (p={p_value:.4e}, {effect_interpretation} effect)"
            else:
                interpretation = f"Significant degradation (p={p_value:.4e}, {effect_interpretation} effect)"
        else:
            interpretation = f"No significant difference (p={p_value:.4e})"
        return {
            'name': name,
            'n_samples': len(ranks_before_np),
            'n_improved': int(n_improved),
            'n_worsened': int(n_worsened),
            'n_unchanged': int(n_unchanged),
            'median_before': median_before,
            'median_after': median_after,
            'mean_change': mean_change,
            'statistic': statistic,
            'p_value': p_value,
            'effect_size': effect_size,
            'significant': significant,
            'interpretation': interpretation
        }
    except Exception as e:
        return {
            'name': name,
            'n_samples': len(ranks_before_np),
            'n_improved': int(n_improved),
            'n_worsened': int(n_worsened),
            'n_unchanged': int(n_unchanged),
            'median_before': median_before,
            'median_after': median_after,
            'mean_change': mean_change,
            'statistic': None,
            'p_value': None,
            'effect_size': None,
            'significant': None,
            'interpretation': f"Test failed: {str(e)}"
        }


def print_wilcoxon_results(results):
    """Print Wilcoxon test results in formatted table."""
    print("\n" + "=" * 80)
    print(f"WILCOXON SIGNED-RANK TEST: {results['name']}")
    print("=" * 80)
    print(f"\n{'Sample Statistics':<40}")
    print("-" * 80)
    print(f"  Total samples:                    {results['n_samples']:>10}")
    print(f"  Improved (rank decreased):        {results['n_improved']:>10} ({100*results['n_improved']/results['n_samples']:.2f}%)")
    print(f"  Worsened (rank increased):        {results['n_worsened']:>10} ({100*results['n_worsened']/results['n_samples']:.2f}%)")
    print(f"  Unchanged:                        {results['n_unchanged']:>10} ({100*results['n_unchanged']/results['n_samples']:.2f}%)")
    print(f"\n{'Rank Statistics':<40}")
    print("-" * 80)
    print(f"  Median rank (before):             {results['median_before']:>10.2f}")
    print(f"  Median rank (after):              {results['median_after']:>10.2f}")
    print(f"  Mean rank change (before-after):  {results['mean_change']:>10.4f}")
    print(f"\n{'Test Results':<40}")
    print("-" * 80)
    if results['statistic'] is not None:
        print(f"  Test statistic (W):               {results['statistic']:>10.2f}")
        print(f"  P-value:                          {results['p_value']:>10.4e}")
        print(f"  Effect size (r):                  {results['effect_size']:>10.4f}")
        print(f"  Significant (p < 0.05):           {'Yes' if results['significant'] else 'No':>10}")
    print(f"\n{'Interpretation':<40}")
    print("-" * 80)
    print(f"  {results['interpretation']}")
    print("=" * 80 + "\n")


# ============================================================================
# PART 11: EVALUATION PIPELINE
# ============================================================================

def evaluate_model(model, history_list, test_list, num_rels, num_nodes,
                   all_ans_list, static_graph, time_list,
                   history_time_nogt, longtail_entities, calibrator=None,
                   apply_calibration=False, dataset_name="Test", args=None,
                   track_rank_changes=False, collect_ranks_for_test=False):
    """Evaluate model on test/validation data (CPU only)."""
    ranks_raw, ranks_filter = [], []
    ranks_raw_lt, ranks_filter_lt = [], []
    all_test_triples, all_final_scores, all_longtail_masks, all_ans_collected = [], [], [], []
    total_rank_changes = {'boosted': 0, 'lowered': 0, 'unchanged': 0, 'skipped': 0, 'total': 0}
    all_individual_ranks_raw = []
    all_individual_ranks_filter = []
    all_individual_ranks_raw_lt = []
    all_individual_ranks_filter_lt = []
    all_lt_indices = []
    device = torch.device('cpu')
    model.eval()
    input_list = [snap for snap in history_list[-args.test_history_len:]]
    if args.multi_step:
        all_tail_seq = sp.load_npz('./data/{}/history/tail_history_{}.npz'.format(args.dataset, history_time_nogt))
        all_rel_seq = sp.load_npz('./data/{}/history/rel_history_{}.npz'.format(args.dataset, history_time_nogt))
    for time_idx, test_snap in enumerate(tqdm(test_list, desc=f"Evaluating {dataset_name}")):
        history_glist = [build_sub_graph(num_nodes, num_rels, g, False, -1) for g in input_list]
        test_triples_input = torch.LongTensor(test_snap).to(device)
        histroy_data = test_triples_input
        inverse_histroy_data = histroy_data[:, [2, 1, 0, 3]]
        inverse_histroy_data[:, 1] = inverse_histroy_data[:, 1] + num_rels
        histroy_data = torch.cat([histroy_data, inverse_histroy_data])
        histroy_data_np = histroy_data.cpu().numpy()
        if args.multi_step:
            seq_idx = histroy_data_np[:, 0] * num_rels * 2 + histroy_data_np[:, 1]
            tail_seq = torch.Tensor(all_tail_seq[seq_idx].todense())
            one_hot_tail_seq = tail_seq.masked_fill(tail_seq != 0, 1)
            rel_seq_idx = histroy_data_np[:, 0] * num_nodes + histroy_data_np[:, 2]
            rel_seq = torch.Tensor(all_rel_seq[rel_seq_idx].todense())
            one_hot_rel_seq = rel_seq.masked_fill(rel_seq != 0, 1)
        else:
            all_tail_seq_t = sp.load_npz('./data/{}/history/tail_history_{}.npz'.format(args.dataset, time_list[time_idx]))
            seq_idx = histroy_data_np[:, 0] * num_rels * 2 + histroy_data_np[:, 1]
            tail_seq = torch.Tensor(all_tail_seq_t[seq_idx].todense())
            one_hot_tail_seq = tail_seq.masked_fill(tail_seq != 0, 1)
            all_rel_seq_t = sp.load_npz('./data/{}/history/rel_history_{}.npz'.format(args.dataset, time_list[time_idx]))
            rel_seq_idx = histroy_data_np[:, 0] * num_nodes + histroy_data_np[:, 2]
            rel_seq = torch.Tensor(all_rel_seq_t[rel_seq_idx].todense())
            one_hot_rel_seq = rel_seq.masked_fill(rel_seq != 0, 1)
        test_triples, final_score, _ = model.predict(
            history_glist, num_rels, static_graph,
            test_triples_input, one_hot_tail_seq,
            one_hot_rel_seq, False
        )
        all_test_triples.append(test_triples.clone())
        all_final_scores.append(final_score.clone())
        all_ans_collected.append(all_ans_list[time_idx])
        longtail_mask = torch.tensor([
            (triple[0].item() in longtail_entities) or (triple[2].item() in longtail_entities)
            for triple in test_triples.cpu()
        ])
        all_longtail_masks.append(longtail_mask)
        if apply_calibration and calibrator is not None:
            targets = test_triples[:, 2]
            eval_score = calibrator.calibrate_scores(
                test_triples, final_score, num_rels,
                targets=targets, track_changes=track_rank_changes
            )
            if track_rank_changes:
                snap_stats = calibrator.get_rank_change_stats()
                if snap_stats:
                    for k in total_rank_changes:
                        total_rank_changes[k] += snap_stats.get(k, 0)
        else:
            eval_score = final_score
        _, _, rank_raw, rank_filter = get_total_rank(test_triples, eval_score, all_ans_list[time_idx], eval_bz=1000)
        ranks_raw.append(rank_raw)
        ranks_filter.append(rank_filter)
        if collect_ranks_for_test:
            all_individual_ranks_raw.append(rank_raw)
            all_individual_ranks_filter.append(rank_filter)
            lt_indices = longtail_mask.nonzero(as_tuple=True)[0]
            all_lt_indices.append(lt_indices + sum(len(r) for r in all_individual_ranks_raw[:-1]))
        if longtail_mask.any():
            lt_test_triples = test_triples[longtail_mask]
            lt_eval_score = eval_score[longtail_mask]
            lt_all_ans = {}
            for i, is_lt in enumerate(longtail_mask):
                if is_lt:
                    triple = test_triples[i].cpu()
                    h, r = triple[0].item(), triple[1].item()
                    if h in all_ans_list[time_idx] and r in all_ans_list[time_idx][h]:
                        if h not in lt_all_ans:
                            lt_all_ans[h] = {}
                        lt_all_ans[h][r] = all_ans_list[time_idx][h][r]
            _, _, rank_raw_lt, rank_filter_lt = get_total_rank(lt_test_triples, lt_eval_score, lt_all_ans, eval_bz=1000)
            ranks_raw_lt.append(rank_raw_lt)
            ranks_filter_lt.append(rank_filter_lt)
            if collect_ranks_for_test:
                all_individual_ranks_raw_lt.append(rank_raw_lt)
                all_individual_ranks_filter_lt.append(rank_filter_lt)
        if args.multi_step:
            predicted_snap = construct_snap(test_triples, num_nodes, num_rels, final_score, args.topk)
            if len(predicted_snap):
                input_list.pop(0)
                input_list.append(predicted_snap)
        else:
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
    total_queries = sum(len(t) for t in all_test_triples)
    lt_queries = sum(m.sum().item() for m in all_longtail_masks)
    results = {
        'mrr': mrr_filter, 'mrr_raw': mrr_raw, 'hits': hits_filter, 'hits_raw': hits_raw,
        'mrr_lt': mrr_filter_lt, 'mrr_lt_raw': mrr_raw_lt, 'hits_lt': hits_filter_lt, 'hits_lt_raw': hits_raw_lt,
        'total_queries': total_queries, 'lt_queries': lt_queries,
        'all_triples': all_test_triples, 'all_scores': all_final_scores,
        'all_masks': all_longtail_masks, 'all_ans': all_ans_collected,
        'rank_changes': total_rank_changes if track_rank_changes else None
    }
    if collect_ranks_for_test:
        results['all_ranks_raw'] = torch.cat(all_individual_ranks_raw) if all_individual_ranks_raw else torch.tensor([])
        results['all_ranks_filter'] = torch.cat(all_individual_ranks_filter) if all_individual_ranks_filter else torch.tensor([])
        results['all_ranks_raw_lt'] = torch.cat(all_individual_ranks_raw_lt) if all_individual_ranks_raw_lt else torch.tensor([])
        results['all_ranks_filter_lt'] = torch.cat(all_individual_ranks_filter_lt) if all_individual_ranks_filter_lt else torch.tensor([])
    return results


def construct_snap(test_triples, num_nodes, num_rels, final_score, topK):
    _, indices = torch.sort(final_score, dim=1, descending=True)
    top_indices = indices[:, :topK]
    predict_triples = []
    for idx in range(len(test_triples)):
        for index in top_indices[idx]:
            h, r = test_triples[idx][0], test_triples[idx][1]
            if r < num_rels:
                predict_triples.append([test_triples[idx][0], r, index, test_triples[idx][3]])
            else:
                predict_triples.append([index, r - num_rels, test_triples[idx][0], test_triples[idx][3]])
    return np.array(predict_triples, dtype=int)


# ============================================================================
# PART 12: RESULTS SAVING
# ============================================================================

def save_results_to_file(args, test_results_before, test_results_after, calibrator_params, longtail_stats, wilcoxon_results):
    filename = f"4_tirgn_results_{args.dataset}_hard_gpu{args.gpu}.txt"
    filepath = os.path.join("./", filename)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(filepath, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("TEMPORAL KNOWLEDGE GRAPH REASONING - TEST RESULTS (HARD MODE)\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Timestamp: {timestamp}\n\n")
        f.write("-" * 80 + "\n")
        f.write("CONFIGURATION\n")
        f.write("-" * 80 + "\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"GPU (for checkpoint name): {args.gpu}\n")
        f.write(f"History length (test): {args.test_history_len}\n")
        f.write(f"History rate: {args.history_rate}\n")
        f.write(f"Long-tail threshold: {args.longtail_threshold}\n")
        f.write(f"Early stop ratio: {args.early_stop_ratio}\n")
        f.write(f"Multi-step: {args.multi_step}\n\n")
        f.write("-" * 80 + "\n")
        f.write("LONG-TAIL STATISTICS\n")
        f.write("-" * 80 + "\n")
        if longtail_stats:
            f.write(f"Long-tail entities: {longtail_stats.get('num_lt_entities', 'N/A')}\n")
            f.write(f"Total entities: {longtail_stats.get('num_entities', 'N/A')}\n")
            f.write(f"Long-tail percentage: {longtail_stats.get('lt_percentage', 'N/A'):.2f}%\n\n")
        f.write("-" * 80 + "\n")
        f.write("TEST RESULTS - BEFORE POST-PROCESSING\n")
        f.write("-" * 80 + "\n\n")
        f.write(">>> Overall Entity Prediction\n")
        f.write(f"{'Metric':<15} {'Raw':>15} {'Filtered':>15}\n")
        f.write("-" * 45 + "\n")
        f.write(f"{'MRR':<15} {test_results_before['mrr_raw']:>15.6f} {test_results_before['mrr']:>15.6f}\n")
        f.write(f"{'Hits@1':<15} {test_results_before['hits_raw'][0]:>15.6f} {test_results_before['hits'][0]:>15.6f}\n")
        f.write(f"{'Hits@3':<15} {test_results_before['hits_raw'][1]:>15.6f} {test_results_before['hits'][1]:>15.6f}\n")
        f.write(f"{'Hits@10':<15} {test_results_before['hits_raw'][2]:>15.6f} {test_results_before['hits'][2]:>15.6f}\n")
        f.write(f"Total queries: {test_results_before['total_queries']}\n\n")
        f.write(">>> Long-tail Entity Prediction\n")
        f.write(f"{'Metric':<15} {'Raw':>15} {'Filtered':>15}\n")
        f.write("-" * 45 + "\n")
        f.write(f"{'MRR':<15} {test_results_before['mrr_lt_raw']:>15.6f} {test_results_before['mrr_lt']:>15.6f}\n")
        f.write(f"{'Hits@1':<15} {test_results_before['hits_lt_raw'][0]:>15.6f} {test_results_before['hits_lt'][0]:>15.6f}\n")
        f.write(f"{'Hits@3':<15} {test_results_before['hits_lt_raw'][1]:>15.6f} {test_results_before['hits_lt'][1]:>15.6f}\n")
        f.write(f"{'Hits@10':<15} {test_results_before['hits_lt_raw'][2]:>15.6f} {test_results_before['hits_lt'][2]:>15.6f}\n")
        f.write(f"Long-tail queries: {test_results_before['lt_queries']}\n\n")
        f.write("-" * 80 + "\n")
        f.write("TEST RESULTS - AFTER POST-PROCESSING (HARD MODE)\n")
        f.write("-" * 80 + "\n\n")
        f.write(">>> Overall Entity Prediction\n")
        f.write(f"{'Metric':<15} {'Raw':>15} {'Filtered':>15}\n")
        f.write("-" * 45 + "\n")
        f.write(f"{'MRR':<15} {test_results_after['mrr_raw']:>15.6f} {test_results_after['mrr']:>15.6f}\n")
        f.write(f"{'Hits@1':<15} {test_results_after['hits_raw'][0]:>15.6f} {test_results_after['hits'][0]:>15.6f}\n")
        f.write(f"{'Hits@3':<15} {test_results_after['hits_raw'][1]:>15.6f} {test_results_after['hits'][1]:>15.6f}\n")
        f.write(f"{'Hits@10':<15} {test_results_after['hits_raw'][2]:>15.6f} {test_results_after['hits'][2]:>15.6f}\n")
        f.write(f"Total queries: {test_results_after['total_queries']}\n\n")
        f.write(">>> Long-tail Entity Prediction\n")
        f.write(f"{'Metric':<15} {'Raw':>15} {'Filtered':>15}\n")
        f.write("-" * 45 + "\n")
        f.write(f"{'MRR':<15} {test_results_after['mrr_lt_raw']:>15.6f} {test_results_after['mrr_lt']:>15.6f}\n")
        f.write(f"{'Hits@1':<15} {test_results_after['hits_lt_raw'][0]:>15.6f} {test_results_after['hits_lt'][0]:>15.6f}\n")
        f.write(f"{'Hits@3':<15} {test_results_after['hits_lt_raw'][1]:>15.6f} {test_results_after['hits_lt'][1]:>15.6f}\n")
        f.write(f"{'Hits@10':<15} {test_results_after['hits_lt_raw'][2]:>15.6f} {test_results_after['hits_lt'][2]:>15.6f}\n")
        f.write(f"Long-tail queries: {test_results_after['lt_queries']}\n\n")
        f.write("-" * 80 + "\n")
        f.write("IMPROVEMENT SUMMARY (AFTER - BEFORE)\n")
        f.write("-" * 80 + "\n\n")
        f.write(">>> Overall Entity Prediction\n")
        f.write(f"{'Metric':<15} {'Raw Delta':>15} {'Filtered Delta':>15}\n")
        f.write("-" * 45 + "\n")
        for i, m in enumerate(['MRR', 'Hits@1', 'Hits@3', 'Hits@10']):
            if m == 'MRR':
                raw_diff = test_results_after['mrr_raw'] - test_results_before['mrr_raw']
                filt_diff = test_results_after['mrr'] - test_results_before['mrr']
            else:
                idx = i - 1
                raw_diff = test_results_after['hits_raw'][idx] - test_results_before['hits_raw'][idx]
                filt_diff = test_results_after['hits'][idx] - test_results_before['hits'][idx]
            f.write(f"{m:<15} {raw_diff:>+15.6f} {filt_diff:>+15.6f}\n")
        f.write("\n>>> Long-tail Entity Prediction\n")
        f.write(f"{'Metric':<15} {'Raw Delta':>15} {'Filtered Delta':>15}\n")
        f.write("-" * 45 + "\n")
        for i, m in enumerate(['MRR', 'Hits@1', 'Hits@3', 'Hits@10']):
            if m == 'MRR':
                raw_diff = test_results_after['mrr_lt_raw'] - test_results_before['mrr_lt_raw']
                filt_diff = test_results_after['mrr_lt'] - test_results_before['mrr_lt']
            else:
                idx = i - 1
                raw_diff = test_results_after['hits_lt_raw'][idx] - test_results_before['hits_lt_raw'][idx]
                filt_diff = test_results_after['hits_lt'][idx] - test_results_before['hits_lt'][idx]
            f.write(f"{m:<15} {raw_diff:>+15.6f} {filt_diff:>+15.6f}\n")
        f.write("\n")
        if test_results_after.get('rank_changes'):
            f.write("-" * 80 + "\n")
            f.write("RANK CHANGE STATISTICS\n")
            f.write("-" * 80 + "\n\n")
            rc = test_results_after['rank_changes']
            total = rc['total']
            boosted, lowered, unchanged, skipped = rc['boosted'], rc['lowered'], rc['unchanged'], rc['skipped']
            modified = boosted + lowered + unchanged
            f.write(f"Total test queries: {total}\n")
            f.write(f"Skipped (not calibrated): {skipped} ({100.0*skipped/total:.2f}%)\n")
            f.write(f"Modified (calibrated): {modified} ({100.0*modified/total:.2f}%)\n\n")
            if modified > 0:
                f.write(f"Effect on modified queries:\n")
                f.write(f"  Boosted up: {boosted} ({100.0*boosted/modified:.2f}% of modified)\n")
                f.write(f"  Lowered down: {lowered} ({100.0*lowered/modified:.2f}% of modified)\n")
                f.write(f"  Unchanged: {unchanged} ({100.0*unchanged/modified:.2f}% of modified)\n\n")
        # Wilcoxon test results
        f.write("-" * 80 + "\n")
        f.write("WILCOXON SIGNED-RANK TEST RESULTS\n")
        f.write("-" * 80 + "\n\n")
        for w_result in wilcoxon_results:
            f.write(f">>> {w_result['name']}\n")
            f.write(f"  Total samples: {w_result['n_samples']}\n")
            f.write(f"  Improved (rank decreased): {w_result['n_improved']} ({100*w_result['n_improved']/w_result['n_samples']:.2f}%)\n")
            f.write(f"  Worsened (rank increased): {w_result['n_worsened']} ({100*w_result['n_worsened']/w_result['n_samples']:.2f}%)\n")
            f.write(f"  Unchanged: {w_result['n_unchanged']} ({100*w_result['n_unchanged']/w_result['n_samples']:.2f}%)\n")
            f.write(f"  Median rank (before): {w_result['median_before']:.2f}\n")
            f.write(f"  Median rank (after): {w_result['median_after']:.2f}\n")
            f.write(f"  Mean rank change: {w_result['mean_change']:.4f}\n")
            if w_result['statistic'] is not None:
                f.write(f"  Test statistic (W): {w_result['statistic']:.2f}\n")
                f.write(f"  P-value: {w_result['p_value']:.4e}\n")
                f.write(f"  Effect size (r): {w_result['effect_size']:.4f}\n")
                f.write(f"  Significant (p < 0.05): {'Yes' if w_result['significant'] else 'No'}\n")
            f.write(f"  Interpretation: {w_result['interpretation']}\n\n")
        if calibrator_params:
            f.write("-" * 80 + "\n")
            f.write("CALIBRATOR PARAMETERS (HARD MODE)\n")
            f.write("-" * 80 + "\n\n")
            for key, value in calibrator_params.items():
                if not key.startswith('entity_boost') and not key.startswith('relation_boost'):
                    f.write(f"{key}: {value}\n")
        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF RESULTS\n")
        f.write("=" * 80 + "\n")
    print(f"\n[INFO] Results saved to: {filepath}")
    return filepath


# ============================================================================
# PART 13: MAIN EXPERIMENT
# ============================================================================

def run_experiment(args):
    print("\n" + "=" * 70)
    print("LOADING DATASET")
    print("=" * 70)
    data = load_data(args.dataset)
    train_list, train_times = split_by_time(data.train)
    valid_list, valid_times = split_by_time(data.valid)
    test_list, test_times = split_by_time(data.test)
    num_nodes = data.num_nodes
    num_rels = data.num_rels
    print("\n" + "=" * 70)
    print("DATASET STATISTICS")
    print("=" * 70)
    print(f"Entities: {num_nodes}, Relations: {num_rels}")
    print(f"Train: {len(data.train)}, Valid: {len(data.valid)}, Test: {len(data.test)}")
    entity_freq = count_entity_frequencies(data.train, num_rels)
    longtail_entities = identify_longtail_entities(entity_freq, args.longtail_threshold, num_nodes)
    longtail_stats = {
        'num_lt_entities': len(longtail_entities),
        'num_entities': num_nodes,
        'lt_percentage': 100.0 * len(longtail_entities) / num_nodes
    }
    print("\n" + "=" * 70)
    print("LONG-TAIL ENTITY ANALYSIS")
    print("=" * 70)
    print(f"Threshold: {args.longtail_threshold}, LT entities: {len(longtail_entities)} ({longtail_stats['lt_percentage']:.2f}%)")
    num_times = len(train_list) + len(valid_list) + len(test_list) + (1 if args.dataset == "ICEWS14s" else 0)
    time_interval = train_times[1] - train_times[0]
    history_val_time_nogt = valid_times[0]
    history_test_time_nogt = test_times[0]
    all_ans_list_test = load_all_answers_for_time_filter(data.test, num_rels, num_nodes, False)
    all_ans_list_valid = load_all_answers_for_time_filter(data.valid, num_rels, num_nodes, False)
    model_name = "gl_rate_{}-{}-convgcn-timeconvtranse-ly2-dilate1-his{}-weight_0.5-discount_1.0-angle_14-dp0.2_0.2_0.2_0.2-gpu{}-checkpoint".format(
        args.history_rate, args.dataset, args.train_history_len, args.gpu
    )
    model_state_file = os.path.join('./models/', model_name)
    print(f"\nModel checkpoint: {model_state_file}")
    if not os.path.exists(model_state_file):
        print(f"ERROR: Model file not found: {model_state_file}")
        return None, None
    if args.add_static_graph:
        static_triples = np.array(_read_triplets_as_list(
            "./data/" + args.dataset + "/e-w-graph.txt", {}, {}, load_time=False
        ))
        num_static_rels = len(np.unique(static_triples[:, 1]))
        num_words = len(np.unique(static_triples[:, 2]))
        static_triples[:, 2] = static_triples[:, 2] + num_nodes
        static_node_id = torch.from_numpy(np.arange(num_words + data.num_nodes)).view(-1, 1).long()
    else:
        num_static_rels, num_words, static_triples, static_graph = 0, 0, [], None
    model = RecurrentRGCN(
        args.decoder, args.encoder, num_nodes, num_rels, num_static_rels, num_words, num_times,
        time_interval, args.n_hidden, args.opn, args.history_rate,
        sequence_len=args.train_history_len, num_bases=args.n_bases, num_basis=args.n_basis,
        num_hidden_layers=args.n_layers, dropout=args.dropout, self_loop=args.self_loop,
        skip_connect=args.skip_connect, layer_norm=args.layer_norm, input_dropout=args.input_dropout,
        hidden_dropout=args.hidden_dropout, feat_dropout=args.feat_dropout,
        aggregation=args.aggregation, weight=args.weight, discount=args.discount,
        angle=args.angle, use_static=args.add_static_graph, entity_prediction=args.entity_prediction,
        relation_prediction=args.relation_prediction, use_cuda=False, gpu=-1, analysis=args.run_analysis
    )
    if args.add_static_graph:
        static_graph = build_sub_graph(len(static_node_id), num_static_rels, static_triples, False, -1)
    print("\n" + "=" * 70)
    print("LOADING MODEL")
    print("=" * 70)
    ckpt = torch.load(model_state_file, map_location=torch.device('cpu'))
    print(f"Loaded from epoch: {ckpt.get('epoch', 'unknown')}")
    # Use strict=False to handle unexpected keys
    model.load_state_dict(ckpt['state_dict'], strict=False)
    model.to(torch.device('cpu'))
    print("[INFO] Model loaded with strict=False (ignoring extra keys)")
    print("\n[INFO] Using HARD mode calibration (DATA LEAKAGE WARNING)")
    calibrator = HardModeCalibrator(longtail_entities, num_nodes, num_rels)
    calibrator.learn_from_training(data.train, entity_freq)
    # STEP 1: Validation
    print("\n" + "=" * 70)
    print("STEP 1: EVALUATING VALIDATION SET")
    print("=" * 70)
    val_results = evaluate_model(
        model, train_list, valid_list, num_rels, num_nodes,
        all_ans_list_valid, static_graph, valid_times,
        history_val_time_nogt, longtail_entities,
        calibrator=None, apply_calibration=False,
        dataset_name="Validation", args=args
    )
    print_metrics_table("VALIDATION - Overall", val_results['mrr_raw'], val_results['hits_raw'],
                        val_results['mrr'], val_results['hits'], val_results['total_queries'])
    if val_results['mrr_lt'] > 0:
        print_metrics_table("VALIDATION - Long-tail", val_results['mrr_lt_raw'], val_results['hits_lt_raw'],
                            val_results['mrr_lt'], val_results['hits_lt'], val_results['lt_queries'])
    val_triples_cat = torch.cat(val_results['all_triples'], dim=0)
    val_scores_cat = torch.cat(val_results['all_scores'], dim=0)
    calibrator.learn_from_validation(val_triples_cat, val_scores_cat, val_triples_cat[:, 2], args.early_stop_ratio)
    calibrator_params = {
        'mode': 'hard', 'boost_scale': calibrator.boost_scale,
        'neighbor_w1': calibrator.neighbor_weight_1hop, 'neighbor_w2': calibrator.neighbor_weight_2hop,
        'adaptive_type': calibrator.adaptive_scale_type, 'adaptive_factor': calibrator.adaptive_scale_factor,
        'mult_both': calibrator.mult_both, 'mult_one': calibrator.mult_one,
    }
    # STEP 2: Test BEFORE
    print("\n" + "=" * 70)
    print("STEP 2: EVALUATING TEST SET (BEFORE POST-PROCESSING)")
    print("=" * 70)
    test_results_before = evaluate_model(
        model, train_list + valid_list, test_list, num_rels, num_nodes,
        all_ans_list_test, static_graph, test_times,
        history_test_time_nogt, longtail_entities,
        calibrator=None, apply_calibration=False,
        dataset_name="Test (Before)", args=args, collect_ranks_for_test=True
    )
    print_metrics_table("TEST (BEFORE) - Overall", test_results_before['mrr_raw'], test_results_before['hits_raw'],
                        test_results_before['mrr'], test_results_before['hits'], test_results_before['total_queries'])
    if test_results_before['mrr_lt'] > 0:
        print_metrics_table("TEST (BEFORE) - Long-tail", test_results_before['mrr_lt_raw'], test_results_before['hits_lt_raw'],
                            test_results_before['mrr_lt'], test_results_before['hits_lt'], test_results_before['lt_queries'])
    # STEP 3: Test AFTER
    print("\n" + "=" * 70)
    print("STEP 3: APPLYING POST-PROCESSING (HARD MODE)")
    print("=" * 70)
    test_results_after = evaluate_model(
        model, train_list + valid_list, test_list, num_rels, num_nodes,
        all_ans_list_test, static_graph, test_times,
        history_test_time_nogt, longtail_entities,
        calibrator=calibrator, apply_calibration=True,
        dataset_name="Test (HARD)", args=args,
        track_rank_changes=True, collect_ranks_for_test=True
    )
    print_metrics_table("TEST (AFTER) - Overall", test_results_after['mrr_raw'], test_results_after['hits_raw'],
                        test_results_after['mrr'], test_results_after['hits'], test_results_after['total_queries'])
    if test_results_after['mrr_lt'] > 0:
        print_metrics_table("TEST (AFTER) - Long-tail", test_results_after['mrr_lt_raw'], test_results_after['hits_lt_raw'],
                            test_results_after['mrr_lt'], test_results_after['hits_lt'], test_results_after['lt_queries'])
    print_comparison_table("TEST: BEFORE vs AFTER (HARD MODE)", test_results_before, test_results_after)
    if test_results_after['rank_changes']:
        print_rank_change_stats(test_results_after['rank_changes'], test_results_after['total_queries'])
    # Wilcoxon tests
    print("\n" + "=" * 70)
    print("STEP 4: STATISTICAL SIGNIFICANCE TESTING (WILCOXON)")
    print("=" * 70)
    wilcoxon_results = []
    # Overall Raw
    w_overall_raw = compute_wilcoxon_test(
        test_results_before['all_ranks_raw'], test_results_after['all_ranks_raw'],
        "Overall Entity Prediction (Raw Ranks)"
    )
    wilcoxon_results.append(w_overall_raw)
    print_wilcoxon_results(w_overall_raw)
    # Overall Filtered
    w_overall_filter = compute_wilcoxon_test(
        test_results_before['all_ranks_filter'], test_results_after['all_ranks_filter'],
        "Overall Entity Prediction (Filtered Ranks)"
    )
    wilcoxon_results.append(w_overall_filter)
    print_wilcoxon_results(w_overall_filter)
    # Long-tail Raw
    if len(test_results_before['all_ranks_raw_lt']) > 0:
        w_lt_raw = compute_wilcoxon_test(
            test_results_before['all_ranks_raw_lt'], test_results_after['all_ranks_raw_lt'],
            "Long-tail Entity Prediction (Raw Ranks)"
        )
        wilcoxon_results.append(w_lt_raw)
        print_wilcoxon_results(w_lt_raw)
        # Long-tail Filtered
        w_lt_filter = compute_wilcoxon_test(
            test_results_before['all_ranks_filter_lt'], test_results_after['all_ranks_filter_lt'],
            "Long-tail Entity Prediction (Filtered Ranks)"
        )
        wilcoxon_results.append(w_lt_filter)
        print_wilcoxon_results(w_lt_filter)
    save_results_to_file(args, test_results_before, test_results_after, calibrator_params, longtail_stats, wilcoxon_results)
    return test_results_before, test_results_after


# ============================================================================
# PART 14: ARGUMENT PARSER AND MAIN
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='TKG Evaluation with HARD Mode Post-Processing')
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID for checkpoint name")
    parser.add_argument("-d", "--dataset", type=str, default='ICEWS14',
                        choices=['ICEWS18', 'ICEWS14', 'ICEWS05-15', 'YAGO', 'WIKI'])
    parser.add_argument("--run-analysis", action='store_true', default=False)
    parser.add_argument("--multi-step", action='store_true', default=False)
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--add-static-graph", action='store_true', default=True)
    parser.add_argument("--longtail-threshold", type=int, default=20)
    parser.add_argument("--early-stop-ratio", type=float, default=0.5)
    parser.add_argument("--encoder", type=str, default="convgcn")
    parser.add_argument("--decoder", type=str, default="timeconvtranse")
    parser.add_argument("--aggregation", type=str, default="none")
    parser.add_argument("--opn", type=str, default="sub")
    parser.add_argument("--n-hidden", type=int, default=200)
    parser.add_argument("--n-bases", type=int, default=100)
    parser.add_argument("--n-basis", type=int, default=100)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--input-dropout", type=float, default=0.2)
    parser.add_argument("--hidden-dropout", type=float, default=0.2)
    parser.add_argument("--feat-dropout", type=float, default=0.2)
    parser.add_argument("--skip-connect", action='store_true', default=False)
    parser.add_argument("--self-loop", action='store_true', default=True)
    parser.add_argument("--layer-norm", action='store_true', default=True)
    parser.add_argument("--relation-prediction", action='store_true', default=True)
    parser.add_argument("--entity-prediction", action='store_true', default=True)
    parser.add_argument("--train-history-len", type=int, default=12)
    parser.add_argument("--test-history-len", type=int, default=12)
    parser.add_argument("--history-rate", type=float, default=0.3)
    parser.add_argument("--weight", type=float, default=0.5)
    parser.add_argument("--discount", type=float, default=1.0)
    parser.add_argument("--angle", type=int, default=14)
    return parser.parse_args()


def main():
    import warnings
    warnings.filterwarnings("ignore")
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    if hasattr(torch.backends, 'mps'):
        torch.backends.mps.is_available = lambda: False
    args = parse_args()
    if args.early_stop_ratio <= 0 or args.early_stop_ratio > 1:
        print("ERROR: early-stop-ratio must be in (0, 1]")
        sys.exit(1)
    print("\n" + "=" * 70)
    print("TKG EVALUATION WITH HARD MODE POST-PROCESSING")
    print("+ Wilcoxon Signed-Rank Test for Statistical Significance")
    print("=" * 70)
    print("\nConfiguration:")
    for arg, value in sorted(vars(args).items()):
        print(f"  {arg}: {value}")
    print("=" * 70 + "\n")
    run_experiment(args)


if __name__ == '__main__':
    main()