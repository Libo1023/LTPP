import argparse
import os
import pickle
import random
import math
import warnings
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from scipy import stats
from tqdm import tqdm

warnings.filterwarnings('ignore', category=FutureWarning)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def set_seed(seed=999):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_total_number(data_path, filename):
    with open(os.path.join(data_path, filename), 'r') as f:
        line = f.readline().strip().split()
        return int(line[0]), int(line[1])


def load_quadruples(data_path, *filenames):
    quadruples, times = [], set()
    for filename in filenames:
        filepath = os.path.join(data_path, filename)
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4:
                    h, r, t, ts = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
                    quadruples.append([h, r, t, ts])
                    times.add(ts)
    return np.array(quadruples), np.array(sorted(list(times)))


def comp_deg_norm(g):
    in_deg = g.in_degrees(range(g.number_of_nodes())).float()
    in_deg[torch.nonzero(in_deg == 0, as_tuple=False).view(-1)] = 1
    return 1.0 / in_deg


def get_big_graph(data, num_rels):
    if data is None or len(data) == 0:
        return None
    src, rel, dst = data[:, 0], data[:, 1], data[:, 2]
    uniq_v, edges = np.unique((src, dst), return_inverse=True)
    src_idx, dst_idx = np.reshape(edges, (2, -1))
    g = dgl.DGLGraph()
    g.add_nodes(len(uniq_v))
    src_all = np.concatenate([src_idx, dst_idx])
    dst_all = np.concatenate([dst_idx, src_idx])
    rel_s = np.concatenate([rel, rel + num_rels])
    rel_o = np.concatenate([rel + num_rels, rel])
    g.add_edges(src_all, dst_all)
    norm = comp_deg_norm(g)
    g.ndata.update({'id': torch.from_numpy(uniq_v).long().view(-1, 1), 'norm': norm.view(-1, 1)})
    g.edata['type_s'] = torch.LongTensor(rel_s)
    g.edata['type_o'] = torch.LongTensor(rel_o)
    g.ids = {int(uniq_v[i]): i for i in range(len(uniq_v))}
    return g


def make_subgraph(g, nodes):
    if g is None:
        return None
    nodes = list(nodes)
    relabeled_nodes = [g.ids[n] for n in nodes if n in g.ids]
    if len(relabeled_nodes) == 0:
        return None
    sub_g = g.subgraph(relabeled_nodes)
    sub_g.ndata.update({k: g.ndata[k][sub_g.ndata[dgl.NID]] for k in g.ndata if k != 'norm'})
    sub_g.edata.update({k: g.edata[k][sub_g.edata[dgl.EID]] for k in g.edata})
    norm = comp_deg_norm(sub_g)
    sub_g.ndata['norm'] = norm.view(-1, 1)
    node_id = sub_g.ndata['id'].view(-1).tolist()
    sub_g.ids = dict(zip(node_id, range(sub_g.number_of_nodes())))
    return sub_g


# =============================================================================
# RGCN LAYERS
# =============================================================================

class RGCNLayer(nn.Module):
    def __init__(self, in_feat, out_feat, bias=None, activation=None, self_loop=False, dropout=0.0):
        super().__init__()
        self.bias_flag = bias
        self.activation = activation
        self.self_loop = self_loop
        if self.bias_flag:
            self.bias = nn.Parameter(torch.Tensor(out_feat))
            nn.init.zeros_(self.bias)
        if self.self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight, gain=nn.init.calculate_gain('relu'))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def propagate(self, g, reverse):
        raise NotImplementedError

    def forward(self, g, reverse=False):
        if self.self_loop:
            loop_message = torch.mm(g.ndata['h'], self.loop_weight)
            if self.dropout:
                loop_message = self.dropout(loop_message)
        self.propagate(g, reverse)
        node_repr = g.ndata['h']
        if self.bias_flag:
            node_repr = node_repr + self.bias
        if self.self_loop:
            node_repr = node_repr + loop_message
        if self.activation:
            node_repr = self.activation(node_repr)
        g.ndata['h'] = node_repr
        return g


class RGCNBlockLayer(RGCNLayer):
    def __init__(self, in_feat, out_feat, num_rels, num_bases, bias=None,
                 activation=None, self_loop=False, dropout=0.0):
        super().__init__(in_feat, out_feat, bias, activation, self_loop, dropout)
        self.num_rels = num_rels
        self.num_bases = num_bases
        assert self.num_bases > 0
        self.out_feat = out_feat
        self.submat_in = in_feat // self.num_bases
        self.submat_out = out_feat // self.num_bases
        self.weight = nn.Parameter(torch.Tensor(self.num_rels, self.num_bases * self.submat_in * self.submat_out))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

    def msg_func(self, edges, reverse):
        if reverse:
            weight = self.weight.index_select(0, edges.data['type_o']).view(-1, self.submat_in, self.submat_out)
        else:
            weight = self.weight.index_select(0, edges.data['type_s']).view(-1, self.submat_in, self.submat_out)
        node = edges.src['h'].view(-1, 1, self.submat_in)
        msg = torch.bmm(node, weight).view(-1, self.out_feat)
        return {'msg': msg}

    def propagate(self, g, reverse):
        g.update_all(lambda x: self.msg_func(x, reverse), fn.sum(msg='msg', out='h'), self.apply_func)

    def apply_func(self, nodes):
        return {'h': nodes.data['h'] * nodes.data['norm']}


# =============================================================================
# AGGREGATORS
# =============================================================================

class RGCNAggregatorGlobal(nn.Module):
    def __init__(self, h_dim, dropout, num_nodes, num_rels, num_bases, seq_len=10, maxpool=1):
        super().__init__()
        self.h_dim = h_dim
        self.dropout = nn.Dropout(dropout)
        self.seq_len = seq_len
        self.num_rels = num_rels
        self.num_nodes = num_nodes
        self.maxpool = maxpool
        self.rgcn1 = RGCNBlockLayer(h_dim, h_dim, 2*num_rels, num_bases, activation=F.relu, self_loop=True, dropout=dropout)
        self.rgcn2 = RGCNBlockLayer(h_dim, h_dim, 2*num_rels, num_bases, activation=None, self_loop=True, dropout=dropout)

    def forward(self, t_list, ent_embeds, graph_dict, device, reverse=False):
        times = sorted(graph_dict.keys())
        if len(times) < 2:
            return None
        t_list_cpu = t_list.cpu()
        num_non_zero = len(torch.nonzero(t_list_cpu, as_tuple=False))
        if num_non_zero == 0:
            return None
        t_list_cpu = t_list_cpu[:num_non_zero]
        time_list, len_non_zero = [], []
        for tim in t_list_cpu:
            tim_val = tim.item()
            length = sum(1 for tt in times if tt < tim_val)
            if self.seq_len <= length:
                time_list.append(times[length - self.seq_len:length])
                len_non_zero.append(self.seq_len)
            elif length > 0:
                time_list.append(times[:length])
                len_non_zero.append(length)
            else:
                time_list.append([])
                len_non_zero.append(0)
        valid_idx = [i for i, l in enumerate(len_non_zero) if l > 0]
        if len(valid_idx) == 0:
            return None
        time_list = [time_list[i] for i in valid_idx]
        len_non_zero = [len_non_zero[i] for i in valid_idx]
        unique_t = sorted(set(t for tl in time_list for t in tl))
        g_list, time_to_idx = [], {}
        for t in unique_t:
            if t in graph_dict and graph_dict[t] is not None:
                time_to_idx[t] = len(g_list)
                g_list.append(graph_dict[t].to(device))
        if len(g_list) == 0:
            return None
        batched_graph = dgl.batch(g_list)
        batched_graph.ndata['h'] = ent_embeds[batched_graph.ndata['id'].view(-1)].view(-1, self.h_dim)
        self.rgcn1(batched_graph, reverse)
        self.rgcn2(batched_graph, reverse)
        global_info = dgl.max_nodes(batched_graph, 'h') if self.maxpool == 1 else dgl.mean_nodes(batched_graph, 'h')
        batched_graph.ndata.pop('h')
        embed_seq_tensor = torch.zeros(len(len_non_zero), self.seq_len, self.h_dim, device=device)
        for i, times_i in enumerate(time_list):
            for j, t in enumerate(times_i):
                if t in time_to_idx:
                    embed_seq_tensor[i, j, :] = global_info[time_to_idx[t]]
        embed_seq_tensor = self.dropout(embed_seq_tensor)
        packed_input = nn.utils.rnn.pack_padded_sequence(embed_seq_tensor, len_non_zero, batch_first=True, enforce_sorted=False)
        return packed_input

    def predict(self, t, ent_embeds, graph_dict, device, reverse=False):
        times = sorted(graph_dict.keys())
        if len(times) == 0:
            return torch.zeros(1, self.h_dim, device=device)
        idx = sum(1 for tt in times if tt < t)
        time_seq = times[idx - self.seq_len:idx] if self.seq_len <= idx else times[:idx]
        if len(time_seq) == 0:
            return torch.zeros(1, self.h_dim, device=device)
        g_list = [graph_dict[tim].to(device) for tim in time_seq if tim in graph_dict and graph_dict[tim] is not None]
        if len(g_list) == 0:
            return torch.zeros(1, self.h_dim, device=device)
        batched_graph = dgl.batch(g_list)
        batched_graph.ndata['h'] = ent_embeds[batched_graph.ndata['id'].view(-1)].view(-1, self.h_dim)
        self.rgcn1(batched_graph, reverse)
        self.rgcn2(batched_graph, reverse)
        global_info = dgl.max_nodes(batched_graph, 'h') if self.maxpool == 1 else dgl.mean_nodes(batched_graph, 'h')
        batched_graph.ndata.pop('h')
        return global_info


class RGCNAggregatorLocal(nn.Module):
    def __init__(self, h_dim, dropout, num_nodes, num_rels, num_bases, seq_len=10):
        super().__init__()
        self.h_dim = h_dim
        self.dropout = nn.Dropout(dropout)
        self.seq_len = seq_len
        self.num_rels = num_rels
        self.num_nodes = num_nodes
        self.rgcn1 = RGCNBlockLayer(h_dim, h_dim, 2*num_rels, num_bases, activation=F.relu, self_loop=True, dropout=dropout)
        self.rgcn2 = RGCNBlockLayer(h_dim, h_dim, 2*num_rels, num_bases, activation=None, self_loop=True, dropout=dropout)

    def forward(self, s_hist, s, r, ent_embeds, rel_embeds, graph_dict, global_emb, device, reverse=False):
        s_hist_data, s_hist_t = s_hist
        total_len = sum(len(h) for h in s_hist_data)
        if total_len == 0:
            return None, None
        hist_len = torch.LongTensor([len(h) for h in s_hist_data])
        s_len, s_idx = hist_len.sort(0, descending=True)
        num_non_zero = len(torch.nonzero(s_len, as_tuple=False))
        if num_non_zero == 0:
            return None, None
        s_len_non_zero = s_len[:num_non_zero].tolist()
        s_idx_list = s_idx.tolist()
        s_hist_sorted = [s_hist_data[idx] for idx in s_idx_list[:num_non_zero]]
        s_hist_t_sorted = [s_hist_t[idx] for idx in s_idx_list[:num_non_zero]]
        s_tem, r_tem = s[s_idx], r[s_idx]
        neighs_t = defaultdict(set)
        for i, (hist, hist_t) in enumerate(zip(s_hist_sorted, s_hist_t_sorted)):
            for neighs, t in zip(hist, hist_t):
                if len(neighs) > 0:
                    neighs_t[t].update(neighs[:, 1].tolist())
                    neighs_t[t].add(s_tem[i].item())
        if len(neighs_t) == 0:
            return None, None
        g_list, g_id_dict = [], {}
        for tim in sorted(neighs_t.keys()):
            if tim in graph_dict and graph_dict[tim] is not None:
                sub_g = make_subgraph(graph_dict[tim], neighs_t[tim])
                if sub_g is not None:
                    g_id_dict[tim] = len(g_list)
                    sub_g.start_id = 0 if len(g_list) == 0 else g_list[-1].start_id + g_list[-1].number_of_nodes()
                    g_list.append(sub_g)
        if len(g_list) == 0:
            return None, None
        node_ids_graph, global_emb_list, actual_lens = [], [], []
        for i, (hist, hist_t) in enumerate(zip(s_hist_sorted, s_hist_t_sorted)):
            count = 0
            for j, (neighs, t) in enumerate(zip(hist, hist_t)):
                if t in g_id_dict and len(neighs) > 0:
                    graph = g_list[g_id_dict[t]]
                    s_val = s_tem[i].item()
                    if s_val in graph.ids:
                        node_ids_graph.append(graph.ids[s_val] + graph.start_id)
                        global_emb_list.append(global_emb[t].view(1, -1) if t in global_emb else torch.zeros(1, self.h_dim, device=device))
                        count += 1
            actual_lens.append(min(count, self.seq_len) if count > 0 else 0)
        if len(node_ids_graph) == 0:
            return None, None
        g_list = [g.to(device) for g in g_list]
        batched_graph = dgl.batch(g_list)
        batched_graph.ndata['h'] = ent_embeds[batched_graph.ndata['id'].view(-1)].view(-1, self.h_dim)
        self.rgcn1(batched_graph, reverse)
        self.rgcn2(batched_graph, reverse)
        embeds_mean = batched_graph.ndata.pop('h')
        node_ids_tensor = torch.LongTensor(node_ids_graph).to(device)
        embeds_mean = embeds_mean[node_ids_tensor]
        global_emb_cat = torch.cat(global_emb_list, dim=0).to(device) if global_emb_list else torch.zeros(len(node_ids_graph), self.h_dim, device=device)
        s_embed_seq = torch.zeros(num_non_zero, self.seq_len, 4*self.h_dim, device=device)
        s_embed_seq_r = torch.zeros(num_non_zero, self.seq_len, 3*self.h_dim, device=device)
        embed_idx, final_lens = 0, []
        for i in range(num_non_zero):
            hist_len_i = actual_lens[i]
            if hist_len_i > 0 and embed_idx < len(embeds_mean):
                avail = min(hist_len_i, len(embeds_mean) - embed_idx)
                for j in range(avail):
                    idx = embed_idx + j
                    if idx < len(embeds_mean):
                        s_embed_seq[i, j, :] = torch.cat([embeds_mean[idx], ent_embeds[s_tem[i]], rel_embeds[r_tem[i]], global_emb_cat[idx] if idx < len(global_emb_cat) else torch.zeros(self.h_dim, device=device)])
                        s_embed_seq_r[i, j, :] = torch.cat([embeds_mean[idx], ent_embeds[s_tem[i]], global_emb_cat[idx] if idx < len(global_emb_cat) else torch.zeros(self.h_dim, device=device)])
                embed_idx += avail
                final_lens.append(max(1, avail))
            else:
                final_lens.append(1)
                s_embed_seq[i, 0, :self.h_dim] = ent_embeds[s_tem[i]]
                s_embed_seq_r[i, 0, :self.h_dim] = ent_embeds[s_tem[i]]
        s_embed_seq = self.dropout(s_embed_seq)
        s_embed_seq_r = self.dropout(s_embed_seq_r)
        s_packed = nn.utils.rnn.pack_padded_sequence(s_embed_seq, final_lens, batch_first=True, enforce_sorted=False)
        s_packed_r = nn.utils.rnn.pack_padded_sequence(s_embed_seq_r, final_lens, batch_first=True, enforce_sorted=False)
        return s_packed, s_packed_r


# =============================================================================
# RE-NET MODEL
# =============================================================================

class RENet(nn.Module):
    def __init__(self, num_nodes, h_dim, num_rels, dropout=0.5, seq_len=10, num_bases=100):
        super().__init__()
        self.num_nodes = num_nodes
        self.h_dim = h_dim
        self.num_rels = num_rels
        self.seq_len = seq_len
        self.rel_embeds = nn.Parameter(torch.Tensor(2*num_rels, h_dim))
        nn.init.xavier_uniform_(self.rel_embeds, gain=nn.init.calculate_gain('relu'))
        self.ent_embeds = nn.Parameter(torch.Tensor(num_nodes, h_dim))
        nn.init.xavier_uniform_(self.ent_embeds, gain=nn.init.calculate_gain('relu'))
        self.dropout = nn.Dropout(dropout)
        self.encoder = nn.GRU(4*h_dim, h_dim, batch_first=True)
        self.encoder_r = nn.GRU(3*h_dim, h_dim, batch_first=True)
        self.aggregator = RGCNAggregatorLocal(h_dim, dropout, num_nodes, num_rels, num_bases, seq_len)
        self.linear = nn.Linear(3*h_dim, num_nodes)
        self.linear_r = nn.Linear(2*h_dim, num_rels)
        self.criterion = nn.CrossEntropyLoss()
        self.global_emb = None
        self.graph_dict = None

    def predict_single(self, triplet, s_hist, o_hist, global_emb, graph_dict, device):
        s, r, o = triplet[0].item(), triplet[1].item(), triplet[2].item()
        s_h = torch.zeros(self.h_dim, device=device)
        o_h = torch.zeros(self.h_dim, device=device)
        ob_pred = self.linear(torch.cat([self.ent_embeds[s], s_h, self.rel_embeds[:self.num_rels][r]]))
        sub_pred = self.linear(torch.cat([self.ent_embeds[o], o_h, self.rel_embeds[self.num_rels:][r]]))
        return sub_pred, ob_pred


# =============================================================================
# LONG-TAIL UTILITIES
# =============================================================================

def count_entity_frequencies(train_data, num_rels):
    entity_freq = defaultdict(int)
    for quad in train_data:
        s, r, o = int(quad[0]), int(quad[1]), int(quad[2])
        entity_freq[s] += 2
        entity_freq[o] += 2
    return entity_freq


def identify_longtail_entities(entity_freq, threshold, num_entities):
    return {eid for eid in range(num_entities) if entity_freq.get(eid, 0) <= threshold}


def compute_longtail_statistics(data, longtail_entities, num_rels, split_name=""):
    total = len(data)
    both_lt = one_lt = neither_lt = subj_lt_only = obj_lt_only = 0
    for quad in data:
        s, o = int(quad[0]), int(quad[2])
        s_lt, o_lt = s in longtail_entities, o in longtail_entities
        if s_lt and o_lt:
            both_lt += 1
        elif s_lt or o_lt:
            one_lt += 1
            subj_lt_only += 1 if s_lt else 0
            obj_lt_only += 1 if o_lt else 0
        else:
            neither_lt += 1
    contains_lt = both_lt + one_lt
    return {
        'total': total, 'both_lt': both_lt, 'one_lt': one_lt, 'neither_lt': neither_lt,
        'subj_lt_only': subj_lt_only, 'obj_lt_only': obj_lt_only, 'contains_lt': contains_lt,
        'both_lt_pct': 100.0*both_lt/total if total > 0 else 0,
        'one_lt_pct': 100.0*one_lt/total if total > 0 else 0,
        'neither_lt_pct': 100.0*neither_lt/total if total > 0 else 0,
        'contains_lt_pct': 100.0*contains_lt/total if total > 0 else 0,
    }


# =============================================================================
# HARD MODE CALIBRATOR
# =============================================================================

class HardModeCalibrator:
    """Hard mode calibrator - uses ground truth object (DATA LEAKAGE)."""
    
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

    def _set_no_calibration(self):
        self.longtail_entity_boost = torch.zeros(self.num_entities)
        self.relation_lt_boost = torch.zeros(self.num_rels * 2)
        self.neighbor_weight_1hop = self.neighbor_weight_2hop = 0.0
        self.mult_both = self.mult_one = 1.0
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
        print(f"\nScore-adaptive scaling: Type={best_params['adaptive_type']}, Factor={best_params['adaptive_factor']}")
        print(f"Boost: scale={best_params['boost_scale']}, thresholds=({rt[0]},{rt[1]},{rt[2]})")
        print(f"Neighbor weights: 1-hop={best_params['neighbor_w1']}, 2-hop={best_params['neighbor_w2']}")
        print(f"Validation LT MRR: {baseline_mrr:.4f} -> {best_mrr:.4f}")
        print("=" * 70 + "\n")

    def _precompute_masks(self):
        self.lt_mask = torch.zeros(self.num_entities, dtype=torch.bool)
        for lt_ent in self.longtail_entities_set:
            if lt_ent < self.num_entities:
                self.lt_mask[lt_ent] = True

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
            track_changes=track_changes, targets=targets
        )
        if track_changes:
            self.rank_change_stats = rank_changes
        return calibrated

    def get_rank_change_stats(self):
        return self.rank_change_stats


# =============================================================================
# WILCOXON SIGNED-RANK TEST
# =============================================================================

def compute_wilcoxon_test(ranks_before, ranks_after, name=""):
    """Perform Wilcoxon signed-rank test comparing ranks before and after post-processing."""
    ranks_before_np = ranks_before.cpu().numpy().astype(float) if torch.is_tensor(ranks_before) else np.array(ranks_before).astype(float)
    ranks_after_np = ranks_after.cpu().numpy().astype(float) if torch.is_tensor(ranks_after) else np.array(ranks_after).astype(float)
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
            'name': name, 'n_samples': len(ranks_before_np),
            'n_improved': int(n_improved), 'n_worsened': int(n_worsened), 'n_unchanged': int(n_unchanged),
            'median_before': median_before, 'median_after': median_after, 'mean_change': mean_change,
            'statistic': None, 'p_value': None, 'effect_size': None, 'significant': None,
            'interpretation': "Insufficient non-zero differences for test (n < 10)"
        }
    
    try:
        statistic, p_value = stats.wilcoxon(ranks_before_np, ranks_after_np, alternative='two-sided')
        n = len(non_zero_diff)
        z_score = stats.norm.ppf(1 - p_value / 2)
        effect_size = abs(z_score) / np.sqrt(n)
        effect_interpretation = "small" if effect_size < 0.3 else ("medium" if effect_size < 0.5 else "large")
        significant = p_value < 0.05
        if significant:
            interpretation = f"Significant {'improvement' if mean_change > 0 else 'degradation'} (p={p_value:.4e}, {effect_interpretation} effect)"
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
    """Print Wilcoxon test results in formatted table."""
    print("\n" + "=" * 80)
    print(f"WILCOXON SIGNED-RANK TEST: {results['name']}")
    print("=" * 80)
    print(f"\n{'Sample Statistics':<40}")
    print("-" * 80)
    n = results['n_samples']
    print(f"  Total samples:                    {n:>10}")
    print(f"  Improved (rank decreased):        {results['n_improved']:>10} ({100*results['n_improved']/n:.2f}%)")
    print(f"  Worsened (rank increased):        {results['n_worsened']:>10} ({100*results['n_worsened']/n:.2f}%)")
    print(f"  Unchanged:                        {results['n_unchanged']:>10} ({100*results['n_unchanged']/n:.2f}%)")
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


# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

def evaluate_filter(pred, target, all_triplets, s, r, is_subject=True):
    pred = pred.clone()
    ground = pred[target].item()
    if is_subject:
        mask = (all_triplets[:, 0] == s) & (all_triplets[:, 1] == r)
        filter_ents = all_triplets[mask, 2]
    else:
        mask = (all_triplets[:, 2] == target) & (all_triplets[:, 1] == r)
        filter_ents = all_triplets[mask, 0]
    pred[filter_ents] = -1e9
    pred[target] = ground
    return (pred > ground).sum().item() + 1


def calc_mrr_hits(ranks):
    ranks = np.array(ranks)
    mrr = np.mean(1.0 / ranks)
    return mrr, {k: np.mean(ranks <= k) for k in [1, 3, 10]}


def evaluate_renet(model, test_data, test_s_hist, test_o_hist, global_emb, graph_dict,
                   total_data_tensor, longtail_entities, num_rels, device,
                   calibrator=None, apply_calibration=False, track_changes=False,
                   collect_individual_ranks=False):
    """Evaluate RE-NET model with optional calibration and rank collection."""
    model.eval()
    
    all_triples, all_ob_logits, all_sub_logits = [], [], []
    all_ob_targets, all_sub_targets = [], []
    
    with torch.no_grad():
        for i in tqdm(range(len(test_data)), desc="Collecting predictions"):
            triplet = torch.from_numpy(test_data[i]).long().to(device)
            s_hist = (test_s_hist[0][i], test_s_hist[1][i])
            o_hist = (test_o_hist[0][i], test_o_hist[1][i])
            sub_pred, ob_pred = model.predict_single(triplet, s_hist, o_hist, global_emb, graph_dict, device)
            s, r, o = triplet[0].item(), triplet[1].item(), triplet[2].item()
            all_triples.append([s, r, o])
            all_ob_logits.append(ob_pred.unsqueeze(0))
            all_sub_logits.append(sub_pred.unsqueeze(0))
            all_ob_targets.append(o)
            all_sub_targets.append(s)
    
    all_triples = torch.tensor(all_triples)
    all_ob_logits = torch.cat(all_ob_logits, dim=0)
    all_sub_logits = torch.cat(all_sub_logits, dim=0)
    all_ob_targets = torch.tensor(all_ob_targets)
    all_sub_targets = torch.tensor(all_sub_targets)
    
    if apply_calibration and calibrator is not None:
        ob_triples_for_calib = all_triples.clone()
        all_ob_logits = calibrator.calibrate_scores(ob_triples_for_calib, all_ob_logits, num_rels,
                                                     targets=all_ob_targets, track_changes=track_changes)
        sub_triples_for_calib = torch.stack([all_triples[:, 2], all_triples[:, 1], all_triples[:, 0]], dim=1)
        all_sub_logits = calibrator.calibrate_scores(sub_triples_for_calib, all_sub_logits, num_rels,
                                                      targets=all_sub_targets, track_changes=False)
    
    ranks_raw, ranks_filter = [], []
    ranks_raw_lt, ranks_filter_lt = [], []
    individual_ranks_raw, individual_ranks_filter = [], []
    individual_ranks_raw_lt, individual_ranks_filter_lt = [], []
    
    for i in range(len(test_data)):
        s, r, o = all_triples[i, 0].item(), all_triples[i, 1].item(), all_triples[i, 2].item()
        ob_pred, sub_pred = all_ob_logits[i], all_sub_logits[i]
        
        ground_ob, ground_sub = ob_pred[o].item(), sub_pred[s].item()
        rank_o_raw = (ob_pred > ground_ob).sum().item() + 1
        rank_s_raw = (sub_pred > ground_sub).sum().item() + 1
        rank_o_filter = evaluate_filter(ob_pred, o, total_data_tensor, s, r, is_subject=True)
        rank_s_filter = evaluate_filter(sub_pred, s, total_data_tensor, o, r, is_subject=False)
        
        ranks_raw.extend([rank_s_raw, rank_o_raw])
        ranks_filter.extend([rank_s_filter, rank_o_filter])
        
        if collect_individual_ranks:
            individual_ranks_raw.extend([rank_s_raw, rank_o_raw])
            individual_ranks_filter.extend([rank_s_filter, rank_o_filter])
        
        is_lt = (s in longtail_entities) or (o in longtail_entities)
        if is_lt:
            ranks_raw_lt.extend([rank_s_raw, rank_o_raw])
            ranks_filter_lt.extend([rank_s_filter, rank_o_filter])
            if collect_individual_ranks:
                individual_ranks_raw_lt.extend([rank_s_raw, rank_o_raw])
                individual_ranks_filter_lt.extend([rank_s_filter, rank_o_filter])
    
    mrr_raw, hits_raw = calc_mrr_hits(ranks_raw)
    mrr_filter, hits_filter = calc_mrr_hits(ranks_filter)
    
    if ranks_raw_lt:
        mrr_raw_lt, hits_raw_lt = calc_mrr_hits(ranks_raw_lt)
        mrr_filter_lt, hits_filter_lt = calc_mrr_hits(ranks_filter_lt)
    else:
        mrr_raw_lt = mrr_filter_lt = 0.0
        hits_raw_lt = hits_filter_lt = {1: 0.0, 3: 0.0, 10: 0.0}
    
    results = {
        'mrr': mrr_filter, 'mrr_raw': mrr_raw,
        'hits': [hits_filter[1], hits_filter[3], hits_filter[10]],
        'hits_raw': [hits_raw[1], hits_raw[3], hits_raw[10]],
        'mrr_lt': mrr_filter_lt, 'mrr_lt_raw': mrr_raw_lt,
        'hits_lt': [hits_filter_lt[1], hits_filter_lt[3], hits_filter_lt[10]],
        'hits_lt_raw': [hits_raw_lt[1], hits_raw_lt[3], hits_raw_lt[10]],
        'total_queries': len(test_data) * 2,
        'lt_queries': len(ranks_raw_lt),
        'all_triples': all_triples,
        'all_ob_logits': all_ob_logits,
        'all_sub_logits': all_sub_logits,
        'all_ob_targets': all_ob_targets,
        'all_sub_targets': all_sub_targets,
        'rank_changes': calibrator.get_rank_change_stats() if (apply_calibration and calibrator) else None
    }
    
    if collect_individual_ranks:
        results['all_ranks_raw'] = torch.tensor(individual_ranks_raw)
        results['all_ranks_filter'] = torch.tensor(individual_ranks_filter)
        results['all_ranks_raw_lt'] = torch.tensor(individual_ranks_raw_lt) if individual_ranks_raw_lt else torch.tensor([])
        results['all_ranks_filter_lt'] = torch.tensor(individual_ranks_filter_lt) if individual_ranks_filter_lt else torch.tensor([])
    
    return results


# =============================================================================
# PRINTING AND SAVING FUNCTIONS
# =============================================================================

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
    for i, m in enumerate(['MRR', 'Hits@1', 'Hits@3', 'Hits@10']):
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
        for i, m in enumerate(['MRR', 'Hits@1', 'Hits@3', 'Hits@10']):
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
    boosted, lowered, unchanged, skipped = rank_stats['boosted'], rank_stats['lowered'], rank_stats['unchanged'], rank_stats['skipped']
    modified = boosted + lowered + unchanged
    print(f"\nTotal test queries: {total_queries}")
    print(f"Skipped (not calibrated): {skipped:>8} ({100.0*skipped/total_queries:>6.2f}%)")
    print(f"Modified (calibrated):    {modified:>8} ({100.0*modified/total_queries:>6.2f}%)")
    if modified > 0:
        print(f"\nEffect on modified queries:")
        print(f"  Boosted UP:   {boosted:>8} ({100.0*boosted/modified:>6.2f}%)")
        print(f"  Lowered DOWN: {lowered:>8} ({100.0*lowered/modified:>6.2f}%)")
        print(f"  Unchanged:    {unchanged:>8} ({100.0*unchanged/modified:>6.2f}%)")
    print("=" * 70 + "\n")


def save_results_to_file(args, test_results_before, test_results_after, calibrator_params, longtail_stats, wilcoxon_results):
    filename = f"2_results_renet_{args.dataset}_h{args.n_hidden}_seq{args.seq_len}_hard_gpu{args.gpu}.txt"
    filepath = os.path.join("./", filename)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(filepath, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("RE-NET EVALUATION WITH HARD MODE POST-PROCESSING - TEST RESULTS\n")
        f.write("+ Wilcoxon Signed-Rank Test for Statistical Significance\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Timestamp: {timestamp}\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("CONFIGURATION\n")
        f.write("-" * 80 + "\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Hidden dimension: {args.n_hidden}\n")
        f.write(f"Sequence length: {args.seq_len}\n")
        f.write(f"GPU (for checkpoint): {args.gpu}\n")
        f.write(f"Long-tail threshold: {args.longtail_threshold}\n")
        f.write(f"Early stop ratio: {args.early_stop_ratio}\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("LONG-TAIL STATISTICS\n")
        f.write("-" * 80 + "\n")
        if longtail_stats:
            f.write(f"Long-tail entities: {longtail_stats.get('num_lt_entities', 'N/A')}\n")
            f.write(f"Total entities: {longtail_stats.get('num_entities', 'N/A')}\n")
            f.write(f"Long-tail percentage: {longtail_stats.get('lt_percentage', 'N/A'):.2f}%\n\n")
        
        # Before results
        f.write("-" * 80 + "\n")
        f.write("TEST RESULTS - BEFORE POST-PROCESSING\n")
        f.write("-" * 80 + "\n\n")
        f.write(">>> Overall Entity Prediction\n")
        f.write(f"{'Metric':<15} {'Raw':>15} {'Filtered':>15}\n")
        f.write("-" * 45 + "\n")
        f.write(f"{'MRR':<15} {test_results_before['mrr_raw']:>15.6f} {test_results_before['mrr']:>15.6f}\n")
        for i, h in enumerate(['Hits@1', 'Hits@3', 'Hits@10']):
            f.write(f"{h:<15} {test_results_before['hits_raw'][i]:>15.6f} {test_results_before['hits'][i]:>15.6f}\n")
        f.write(f"Total queries: {test_results_before['total_queries']}\n\n")
        f.write(">>> Long-tail Entity Prediction\n")
        f.write(f"{'Metric':<15} {'Raw':>15} {'Filtered':>15}\n")
        f.write("-" * 45 + "\n")
        f.write(f"{'MRR':<15} {test_results_before['mrr_lt_raw']:>15.6f} {test_results_before['mrr_lt']:>15.6f}\n")
        for i, h in enumerate(['Hits@1', 'Hits@3', 'Hits@10']):
            f.write(f"{h:<15} {test_results_before['hits_lt_raw'][i]:>15.6f} {test_results_before['hits_lt'][i]:>15.6f}\n")
        f.write(f"Long-tail queries: {test_results_before['lt_queries']}\n\n")
        
        # After results
        f.write("-" * 80 + "\n")
        f.write("TEST RESULTS - AFTER POST-PROCESSING (HARD MODE)\n")
        f.write("-" * 80 + "\n\n")
        f.write(">>> Overall Entity Prediction\n")
        f.write(f"{'Metric':<15} {'Raw':>15} {'Filtered':>15}\n")
        f.write("-" * 45 + "\n")
        f.write(f"{'MRR':<15} {test_results_after['mrr_raw']:>15.6f} {test_results_after['mrr']:>15.6f}\n")
        for i, h in enumerate(['Hits@1', 'Hits@3', 'Hits@10']):
            f.write(f"{h:<15} {test_results_after['hits_raw'][i]:>15.6f} {test_results_after['hits'][i]:>15.6f}\n")
        f.write(f"Total queries: {test_results_after['total_queries']}\n\n")
        f.write(">>> Long-tail Entity Prediction\n")
        f.write(f"{'Metric':<15} {'Raw':>15} {'Filtered':>15}\n")
        f.write("-" * 45 + "\n")
        f.write(f"{'MRR':<15} {test_results_after['mrr_lt_raw']:>15.6f} {test_results_after['mrr_lt']:>15.6f}\n")
        for i, h in enumerate(['Hits@1', 'Hits@3', 'Hits@10']):
            f.write(f"{h:<15} {test_results_after['hits_lt_raw'][i]:>15.6f} {test_results_after['hits_lt'][i]:>15.6f}\n")
        f.write(f"Long-tail queries: {test_results_after['lt_queries']}\n\n")
        
        # Improvement summary
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
        
        # Rank change stats
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
                f.write(f"  Boosted up: {boosted} ({100.0*boosted/modified:.2f}%)\n")
                f.write(f"  Lowered down: {lowered} ({100.0*lowered/modified:.2f}%)\n")
                f.write(f"  Unchanged: {unchanged} ({100.0*unchanged/modified:.2f}%)\n\n")
        
        # Wilcoxon results
        f.write("-" * 80 + "\n")
        f.write("WILCOXON SIGNED-RANK TEST RESULTS\n")
        f.write("-" * 80 + "\n\n")
        for w_result in wilcoxon_results:
            f.write(f">>> {w_result['name']}\n")
            n = w_result['n_samples']
            f.write(f"  Total samples: {n}\n")
            f.write(f"  Improved (rank decreased): {w_result['n_improved']} ({100*w_result['n_improved']/n:.2f}%)\n")
            f.write(f"  Worsened (rank increased): {w_result['n_worsened']} ({100*w_result['n_worsened']/n:.2f}%)\n")
            f.write(f"  Unchanged: {w_result['n_unchanged']} ({100*w_result['n_unchanged']/n:.2f}%)\n")
            f.write(f"  Median rank (before): {w_result['median_before']:.2f}\n")
            f.write(f"  Median rank (after): {w_result['median_after']:.2f}\n")
            f.write(f"  Mean rank change: {w_result['mean_change']:.4f}\n")
            if w_result['statistic'] is not None:
                f.write(f"  Test statistic (W): {w_result['statistic']:.2f}\n")
                f.write(f"  P-value: {w_result['p_value']:.4e}\n")
                f.write(f"  Effect size (r): {w_result['effect_size']:.4f}\n")
                f.write(f"  Significant (p < 0.05): {'Yes' if w_result['significant'] else 'No'}\n")
            f.write(f"  Interpretation: {w_result['interpretation']}\n\n")
        
        # Calibrator params
        if calibrator_params:
            f.write("-" * 80 + "\n")
            f.write("CALIBRATOR PARAMETERS (HARD MODE)\n")
            f.write("-" * 80 + "\n\n")
            for key, value in calibrator_params.items():
                f.write(f"{key}: {value}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF RESULTS\n")
        f.write("=" * 80 + "\n")
    
    print(f"\n[INFO] Results saved to: {filepath}")
    return filepath


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='RE-NET Evaluation with Hard Mode Post-Processing + Wilcoxon Test')
    parser.add_argument('-d', '--dataset', type=str, default='ICEWS14',
                        choices=['ICEWS14', 'ICEWS18', 'ICEWS05-15'])
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--model-dir', type=str, default='./models')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID for checkpoint name (eval on CPU)')
    parser.add_argument('--seed', type=int, default=999)
    parser.add_argument('--num-bases', type=int, default=100)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--longtail-threshold', type=int, default=20)
    parser.add_argument('--early-stop-ratio', type=float, default=0.5, 
                        help='Early stopping ratio for grid search (0-1, default 0.4)')
    
    args = parser.parse_args()
    
    # Validate early stop ratio
    if args.early_stop_ratio <= 0 or args.early_stop_ratio > 1:
        print("ERROR: early-stop-ratio must be in (0, 1]")
        return
    
    # Auto-adjust n_hidden and seq_len based on dataset
    if args.dataset == 'ICEWS05-15':
        args.n_hidden = 100
        args.seq_len = 7
    else:
        args.n_hidden = 200
        args.seq_len = 12
    
    set_seed(args.seed)
    device = torch.device('cpu')
    print(f"\n[INFO] Running evaluation on CPU")
    print(f"[INFO] Loading checkpoint for GPU {args.gpu}")
    
    # Load data
    data_path = os.path.join(args.data_dir, args.dataset)
    num_nodes, num_rels = get_total_number(data_path, 'stat.txt')
    train_data, _ = load_quadruples(data_path, 'train.txt')
    valid_data, _ = load_quadruples(data_path, 'valid.txt')
    test_data, _ = load_quadruples(data_path, 'test.txt')
    total_data, _ = load_quadruples(data_path, 'train.txt', 'valid.txt', 'test.txt')
    
    print("\n" + "=" * 70)
    print("DATASET STATISTICS")
    print("=" * 70)
    print(f"Dataset: {args.dataset}")
    print(f"Entities: {num_nodes}, Relations: {num_rels}")
    print(f"Train: {len(train_data)}, Valid: {len(valid_data)}, Test: {len(test_data)}")
    print(f"Hidden dim: {args.n_hidden}, Seq len: {args.seq_len}")
    
    # Long-tail analysis
    entity_freq = count_entity_frequencies(train_data, num_rels)
    longtail_entities = identify_longtail_entities(entity_freq, args.longtail_threshold, num_nodes)
    longtail_stats = {
        'num_lt_entities': len(longtail_entities),
        'num_entities': num_nodes,
        'lt_percentage': 100.0 * len(longtail_entities) / num_nodes
    }
    
    print("\n" + "=" * 70)
    print("LONG-TAIL ENTITY ANALYSIS")
    print("=" * 70)
    print(f"Threshold: {args.longtail_threshold}")
    print(f"Long-tail entities: {len(longtail_entities)} ({longtail_stats['lt_percentage']:.2f}%)")
    
    # Load history data
    history_path = os.path.join(data_path, 'history')
    with open(os.path.join(history_path, 'train_graphs.pkl'), 'rb') as f:
        graph_dict = pickle.load(f)
    with open(os.path.join(history_path, 'dev_history_sub.pkl'), 'rb') as f:
        valid_s_hist = pickle.load(f)
    with open(os.path.join(history_path, 'dev_history_ob.pkl'), 'rb') as f:
        valid_o_hist = pickle.load(f)
    with open(os.path.join(history_path, 'test_history_sub.pkl'), 'rb') as f:
        test_s_hist = pickle.load(f)
    with open(os.path.join(history_path, 'test_history_ob.pkl'), 'rb') as f:
        test_o_hist = pickle.load(f)
    
    # Load model
    model_name = f"renet-{args.dataset}-h{args.n_hidden}-seq{args.seq_len}-dp{args.dropout}-lr0.001-gpu{args.gpu}.pth"
    model_path = os.path.join(args.model_dir, model_name)
    
    if not os.path.exists(model_path):
        print(f"\n[ERROR] Model checkpoint not found: {model_path}")
        return
    
    model = RENet(num_nodes, args.n_hidden, num_rels, dropout=args.dropout, 
                  seq_len=args.seq_len, num_bases=args.num_bases)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    global_emb = checkpoint.get('global_emb', {})
    print(f"\n[INFO] Loaded model from {model_path}")
    
    model = model.to(device)
    model.eval()
    total_data_tensor = torch.from_numpy(total_data).to(device)
    
    # Create calibrator
    calibrator = HardModeCalibrator(longtail_entities, num_nodes, num_rels)
    calibrator.learn_from_training(train_data, entity_freq)
    
    # STEP 1: Validation
    print("\n" + "=" * 70)
    print("STEP 1: EVALUATING VALIDATION SET (for calibration learning)")
    print("=" * 70)
    
    val_results = evaluate_renet(model, valid_data, valid_s_hist, valid_o_hist, global_emb, graph_dict,
                                  total_data_tensor, longtail_entities, num_rels, device)
    
    print_metrics_table("VALIDATION - Overall", val_results['mrr_raw'], val_results['hits_raw'],
                        val_results['mrr'], val_results['hits'], val_results['total_queries'])
    if val_results['mrr_lt'] > 0:
        print_metrics_table("VALIDATION - Long-tail", val_results['mrr_lt_raw'], val_results['hits_lt_raw'],
                            val_results['mrr_lt'], val_results['hits_lt'], val_results['lt_queries'])
    
    # Prepare validation data for calibration
    val_triples_ob = val_results['all_triples']
    val_triples_sub = torch.stack([val_results['all_triples'][:, 2], val_results['all_triples'][:, 1], 
                                   val_results['all_triples'][:, 0]], dim=1)
    val_triples_cat = torch.cat([val_triples_ob, val_triples_sub], dim=0)
    val_scores_cat = torch.cat([val_results['all_ob_logits'], val_results['all_sub_logits']], dim=0)
    val_targets_cat = torch.cat([val_results['all_ob_targets'], val_results['all_sub_targets']], dim=0)
    
    calibrator.learn_from_validation(val_triples_cat, val_scores_cat, val_targets_cat, args.early_stop_ratio)
    
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
    
    test_results_before = evaluate_renet(model, test_data, test_s_hist, test_o_hist, global_emb, graph_dict,
                                          total_data_tensor, longtail_entities, num_rels, device,
                                          collect_individual_ranks=True)
    
    print_metrics_table("TEST (BEFORE) - Overall", test_results_before['mrr_raw'], test_results_before['hits_raw'],
                        test_results_before['mrr'], test_results_before['hits'], test_results_before['total_queries'])
    if test_results_before['mrr_lt'] > 0:
        print_metrics_table("TEST (BEFORE) - Long-tail", test_results_before['mrr_lt_raw'], test_results_before['hits_lt_raw'],
                            test_results_before['mrr_lt'], test_results_before['hits_lt'], test_results_before['lt_queries'])
    
    # STEP 3: Test AFTER
    print("\n" + "=" * 70)
    print("STEP 3: APPLYING POST-PROCESSING (HARD MODE)")
    print("=" * 70)
    
    test_results_after = evaluate_renet(model, test_data, test_s_hist, test_o_hist, global_emb, graph_dict,
                                         total_data_tensor, longtail_entities, num_rels, device,
                                         calibrator=calibrator, apply_calibration=True, track_changes=True,
                                         collect_individual_ranks=True)
    
    print_metrics_table("TEST (AFTER - HARD) - Overall", test_results_after['mrr_raw'], test_results_after['hits_raw'],
                        test_results_after['mrr'], test_results_after['hits'], test_results_after['total_queries'])
    if test_results_after['mrr_lt'] > 0:
        print_metrics_table("TEST (AFTER - HARD) - Long-tail", test_results_after['mrr_lt_raw'], test_results_after['hits_lt_raw'],
                            test_results_after['mrr_lt'], test_results_after['hits_lt'], test_results_after['lt_queries'])
    
    print_comparison_table("TEST: BEFORE vs AFTER (HARD MODE)", test_results_before, test_results_after)
    
    if test_results_after['rank_changes']:
        print_rank_change_stats(test_results_after['rank_changes'], test_results_after['total_queries'])
    
    # STEP 4: Wilcoxon tests
    print("\n" + "=" * 70)
    print("STEP 4: STATISTICAL SIGNIFICANCE TESTING (WILCOXON)")
    print("=" * 70)
    
    wilcoxon_results = []
    
    # Overall Raw
    w_overall_raw = compute_wilcoxon_test(test_results_before['all_ranks_raw'], test_results_after['all_ranks_raw'],
                                          "Overall Entity Prediction (Raw Ranks)")
    wilcoxon_results.append(w_overall_raw)
    print_wilcoxon_results(w_overall_raw)
    
    # Overall Filtered
    w_overall_filter = compute_wilcoxon_test(test_results_before['all_ranks_filter'], test_results_after['all_ranks_filter'],
                                              "Overall Entity Prediction (Filtered Ranks)")
    wilcoxon_results.append(w_overall_filter)
    print_wilcoxon_results(w_overall_filter)
    
    # Long-tail Raw
    if len(test_results_before['all_ranks_raw_lt']) > 0:
        w_lt_raw = compute_wilcoxon_test(test_results_before['all_ranks_raw_lt'], test_results_after['all_ranks_raw_lt'],
                                          "Long-tail Entity Prediction (Raw Ranks)")
        wilcoxon_results.append(w_lt_raw)
        print_wilcoxon_results(w_lt_raw)
        
        # Long-tail Filtered
        w_lt_filter = compute_wilcoxon_test(test_results_before['all_ranks_filter_lt'], test_results_after['all_ranks_filter_lt'],
                                             "Long-tail Entity Prediction (Filtered Ranks)")
        wilcoxon_results.append(w_lt_filter)
        print_wilcoxon_results(w_lt_filter)
    
    # Save results
    save_results_to_file(args, test_results_before, test_results_after, calibrator_params, longtail_stats, wilcoxon_results)
    
    print("\n[INFO] Evaluation complete!")


if __name__ == '__main__':
    main()