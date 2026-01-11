import argparse
import os
import sys
import time
import pickle
import random
import math
import warnings
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from tqdm import tqdm

# Suppress specific warnings
warnings.filterwarnings('ignore', category=FutureWarning)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def set_seed(seed=999):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_total_number(data_path, filename):
    """Read number of entities and relations from stat.txt"""
    with open(os.path.join(data_path, filename), 'r') as f:
        line = f.readline().strip().split()
        return int(line[0]), int(line[1])


def load_quadruples(data_path, *filenames):
    """Load quadruples from one or more files"""
    quadruples = []
    times = set()
    for filename in filenames:
        filepath = os.path.join(data_path, filename)
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4:
                    h, r, t, ts = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
                    quadruples.append([h, r, t, ts])
                    times.add(ts)
    times = sorted(list(times))
    return np.array(quadruples), np.array(times)


def comp_deg_norm(g):
    """Compute degree normalization for graph"""
    in_deg = g.in_degrees(range(g.number_of_nodes())).float()
    in_deg[torch.nonzero(in_deg == 0, as_tuple=False).view(-1)] = 1
    return 1.0 / in_deg


def move_dgl_to_cuda(g, device):
    """Move DGL graph to specified device, preserving custom attributes"""
    if g is None:
        return None
    
    # Store custom attributes before moving
    custom_attrs = {}
    for attr in ['ids', 'start_id']:
        if hasattr(g, attr):
            custom_attrs[attr] = getattr(g, attr)
    
    # Move the entire graph to device
    g_new = g.to(device)
    
    # Restore custom attributes
    for attr, val in custom_attrs.items():
        setattr(g_new, attr, val)
    
    return g_new


def get_big_graph(data, num_rels):
    """Build a big graph from triplet data for a single timestamp"""
    if data is None or len(data) == 0:
        return None
    src, rel, dst = data[:, 0], data[:, 1], data[:, 2]
    uniq_v, edges = np.unique((src, dst), return_inverse=True)
    src_idx, dst_idx = np.reshape(edges, (2, -1))
    
    g = dgl.DGLGraph()
    g.add_nodes(len(uniq_v))
    
    # Add both directions
    src_all = np.concatenate([src_idx, dst_idx])
    dst_all = np.concatenate([dst_idx, src_idx])
    rel_s = np.concatenate([rel, rel + num_rels])
    rel_o = np.concatenate([rel + num_rels, rel])
    
    g.add_edges(src_all, dst_all)
    norm = comp_deg_norm(g)
    g.ndata.update({'id': torch.from_numpy(uniq_v).long().view(-1, 1), 
                    'norm': norm.view(-1, 1)})
    g.edata['type_s'] = torch.LongTensor(rel_s)
    g.edata['type_o'] = torch.LongTensor(rel_o)
    
    # Build id mapping
    g.ids = {int(uniq_v[i]): i for i in range(len(uniq_v))}
    return g


def make_subgraph(g, nodes):
    """Create a subgraph containing only specified nodes"""
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


def soft_cross_entropy(pred, soft_targets):
    """Soft cross entropy loss for distribution matching"""
    logsoftmax = nn.LogSoftmax(dim=1)
    pred = pred.double()
    soft_targets = soft_targets.double()
    return torch.mean(torch.sum(-soft_targets * logsoftmax(pred), dim=1))


def get_true_distribution(train_data, num_nodes):
    """Get true entity distribution for each timestamp"""
    true_prob_s, true_prob_o = [], []
    true_s = np.zeros(num_nodes)
    true_o = np.zeros(num_nodes)
    current_t = None
    
    for quad in train_data:
        s, r, o, t = quad
        if current_t is not None and current_t != t:
            s_sum, o_sum = np.sum(true_s), np.sum(true_o)
            true_prob_s.append(true_s / s_sum if s_sum > 0 else true_s)
            true_prob_o.append(true_o / o_sum if o_sum > 0 else true_o)
            true_s = np.zeros(num_nodes)
            true_o = np.zeros(num_nodes)
        true_s[s] += 1
        true_o[o] += 1
        current_t = t
    
    # Last timestamp
    s_sum, o_sum = np.sum(true_s), np.sum(true_o)
    true_prob_s.append(true_s / s_sum if s_sum > 0 else true_s)
    true_prob_o.append(true_o / o_sum if o_sum > 0 else true_o)
    
    return np.array(true_prob_s), np.array(true_prob_o)


# =============================================================================
# DATA PREPROCESSING
# =============================================================================

def get_history_graph(data_path, dataset):
    """Generate history graphs and entity histories for RE-NET"""
    print(f"Preprocessing {dataset}...")
    
    num_nodes, num_rels = get_total_number(data_path, 'stat.txt')
    train_data, train_times = load_quadruples(data_path, 'train.txt')
    valid_data, valid_times = load_quadruples(data_path, 'valid.txt')
    test_data, test_times = load_quadruples(data_path, 'test.txt')
    
    history_dir = os.path.join(data_path, 'history')
    os.makedirs(history_dir, exist_ok=True)
    
    # Build graph dictionary for each timestamp
    print("Building graph dictionary...")
    graph_dict = {}
    time_to_data = defaultdict(list)
    
    for quad in train_data:
        time_to_data[quad[3]].append(quad[:3])
    
    for t in tqdm(sorted(time_to_data.keys())):
        triplets = np.array(time_to_data[t])
        graph_dict[t] = get_big_graph(triplets, num_rels)
    
    with open(os.path.join(history_dir, 'train_graphs.pkl'), 'wb') as f:
        pickle.dump(graph_dict, f)
    print(f"Saved train_graphs.pkl with {len(graph_dict)} timestamps")
    
    # Build entity histories
    def build_entity_history(data, all_prev_data=None):
        """Build subject and object history for each quadruple"""
        s_history_all = defaultdict(list)
        o_history_all = defaultdict(list)
        s_history_t_all = defaultdict(list)
        o_history_t_all = defaultdict(list)
        
        if all_prev_data is not None:
            time_to_quads = defaultdict(list)
            for quad in all_prev_data:
                time_to_quads[quad[3]].append(quad)
            
            for t in sorted(time_to_quads.keys()):
                quads_at_t = time_to_quads[t]
                s_to_ro = defaultdict(list)
                o_to_rs = defaultdict(list)
                for q in quads_at_t:
                    s, r, o = q[0], q[1], q[2]
                    s_to_ro[s].append([r, o])
                    o_to_rs[o].append([r, s])
                
                for s, ro_list in s_to_ro.items():
                    s_history_all[s].append(np.array(ro_list))
                    s_history_t_all[s].append(t)
                for o, rs_list in o_to_rs.items():
                    o_history_all[o].append(np.array(rs_list))
                    o_history_t_all[o].append(t)
        
        s_history = []
        s_history_t = []
        o_history = []
        o_history_t = []
        
        current_t = None
        pending_updates_s = defaultdict(list)
        pending_updates_o = defaultdict(list)
        
        for quad in data:
            s, r, o, t = quad
            
            if current_t is not None and current_t != t:
                for ent, ro_list in pending_updates_s.items():
                    s_history_all[ent].append(np.array(ro_list))
                    s_history_t_all[ent].append(current_t)
                for ent, rs_list in pending_updates_o.items():
                    o_history_all[ent].append(np.array(rs_list))
                    o_history_t_all[ent].append(current_t)
                pending_updates_s.clear()
                pending_updates_o.clear()
            
            s_history.append([arr.copy() for arr in s_history_all[s]])
            s_history_t.append(s_history_t_all[s].copy())
            o_history.append([arr.copy() for arr in o_history_all[o]])
            o_history_t.append(o_history_t_all[o].copy())
            
            pending_updates_s[s].append([r, o])
            pending_updates_o[o].append([r, s])
            current_t = t
        
        return (s_history, s_history_t), (o_history, o_history_t)
    
    print("Building train history...")
    train_s_hist, train_o_hist = build_entity_history(train_data)
    with open(os.path.join(history_dir, 'train_history_sub.pkl'), 'wb') as f:
        pickle.dump(train_s_hist, f)
    with open(os.path.join(history_dir, 'train_history_ob.pkl'), 'wb') as f:
        pickle.dump(train_o_hist, f)
    
    print("Building valid history...")
    valid_s_hist, valid_o_hist = build_entity_history(valid_data, train_data)
    with open(os.path.join(history_dir, 'dev_history_sub.pkl'), 'wb') as f:
        pickle.dump(valid_s_hist, f)
    with open(os.path.join(history_dir, 'dev_history_ob.pkl'), 'wb') as f:
        pickle.dump(valid_o_hist, f)
    
    print("Building test history...")
    all_prev = np.concatenate([train_data, valid_data], axis=0)
    test_s_hist, test_o_hist = build_entity_history(test_data, all_prev)
    with open(os.path.join(history_dir, 'test_history_sub.pkl'), 'wb') as f:
        pickle.dump(test_s_hist, f)
    with open(os.path.join(history_dir, 'test_history_ob.pkl'), 'wb') as f:
        pickle.dump(test_o_hist, f)
    
    print("Preprocessing complete!")
    return True


# =============================================================================
# RGCN LAYERS
# =============================================================================

class RGCNLayer(nn.Module):
    """Base RGCN layer"""
    def __init__(self, in_feat, out_feat, bias=None, activation=None,
                 self_loop=False, dropout=0.0):
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
    """RGCN layer with block decomposition"""
    def __init__(self, in_feat, out_feat, num_rels, num_bases, bias=None,
                 activation=None, self_loop=False, dropout=0.0):
        super().__init__(in_feat, out_feat, bias, activation, self_loop, dropout)
        self.num_rels = num_rels
        self.num_bases = num_bases
        assert self.num_bases > 0
        
        self.out_feat = out_feat
        self.submat_in = in_feat // self.num_bases
        self.submat_out = out_feat // self.num_bases
        
        self.weight = nn.Parameter(torch.Tensor(
            self.num_rels, self.num_bases * self.submat_in * self.submat_out))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

    def msg_func(self, edges, reverse):
        if reverse:
            weight = self.weight.index_select(0, edges.data['type_o']).view(
                -1, self.submat_in, self.submat_out)
        else:
            weight = self.weight.index_select(0, edges.data['type_s']).view(
                -1, self.submat_in, self.submat_out)
        node = edges.src['h'].view(-1, 1, self.submat_in)
        msg = torch.bmm(node, weight).view(-1, self.out_feat)
        return {'msg': msg}

    def propagate(self, g, reverse):
        g.update_all(lambda x: self.msg_func(x, reverse), 
                     fn.sum(msg='msg', out='h'), 
                     self.apply_func)

    def apply_func(self, nodes):
        return {'h': nodes.data['h'] * nodes.data['norm']}


# =============================================================================
# AGGREGATORS
# =============================================================================

class RGCNAggregatorGlobal(nn.Module):
    """Global RGCN aggregator for graph-level representations"""
    def __init__(self, h_dim, dropout, num_nodes, num_rels, num_bases, seq_len=10, maxpool=1):
        super().__init__()
        self.h_dim = h_dim
        self.dropout = nn.Dropout(dropout)
        self.seq_len = seq_len
        self.num_rels = num_rels
        self.num_nodes = num_nodes
        self.maxpool = maxpool
        
        self.rgcn1 = RGCNBlockLayer(h_dim, h_dim, 2 * num_rels, num_bases,
                                    activation=F.relu, self_loop=True, dropout=dropout)
        self.rgcn2 = RGCNBlockLayer(h_dim, h_dim, 2 * num_rels, num_bases,
                                    activation=None, self_loop=True, dropout=dropout)

    def forward(self, t_list, ent_embeds, graph_dict, device, reverse=False):
        """
        Args:
            t_list: tensor of timestamps on device
            ent_embeds: entity embeddings on device
            graph_dict: dictionary of graphs (on CPU)
            device: target device
            reverse: whether to use reverse relations
        """
        times = sorted(graph_dict.keys())
        if len(times) < 2:
            return None
        time_unit = times[1] - times[0]
        
        # Move t_list to CPU for indexing
        t_list_cpu = t_list.cpu()
        num_non_zero = len(torch.nonzero(t_list_cpu, as_tuple=False))
        if num_non_zero == 0:
            return None
        t_list_cpu = t_list_cpu[:num_non_zero]
        
        time_list = []
        len_non_zero = []
        
        for tim in t_list_cpu:
            tim_val = tim.item()
            # Find index in times list
            length = 0
            for tt in times:
                if tt < tim_val:
                    length += 1
                else:
                    break
            
            if self.seq_len <= length:
                time_list.append(times[length - self.seq_len:length])
                len_non_zero.append(self.seq_len)
            elif length > 0:
                time_list.append(times[:length])
                len_non_zero.append(length)
            else:
                time_list.append([])
                len_non_zero.append(0)
        
        # Filter out empty sequences
        valid_idx = [i for i, l in enumerate(len_non_zero) if l > 0]
        if len(valid_idx) == 0:
            return None
        
        time_list = [time_list[i] for i in valid_idx]
        len_non_zero = [len_non_zero[i] for i in valid_idx]
        
        # Build g_list and time_to_idx together
        unique_t = sorted(set(t for tl in time_list for t in tl))
        g_list = []
        time_to_idx = {}
        
        for t in unique_t:
            if t in graph_dict and graph_dict[t] is not None:
                time_to_idx[t] = len(g_list)
                g_moved = move_dgl_to_cuda(graph_dict[t], device)
                g_list.append(g_moved)
        
        if len(g_list) == 0:
            return None
        
        batched_graph = dgl.batch(g_list)
        batched_graph.ndata['h'] = ent_embeds[batched_graph.ndata['id'].view(-1)].view(-1, self.h_dim)
        
        self.rgcn1(batched_graph, reverse)
        self.rgcn2(batched_graph, reverse)
        
        if self.maxpool == 1:
            global_info = dgl.max_nodes(batched_graph, 'h')
        else:
            global_info = dgl.mean_nodes(batched_graph, 'h')
        
        batched_graph.ndata.pop('h')
        
        embed_seq_tensor = torch.zeros(len(len_non_zero), self.seq_len, self.h_dim, device=device)
        for i, times_i in enumerate(time_list):
            for j, t in enumerate(times_i):
                if t in time_to_idx:
                    embed_seq_tensor[i, j, :] = global_info[time_to_idx[t]]
        
        embed_seq_tensor = self.dropout(embed_seq_tensor)
        packed_input = nn.utils.rnn.pack_padded_sequence(
            embed_seq_tensor, len_non_zero, batch_first=True, enforce_sorted=False)
        return packed_input

    def predict(self, t, ent_embeds, graph_dict, device, reverse=False):
        times = sorted(graph_dict.keys())
        if len(times) == 0:
            return torch.zeros(1, self.h_dim, device=device)
        
        # Find index
        idx = 0
        for tt in times:
            if tt >= t:
                break
            idx += 1
        
        if self.seq_len <= idx:
            time_seq = times[idx - self.seq_len:idx]
        else:
            time_seq = times[:idx]
        
        if len(time_seq) == 0:
            return torch.zeros(1, self.h_dim, device=device)
        
        g_list = []
        for tim in time_seq:
            if tim in graph_dict and graph_dict[tim] is not None:
                g_moved = move_dgl_to_cuda(graph_dict[tim], device)
                g_list.append(g_moved)
        
        if len(g_list) == 0:
            return torch.zeros(1, self.h_dim, device=device)
        
        batched_graph = dgl.batch(g_list)
        batched_graph.ndata['h'] = ent_embeds[batched_graph.ndata['id'].view(-1)].view(-1, self.h_dim)
        
        self.rgcn1(batched_graph, reverse)
        self.rgcn2(batched_graph, reverse)
        
        if self.maxpool == 1:
            global_info = dgl.max_nodes(batched_graph, 'h')
        else:
            global_info = dgl.mean_nodes(batched_graph, 'h')
        
        batched_graph.ndata.pop('h')
        return global_info


class RGCNAggregatorLocal(nn.Module):
    """Local RGCN aggregator for entity-specific history"""
    def __init__(self, h_dim, dropout, num_nodes, num_rels, num_bases, seq_len=10):
        super().__init__()
        self.h_dim = h_dim
        self.dropout = nn.Dropout(dropout)
        self.seq_len = seq_len
        self.num_rels = num_rels
        self.num_nodes = num_nodes
        
        self.rgcn1 = RGCNBlockLayer(h_dim, h_dim, 2 * num_rels, num_bases,
                                    activation=F.relu, self_loop=True, dropout=dropout)
        self.rgcn2 = RGCNBlockLayer(h_dim, h_dim, 2 * num_rels, num_bases,
                                    activation=None, self_loop=True, dropout=dropout)

    def forward(self, s_hist, s, r, ent_embeds, rel_embeds, graph_dict, 
                global_emb, device, reverse=False):
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
        s_tem = s[s_idx]
        r_tem = r[s_idx]
        
        neighs_t = defaultdict(set)
        for i, (hist, hist_t) in enumerate(zip(s_hist_sorted, s_hist_t_sorted)):
            for neighs, t in zip(hist, hist_t):
                if len(neighs) > 0:
                    neighs_t[t].update(neighs[:, 1].tolist())
                    neighs_t[t].add(s_tem[i].item())
        
        if len(neighs_t) == 0:
            return None, None
        
        g_list = []
        g_id_dict = {}
        for tim in sorted(neighs_t.keys()):
            if tim in graph_dict and graph_dict[tim] is not None:
                sub_g = make_subgraph(graph_dict[tim], neighs_t[tim])
                if sub_g is not None:
                    g_id_dict[tim] = len(g_list)
                    if len(g_list) == 0:
                        sub_g.start_id = 0
                    else:
                        sub_g.start_id = g_list[-1].start_id + g_list[-1].number_of_nodes()
                    g_list.append(sub_g)
        
        if len(g_list) == 0:
            return None, None
        
        node_ids_graph = []
        global_emb_list = []
        actual_lens = []
        
        for i, (hist, hist_t) in enumerate(zip(s_hist_sorted, s_hist_t_sorted)):
            count = 0
            for j, (neighs, t) in enumerate(zip(hist, hist_t)):
                if t in g_id_dict and len(neighs) > 0:
                    graph = g_list[g_id_dict[t]]
                    s_val = s_tem[i].item()
                    if s_val in graph.ids:
                        node_ids_graph.append(graph.ids[s_val] + graph.start_id)
                        if t in global_emb:
                            global_emb_list.append(global_emb[t].view(1, -1))
                        else:
                            global_emb_list.append(torch.zeros(1, self.h_dim, device=device))
                        count += 1
            actual_lens.append(min(count, self.seq_len) if count > 0 else 0)
        
        if len(node_ids_graph) == 0:
            return None, None
        
        g_list = [move_dgl_to_cuda(g, device) for g in g_list]
        
        batched_graph = dgl.batch(g_list)
        batched_graph.ndata['h'] = ent_embeds[batched_graph.ndata['id'].view(-1)].view(-1, self.h_dim)
        
        self.rgcn1(batched_graph, reverse)
        self.rgcn2(batched_graph, reverse)
        
        embeds_mean = batched_graph.ndata.pop('h')
        node_ids_tensor = torch.LongTensor(node_ids_graph).to(device)
        embeds_mean = embeds_mean[node_ids_tensor]
        
        if len(global_emb_list) > 0:
            global_emb_cat = torch.cat(global_emb_list, dim=0).to(device)
        else:
            global_emb_cat = torch.zeros(len(node_ids_graph), self.h_dim, device=device)
        
        s_embed_seq = torch.zeros(num_non_zero, self.seq_len, 4 * self.h_dim, device=device)
        s_embed_seq_r = torch.zeros(num_non_zero, self.seq_len, 3 * self.h_dim, device=device)
        
        embed_idx = 0
        final_lens = []
        for i in range(num_non_zero):
            hist_len_i = actual_lens[i]
            if hist_len_i > 0 and embed_idx < len(embeds_mean):
                avail = min(hist_len_i, len(embeds_mean) - embed_idx)
                for j in range(avail):
                    idx = embed_idx + j
                    if idx < len(embeds_mean):
                        s_embed_seq[i, j, :] = torch.cat([
                            embeds_mean[idx],
                            ent_embeds[s_tem[i]],
                            rel_embeds[r_tem[i]],
                            global_emb_cat[idx] if idx < len(global_emb_cat) else torch.zeros(self.h_dim, device=device)
                        ])
                        s_embed_seq_r[i, j, :] = torch.cat([
                            embeds_mean[idx],
                            ent_embeds[s_tem[i]],
                            global_emb_cat[idx] if idx < len(global_emb_cat) else torch.zeros(self.h_dim, device=device)
                        ])
                embed_idx += avail
                final_lens.append(max(1, avail))
            else:
                final_lens.append(1)
                s_embed_seq[i, 0, :self.h_dim] = ent_embeds[s_tem[i]]
                s_embed_seq_r[i, 0, :self.h_dim] = ent_embeds[s_tem[i]]
        
        s_embed_seq = self.dropout(s_embed_seq)
        s_embed_seq_r = self.dropout(s_embed_seq_r)
        
        s_packed = nn.utils.rnn.pack_padded_sequence(
            s_embed_seq, final_lens, batch_first=True, enforce_sorted=False)
        s_packed_r = nn.utils.rnn.pack_padded_sequence(
            s_embed_seq_r, final_lens, batch_first=True, enforce_sorted=False)
        
        return s_packed, s_packed_r


# =============================================================================
# MODELS
# =============================================================================

class RENetGlobal(nn.Module):
    """Global model for RE-NET"""
    def __init__(self, num_nodes, h_dim, num_rels, dropout=0.5, seq_len=10, 
                 num_bases=100, maxpool=1):
        super().__init__()
        self.num_nodes = num_nodes
        self.h_dim = h_dim
        self.num_rels = num_rels
        self.seq_len = seq_len
        
        self.ent_embeds = nn.Parameter(torch.Tensor(num_nodes, h_dim))
        nn.init.xavier_uniform_(self.ent_embeds, gain=nn.init.calculate_gain('relu'))
        
        self.dropout = nn.Dropout(dropout)
        self.encoder_global = nn.GRU(h_dim, h_dim, batch_first=True)
        
        self.aggregator = RGCNAggregatorGlobal(h_dim, dropout, num_nodes, num_rels, 
                                                num_bases, seq_len, maxpool)
        
        self.linear_s = nn.Linear(h_dim, num_nodes)
        self.linear_o = nn.Linear(h_dim, num_nodes)
        self.global_emb = None

    def forward(self, t_list, true_prob_s, true_prob_o, graph_dict, device, subject=True):
        if subject:
            reverse = False
            linear = self.linear_s
            true_prob = true_prob_o
        else:
            reverse = True
            linear = self.linear_o
            true_prob = true_prob_s
        
        # Sort on CPU, keep idx on CPU for numpy indexing
        t_list_cpu = t_list.cpu()
        sorted_t_cpu, idx_cpu = t_list_cpu.sort(0, descending=True)
        sorted_t = sorted_t_cpu.to(device)
        
        packed_input = self.aggregator(sorted_t, self.ent_embeds, graph_dict, device, reverse)
        
        if packed_input is None:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        _, s_q = self.encoder_global(packed_input)
        s_q = s_q.squeeze(0)
        
        if len(s_q.shape) == 1:
            s_q = s_q.unsqueeze(0)
        
        if s_q.size(0) < len(t_list):
            padding = torch.zeros(len(t_list) - s_q.size(0), self.h_dim, device=device)
            s_q = torch.cat([s_q, padding], dim=0)
        
        pred = linear(s_q)
        
        # Use CPU index for numpy array, then move result to device
        idx_numpy = idx_cpu.numpy()
        true_prob_indexed = torch.from_numpy(true_prob[idx_numpy]).float().to(device)
        
        loss = soft_cross_entropy(pred, true_prob_indexed)
        return loss

    def get_global_emb(self, t_list, graph_dict, device):
        """Get global embeddings for all timestamps"""
        global_emb = {}
        times = sorted(graph_dict.keys())
        if len(times) < 2:
            return global_emb
        time_unit = times[1] - times[0]
        
        prev_t = 0
        for t in t_list:
            if t == 0:
                continue
            emb = self.predict(t, graph_dict, device)
            global_emb[prev_t] = emb.detach()
            prev_t = t
        
        if len(t_list) > 0:
            global_emb[t_list[-1]] = self.predict(t_list[-1] + time_unit, graph_dict, device).detach()
        return global_emb

    def predict(self, t, graph_dict, device, subject=True):
        if subject:
            reverse = False
        else:
            reverse = True
        
        rnn_inp = self.aggregator.predict(t, self.ent_embeds, graph_dict, device, reverse)
        if rnn_inp.dim() == 2:
            rnn_inp = rnn_inp.unsqueeze(0)
        _, s_q = self.encoder_global(rnn_inp.view(1, -1, self.h_dim))
        return s_q.squeeze(0)


class RENet(nn.Module):
    """Main RE-NET model for link prediction"""
    def __init__(self, num_nodes, h_dim, num_rels, dropout=0.5, seq_len=10, num_bases=100):
        super().__init__()
        self.num_nodes = num_nodes
        self.h_dim = h_dim
        self.num_rels = num_rels
        self.seq_len = seq_len
        
        self.rel_embeds = nn.Parameter(torch.Tensor(2 * num_rels, h_dim))
        nn.init.xavier_uniform_(self.rel_embeds, gain=nn.init.calculate_gain('relu'))
        
        self.ent_embeds = nn.Parameter(torch.Tensor(num_nodes, h_dim))
        nn.init.xavier_uniform_(self.ent_embeds, gain=nn.init.calculate_gain('relu'))
        
        self.dropout = nn.Dropout(dropout)
        self.encoder = nn.GRU(4 * h_dim, h_dim, batch_first=True)
        self.encoder_r = nn.GRU(3 * h_dim, h_dim, batch_first=True)
        
        self.aggregator = RGCNAggregatorLocal(h_dim, dropout, num_nodes, num_rels, 
                                               num_bases, seq_len)
        
        self.linear = nn.Linear(3 * h_dim, num_nodes)
        self.linear_r = nn.Linear(2 * h_dim, num_rels)
        
        self.criterion = nn.CrossEntropyLoss()
        
        self.global_emb = None
        self.graph_dict = None
        self.s_hist_test = None
        self.o_hist_test = None
        self.s_hist_test_t = None
        self.o_hist_test_t = None

    def forward(self, triplets, s_hist, o_hist, graph_dict, global_emb, device, subject=True):
        if subject:
            rel_embeds = self.rel_embeds[:self.num_rels]
            s = triplets[:, 0]
            r = triplets[:, 1]
            o = triplets[:, 2]
            hist = s_hist
            reverse = False
        else:
            rel_embeds = self.rel_embeds[self.num_rels:]
            o = triplets[:, 0]
            r = triplets[:, 1]
            s = triplets[:, 2]
            hist = o_hist
            reverse = True
        
        hist_len = torch.LongTensor([len(h) for h in hist[0]])
        s_len, s_idx = hist_len.sort(0, descending=True)
        
        s_packed, s_packed_r = self.aggregator(
            hist, s, r, self.ent_embeds, rel_embeds, graph_dict, global_emb, device, reverse)
        
        if s_packed is None:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        _, s_h = self.encoder(s_packed)
        s_h = s_h.squeeze(0)
        if len(s_h.shape) == 1:
            s_h = s_h.unsqueeze(0)
        
        if s_h.size(0) < len(s):
            padding = torch.zeros(len(s) - s_h.size(0), self.h_dim, device=device)
            s_h = torch.cat([s_h, padding], dim=0)
        
        s_idx = s_idx.to(device)
        ob_pred = self.linear(self.dropout(torch.cat([
            self.ent_embeds[s[s_idx]], s_h, rel_embeds[r[s_idx]]], dim=1)))
        loss_sub = self.criterion(ob_pred, o[s_idx])
        
        _, s_q = self.encoder_r(s_packed_r)
        s_q = s_q.squeeze(0)
        if len(s_q.shape) == 1:
            s_q = s_q.unsqueeze(0)
        if s_q.size(0) < len(s):
            padding = torch.zeros(len(s) - s_q.size(0), self.h_dim, device=device)
            s_q = torch.cat([s_q, padding], dim=0)
        
        ob_pred_r = self.linear_r(self.dropout(torch.cat([
            self.ent_embeds[s[s_idx]], s_q], dim=1)))
        loss_sub_r = self.criterion(ob_pred_r, r[s_idx])
        
        return loss_sub + 0.1 * loss_sub_r

    def predict_single(self, triplet, s_hist, o_hist, global_emb, graph_dict, device):
        """Predict for a single triplet during testing"""
        s, r, o = triplet[0].item(), triplet[1].item(), triplet[2].item()
        
        s_h = torch.zeros(self.h_dim, device=device)
        o_h = torch.zeros(self.h_dim, device=device)
        
        ob_pred = self.linear(torch.cat([
            self.ent_embeds[s], s_h, self.rel_embeds[:self.num_rels][r]]))
        
        sub_pred = self.linear(torch.cat([
            self.ent_embeds[o], o_h, self.rel_embeds[self.num_rels:][r]]))
        
        return sub_pred, ob_pred


# =============================================================================
# EVALUATION METRICS
# =============================================================================

def calc_mrr_hits(ranks):
    """Calculate MRR and Hits@k from ranks"""
    ranks = np.array(ranks)
    mrr = np.mean(1.0 / ranks)
    hits = {}
    for k in [1, 3, 10]:
        hits[k] = np.mean(ranks <= k)
    return mrr, hits


def calc_brier_score(probs, targets, num_classes):
    """Calculate Brier score"""
    one_hot = np.zeros((len(targets), num_classes))
    one_hot[np.arange(len(targets)), targets] = 1
    return np.mean(np.sum((probs - one_hot) ** 2, axis=1))


def calc_ece(probs, targets, num_bins=10):
    """Calculate Expected Calibration Error"""
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == targets).astype(float)
    
    ece = 0.0
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    
    for i in range(num_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece


def evaluate_filter(pred, target, all_triplets, s, r, is_subject=True):
    """Filtered evaluation"""
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
    
    rank = (pred > ground).sum().item() + 1
    return rank


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def pretrain_global(args):
    """Pretrain the global model"""
    set_seed(args.seed)
    data_path = os.path.join(args.data_dir, args.dataset)
    
    num_nodes, num_rels = get_total_number(data_path, 'stat.txt')
    train_data, train_times = load_quadruples(data_path, 'train.txt')
    
    print(f"Dataset: {args.dataset}")
    print(f"Entities: {num_nodes}, Relations: {num_rels}")
    print(f"Training samples: {len(train_data)}, Timestamps: {len(train_times)}")
    
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        torch.cuda.set_device(args.gpu)
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    history_path = os.path.join(data_path, 'history')
    with open(os.path.join(history_path, 'train_graphs.pkl'), 'rb') as f:
        graph_dict = pickle.load(f)
    
    true_prob_s, true_prob_o = get_true_distribution(train_data, num_nodes)
    
    model = RENetGlobal(num_nodes, args.n_hidden, num_rels, 
                        dropout=args.dropout, seq_len=args.seq_len,
                        num_bases=args.num_bases, maxpool=args.maxpool)
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    
    model_dir = os.path.join(args.model_dir, args.dataset)
    os.makedirs(model_dir, exist_ok=True)
    model_name = f"global-{args.dataset}-h{args.n_hidden}-seq{args.seq_len}-dp{args.dropout}-lr{args.lr}-gpu{args.gpu}-maxpool{args.maxpool}"
    model_path = os.path.join(model_dir, f"{model_name}.pth")
    
    print(f"Model will be saved to: {model_path}")
    
    best_loss = float('inf')
    
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        t0 = time.time()
        
        perm = np.random.permutation(len(train_times))
        train_times_shuffled = train_times[perm]
        true_prob_s_shuffled = true_prob_s[perm]
        true_prob_o_shuffled = true_prob_o[perm]
        
        epoch_loss = 0
        n_batches = 0
        
        num_batches = (len(train_times_shuffled) + args.batch_size - 1) // args.batch_size
        pbar = tqdm(range(0, len(train_times_shuffled), args.batch_size),
                    desc=f"Epoch {epoch:04d}", total=num_batches)
        
        for i in pbar:
            batch_times = train_times_shuffled[i:i+args.batch_size]
            batch_prob_s = true_prob_s_shuffled[i:i+args.batch_size]
            batch_prob_o = true_prob_o_shuffled[i:i+args.batch_size]
            
            batch_times = torch.from_numpy(batch_times).long().to(device)
            
            loss = model(batch_times, batch_prob_s, batch_prob_o, graph_dict, device, subject=True)
            
            if loss.requires_grad:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
                optimizer.step()
                optimizer.zero_grad()
            
            epoch_loss += loss.item()
            n_batches += 1
            
            # Update progress bar with current loss
            pbar.set_postfix({'loss': f'{epoch_loss / n_batches:.4f}'})
        
        model.global_emb = model.get_global_emb(train_times.tolist(), graph_dict, device)
        
        avg_loss = epoch_loss / max(n_batches, 1)
        elapsed = time.time() - t0
        print(f"Epoch {epoch:04d} | Loss: {avg_loss:.4f} | Time: {elapsed:.2f}s")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'state_dict': model.state_dict(),
                'global_emb': model.global_emb,
                'epoch': epoch
            }, model_path)
            print(f"  Saved model (best loss: {best_loss:.4f})")
    
    print(f"Pretraining complete. Best loss: {best_loss:.4f}")
    return model_path


def train_model(args):
    """Train the main RE-NET model"""
    set_seed(args.seed)
    data_path = os.path.join(args.data_dir, args.dataset)
    
    num_nodes, num_rels = get_total_number(data_path, 'stat.txt')
    train_data, train_times = load_quadruples(data_path, 'train.txt')
    valid_data, valid_times = load_quadruples(data_path, 'valid.txt')
    total_data, _ = load_quadruples(data_path, 'train.txt', 'valid.txt', 'test.txt')
    
    print(f"Dataset: {args.dataset}")
    print(f"Entities: {num_nodes}, Relations: {num_rels}")
    print(f"Train: {len(train_data)}, Valid: {len(valid_data)}")
    
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        torch.cuda.set_device(args.gpu)
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    history_path = os.path.join(data_path, 'history')
    with open(os.path.join(history_path, 'train_graphs.pkl'), 'rb') as f:
        graph_dict = pickle.load(f)
    with open(os.path.join(history_path, 'train_history_sub.pkl'), 'rb') as f:
        train_s_hist = pickle.load(f)
    with open(os.path.join(history_path, 'train_history_ob.pkl'), 'rb') as f:
        train_o_hist = pickle.load(f)
    with open(os.path.join(history_path, 'dev_history_sub.pkl'), 'rb') as f:
        valid_s_hist = pickle.load(f)
    with open(os.path.join(history_path, 'dev_history_ob.pkl'), 'rb') as f:
        valid_o_hist = pickle.load(f)
    
    model_dir = os.path.join(args.model_dir, args.dataset)
    global_model_name = f"global-{args.dataset}-h{args.n_hidden}-seq{args.seq_len}-dp{args.dropout}-lr{args.lr}-gpu{args.gpu}-maxpool{args.maxpool}"
    global_model_path = os.path.join(model_dir, f"{global_model_name}.pth")
    
    global_model = RENetGlobal(num_nodes, args.n_hidden, num_rels,
                               dropout=args.dropout, seq_len=args.seq_len,
                               num_bases=args.num_bases, maxpool=args.maxpool)
    
    if os.path.exists(global_model_path):
        checkpoint = torch.load(global_model_path, map_location=device, weights_only=False)
        global_model.load_state_dict(checkpoint['state_dict'])
        global_emb = checkpoint['global_emb']
        print(f"Loaded pretrained global model from {global_model_path}")
    else:
        print(f"Warning: No pretrained global model found at {global_model_path}")
        global_emb = {}
    
    global_model = global_model.to(device)
    global_model.eval()
    
    model = RENet(num_nodes, args.n_hidden, num_rels,
                  dropout=args.dropout, seq_len=args.seq_len,
                  num_bases=args.num_bases)
    model = model.to(device)
    model.global_emb = global_emb
    model.graph_dict = graph_dict
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    
    model_name = f"renet-{args.dataset}-h{args.n_hidden}-seq{args.seq_len}-dp{args.dropout}-lr{args.lr}-gpu{args.gpu}"
    model_path = os.path.join(model_dir, f"{model_name}.pth")
    
    print(f"Model will be saved to: {model_path}")
    
    total_data_tensor = torch.from_numpy(total_data).to(device)
    best_mrr = 0.0
    
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        t0 = time.time()
        
        perm = np.random.permutation(len(train_data))
        
        epoch_loss = 0
        n_batches = 0
        
        num_batches = (len(train_data) + args.batch_size - 1) // args.batch_size
        pbar = tqdm(range(0, len(train_data), args.batch_size), 
                    desc=f"Epoch {epoch:04d}", total=num_batches)
        
        for i in pbar:
            batch_idx = perm[i:i+args.batch_size]
            batch_data = torch.from_numpy(train_data[batch_idx]).long().to(device)
            batch_s_hist = ([train_s_hist[0][j] for j in batch_idx], 
                           [train_s_hist[1][j] for j in batch_idx])
            batch_o_hist = ([train_o_hist[0][j] for j in batch_idx],
                           [train_o_hist[1][j] for j in batch_idx])
            
            loss_s = model(batch_data, batch_s_hist, batch_o_hist, 
                          graph_dict, global_emb, device, subject=True)
            loss_o = model(batch_data, batch_s_hist, batch_o_hist,
                          graph_dict, global_emb, device, subject=False)
            loss = loss_s + loss_o
            
            if loss.requires_grad:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
                optimizer.step()
                optimizer.zero_grad()
            
            epoch_loss += loss.item()
            n_batches += 1
            
            # Update progress bar with current loss
            pbar.set_postfix({'loss': f'{epoch_loss / n_batches:.4f}'})
        
        avg_loss = epoch_loss / max(n_batches, 1)
        elapsed = time.time() - t0
        print(f"Epoch {epoch:04d} | Loss: {avg_loss:.4f} | Time: {elapsed:.2f}s")
        
        if epoch % args.valid_every == 0 : 
            model.eval()
            ranks = []
            
            with torch.no_grad():
                for i in tqdm(range(len(valid_data)), desc="Validating"):
                    triplet = torch.from_numpy(valid_data[i]).long().to(device)
                    s_hist = (valid_s_hist[0][i], valid_s_hist[1][i])
                    o_hist = (valid_o_hist[0][i], valid_o_hist[1][i])
                    
                    sub_pred, ob_pred = model.predict_single(
                        triplet, s_hist, o_hist, global_emb, graph_dict, device)
                    
                    s, r, o = triplet[0].item(), triplet[1].item(), triplet[2].item()
                    
                    rank_o = evaluate_filter(ob_pred, o, total_data_tensor, s, r, is_subject=True)
                    rank_s = evaluate_filter(sub_pred, s, total_data_tensor, o, r, is_subject=False)
                    
                    ranks.extend([rank_s, rank_o])
            
            mrr, hits = calc_mrr_hits(ranks)
            print(f"  Valid MRR: {mrr:.4f} | Hits@1: {hits[1]:.4f} | Hits@3: {hits[3]:.4f} | Hits@10: {hits[10]:.4f}")
            
            if mrr > best_mrr:
                best_mrr = mrr
                torch.save({
                    'state_dict': model.state_dict(),
                    'epoch': epoch,
                    'global_emb': global_emb
                }, model_path)
                print(f"  Saved model (best MRR: {best_mrr:.4f})")
    
    print(f"Training complete. Best MRR: {best_mrr:.4f}")
    return model_path


def test_model(args):
    """Test the trained RE-NET model"""
    set_seed(args.seed)
    data_path = os.path.join(args.data_dir, args.dataset)
    
    num_nodes, num_rels = get_total_number(data_path, 'stat.txt')
    test_data, test_times = load_quadruples(data_path, 'test.txt')
    total_data, _ = load_quadruples(data_path, 'train.txt', 'valid.txt', 'test.txt')
    
    print(f"Dataset: {args.dataset}")
    print(f"Test samples: {len(test_data)}")
    
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        torch.cuda.set_device(args.gpu)
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    history_path = os.path.join(data_path, 'history')
    with open(os.path.join(history_path, 'train_graphs.pkl'), 'rb') as f:
        graph_dict = pickle.load(f)
    with open(os.path.join(history_path, 'test_history_sub.pkl'), 'rb') as f:
        test_s_hist = pickle.load(f)
    with open(os.path.join(history_path, 'test_history_ob.pkl'), 'rb') as f:
        test_o_hist = pickle.load(f)
    
    model_dir = os.path.join(args.model_dir, args.dataset)
    model_name = f"renet-{args.dataset}-h{args.n_hidden}-seq{args.seq_len}-dp{args.dropout}-lr{args.lr}-gpu{args.gpu}"
    model_path = os.path.join(model_dir, f"{model_name}.pth")
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
    
    model = RENet(num_nodes, args.n_hidden, num_rels,
                  dropout=args.dropout, seq_len=args.seq_len,
                  num_bases=args.num_bases)
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    global_emb = checkpoint.get('global_emb', {})
    print(f"Loaded model from {model_path}, epoch {checkpoint['epoch']}")
    
    model = model.to(device)
    model.eval()
    
    total_data_tensor = torch.from_numpy(total_data).to(device)
    
    ranks = []
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for i in tqdm(range(len(test_data)), desc="Testing"):
            triplet = torch.from_numpy(test_data[i]).long().to(device)
            s_hist = (test_s_hist[0][i], test_s_hist[1][i])
            o_hist = (test_o_hist[0][i], test_o_hist[1][i])
            
            sub_pred, ob_pred = model.predict_single(
                triplet, s_hist, o_hist, global_emb, graph_dict, device)
            
            s, r, o = triplet[0].item(), triplet[1].item(), triplet[2].item()
            
            rank_o = evaluate_filter(ob_pred, o, total_data_tensor, s, r, is_subject=True)
            rank_s = evaluate_filter(sub_pred, s, total_data_tensor, o, r, is_subject=False)
            
            ranks.extend([rank_s, rank_o])
            
            ob_probs = F.softmax(ob_pred, dim=0).cpu().numpy()
            sub_probs = F.softmax(sub_pred, dim=0).cpu().numpy()
            all_probs.extend([sub_probs, ob_probs])
            all_targets.extend([s, o])
    
    mrr, hits = calc_mrr_hits(ranks)
    
    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)
    brier = calc_brier_score(all_probs, all_targets, num_nodes)
    ece = calc_ece(all_probs, all_targets)
    
    print("\n" + "="*60)
    print("TEST RESULTS (Filtered)")
    print("="*60)
    print(f"MRR: {mrr:.6f}")
    print(f"Hits@1: {hits[1]:.6f}")
    print(f"Hits@3: {hits[3]:.6f}")
    print(f"Hits@10: {hits[10]:.6f}")
    print(f"Brier Score: {brier:.6f}")
    print(f"ECE: {ece:.6f}")
    print("="*60)
    
    return mrr, hits, brier, ece


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='RE-NET Reproduction')
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument('-d', '--dataset', type=str, default='ICEWS14',
                               choices=['ICEWS14', 'ICEWS18', 'ICEWS05-15'],
                               help='Dataset name')
    common_parser.add_argument('--data-dir', type=str, default='./data',
                               help='Data directory')
    common_parser.add_argument('--model-dir', type=str, default='./models',
                               help='Model save directory')
    common_parser.add_argument('--gpu', type=int, default=0,
                               help='GPU device ID (-1 for CPU)')
    common_parser.add_argument('--seed', type=int, default=999,
                               help='Random seed')
    common_parser.add_argument('--n-hidden', type=int, default=200,
                               help='Hidden dimension')
    common_parser.add_argument('--seq-len', type=int, default=10,
                               help='History sequence length')
    common_parser.add_argument('--num-bases', type=int, default=100,
                               help='Number of RGCN bases')
    common_parser.add_argument('--dropout', type=float, default=0.5,
                               help='Dropout rate')
    common_parser.add_argument('--lr', type=float, default=1e-3,
                               help='Learning rate')
    
    preprocess_parser = subparsers.add_parser('preprocess', parents=[common_parser],
                                              help='Preprocess dataset')
    
    pretrain_parser = subparsers.add_parser('pretrain', parents=[common_parser],
                                            help='Pretrain global model')
    pretrain_parser.add_argument('--max-epochs', type=int, default=20,
                                 help='Maximum epochs')
    pretrain_parser.add_argument('--batch-size', type=int, default=1024,
                                 help='Batch size')
    pretrain_parser.add_argument('--grad-norm', type=float, default=1.0,
                                 help='Gradient clipping norm')
    pretrain_parser.add_argument('--maxpool', type=int, default=1,
                                 help='Use maxpool (1) or meanpool (0)')
    
    train_parser = subparsers.add_parser('train', parents=[common_parser],
                                         help='Train RE-NET model')
    train_parser.add_argument('--max-epochs', type=int, default=20,
                              help='Maximum epochs')
    train_parser.add_argument('--batch-size', type=int, default=1024,
                              help='Batch size')
    train_parser.add_argument('--grad-norm', type=float, default=1.0,
                              help='Gradient clipping norm')
    train_parser.add_argument('--maxpool', type=int, default=1,
                              help='Use maxpool (1) or meanpool (0)')
    train_parser.add_argument('--valid-every', type=int, default=1,
                              help='Validate every N epochs')
    
    test_parser = subparsers.add_parser('test', parents=[common_parser],
                                        help='Test RE-NET model')
    test_parser.add_argument('--maxpool', type=int, default=1,
                             help='Use maxpool (1) or meanpool (0)')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    print(f"Command: {args.command}")
    print(f"Arguments: {args}")
    print()
    
    if args.command == 'preprocess':
        data_path = os.path.join(args.data_dir, args.dataset)
        get_history_graph(data_path, args.dataset)
    
    elif args.command == 'pretrain':
        pretrain_global(args)
    
    elif args.command == 'train':
        train_model(args)
    
    elif args.command == 'test':
        test_model(args)


if __name__ == '__main__':
    main()