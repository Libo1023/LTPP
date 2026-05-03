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
from scipy import stats

np.random.seed(123)
torch.manual_seed(123)


# ============================================================================
# PART 1: DATA LOADING UTILITIES
# ============================================================================

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
            yield line.strip().split('\t')


def _read_triplets_as_list(filename, entity_dict, relation_dict, load_time=True):
    l = []
    for triplet in _read_triplets(filename):
        s, r, o = int(triplet[0]), int(triplet[1]), int(triplet[2])
        if load_time:
            l.append([s, r, o, int(triplet[3])])
        else:
            l.append([s, r, o])
    return l


class RGCNLinkDataset:
    def __init__(self, name, dir_path):
        self.name = name
        self.dir = os.path.join(dir_path, self.name)
        print(f"Dataset directory: {self.dir}")

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
        self.num_rels = len(relation_dict)
        self.relation_dict = relation_dict
        self.entity_dict = entity_dict
        
        print(f"# Entities: {self.num_nodes}, Relations: {self.num_rels}")
        print(f"# Train: {len(self.train)}, Valid: {len(self.valid)}, Test: {len(self.test)}")


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
    print(f"# Snapshots: {len(snapshot_list)}")
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


def load_all_answers_for_filter(total_data, num_rel):
    all_ans = {}
    for line in total_data:
        s, r, o = line[:3]
        add_subject(s, o, r, all_ans, num_rel=num_rel)
        add_object(s, o, r, all_ans, num_rel=0)
    return all_ans


def load_all_answers_for_time_filter(total_data, num_rels, num_nodes):
    all_ans_list = []
    all_snap, _ = split_by_time(total_data)
    for snap in all_snap:
        all_ans_list.append(load_all_answers_for_filter(snap, num_rels))
    return all_ans_list


# ============================================================================
# PART 2: LONG-TAIL ENTITY UTILITIES
# ============================================================================

def count_entity_frequencies(train_data, num_rels):
    entity_freq = defaultdict(int)
    for quad in train_data:
        s, r, o, t = quad
        entity_freq[s] += 2
        entity_freq[o] += 2
    return entity_freq 


# ============================================================================
# PART 3: MODEL DEFINITION (TimeConvTransE Decoder)
# ============================================================================

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
        e1_embedded_all = F.tanh(embedding)
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


class RecurrentRGCN(nn.Module):
    def __init__(self, num_ents, num_rels, num_times, time_interval, h_dim,
                 input_dropout=0, hidden_dropout=0, feat_dropout=0, layer_norm=False):
        super(RecurrentRGCN, self).__init__()
        self.num_rels = num_rels
        self.num_ents = num_ents
        self.time_interval = time_interval
        self.h_dim = h_dim
        self.layer_norm = layer_norm
        
        self.entity_embeddings = nn.Parameter(torch.Tensor(num_ents, h_dim))
        torch.nn.init.xavier_normal_(self.entity_embeddings)
        self.relation_embeddings = nn.Parameter(torch.Tensor(num_rels * 2, h_dim))
        torch.nn.init.xavier_normal_(self.relation_embeddings)
        
        self.weight_t1 = nn.Parameter(torch.randn(1, h_dim))
        self.bias_t1 = nn.Parameter(torch.randn(1, h_dim))
        self.weight_t2 = nn.Parameter(torch.randn(1, h_dim))
        self.bias_t2 = nn.Parameter(torch.randn(1, h_dim))
        
        self.decoder = TimeConvTransE(num_ents, h_dim, input_dropout, hidden_dropout, feat_dropout)

    def get_init_time(self, quadrupleList):
        device = self.weight_t1.device
        T_idx = (quadrupleList[:, 3] // self.time_interval).unsqueeze(1).float().to(device)
        t1 = self.weight_t1 * T_idx + self.bias_t1
        t2 = torch.sin(self.weight_t2 * T_idx + self.bias_t2)
        return t1, t2

    def predict(self, test_triplets, num_rels):
        with torch.no_grad():
            inverse_test_triplets = test_triplets[:, [2, 1, 0, 3]]
            inverse_test_triplets[:, 1] = inverse_test_triplets[:, 1] + num_rels
            all_triples = torch.cat((test_triplets, inverse_test_triplets))
            embedding = F.normalize(self.entity_embeddings) if self.layer_norm else self.entity_embeddings
            time_embs = self.get_init_time(all_triples)
            scores_ob = self.decoder.forward(embedding, self.relation_embeddings, time_embs, all_triples)
            scores_ob = scores_ob.view(-1, self.num_ents)
            score = torch.log(F.softmax(scores_ob, dim=1))
            return all_triples, score


# ============================================================================
# PART 4: EVALUATION UTILITIES
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
        if h.item() in all_ans and r.item() in all_ans[h.item()]:
            ans = list(all_ans[h.item()][r.item()])
            if t.item() in ans:
                ans.remove(t.item())
            score[idx][torch.LongTensor(ans)] = -10000000
    return score


def get_total_rank(test_triples, score, all_ans, eval_bz=1000):
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


# ============================================================================
# PART 5: HARD MODE POST-PROCESSING CALIBRATOR
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

    def _compute_lt_mrr(self, triples, calibrated, targets):
        triples_np = triples.cpu().numpy()
        targets_cpu = targets.cpu()
        calibrated_cpu = calibrated.cpu()
        reciprocal_ranks = []
        for i, triple in enumerate(triples_np):
            s, r, o = int(triple[0]), int(triple[1]), int(triple[2])
            if self._get_true_query_type(s, o) == 'neither':
                continue
            target = int(targets_cpu[i])
            score_row = calibrated_cpu[i]
            rank = (score_row > score_row[target]).sum().item() + 1
            reciprocal_ranks.append(1.0 / rank)
        return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0

    def _evaluate_calibration(self, triples, scores, targets, entity_boost, relation_boost,
                               neighbor_w1, neighbor_w2, mult_both, mult_one,
                               adaptive_type, adaptive_factor):
        calibrated, _ = self._apply_calibration_hard(
            triples, scores, entity_boost, relation_boost,
            neighbor_w1, neighbor_w2, mult_both, mult_one,
            adaptive_type, adaptive_factor
        )
        return self._compute_lt_mrr(triples, calibrated, targets) 

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
        improv = 100 * (best_mrr - baseline_mrr) / baseline_mrr if baseline_mrr > 0 else 0
        print(f"\n  - Best LT MRR: {best_mrr:.6f} (+{improv:.2f}%)")
        rt = best_params['rank_thresholds']
        print(f"  - Params: mult_both={best_params['mult_both']}, mult_one={best_params['mult_one']}")
        print(f"    boost_scale={best_params['boost_scale']}, adaptive={best_params['adaptive_type']}/{best_params['adaptive_factor']}")
        print(f"    neighbors={best_params['neighbor_w1']}/{best_params['neighbor_w2']}, thresholds={rt}")

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
            self.mult_both, self.mult_one, self.adaptive_scale_type, self.adaptive_scale_factor,
            track_changes=track_changes, targets=targets
        )
        if track_changes:
            self.rank_change_stats = rank_changes
        return calibrated

    def get_rank_change_stats(self):
        return self.rank_change_stats

    def get_params_dict(self):
        return {
            'mode': 'hard', 'boost_scale': self.boost_scale,
            'neighbor_w1': self.neighbor_weight_1hop, 'neighbor_w2': self.neighbor_weight_2hop,
            'adaptive_type': self.adaptive_scale_type, 'adaptive_factor': self.adaptive_scale_factor,
            'mult_both': self.mult_both, 'mult_one': self.mult_one,
        }


# ============================================================================
# PART 6: WILCOXON SIGNED-RANK TEST
# ============================================================================

def compute_wilcoxon_test(ranks_before, ranks_after, name=""):
    """Perform Wilcoxon signed-rank test comparing ranks before and after."""
    ranks_before_np = ranks_before.cpu().numpy().astype(float)
    ranks_after_np = ranks_after.cpu().numpy().astype(float)
    differences = ranks_before_np - ranks_after_np  # positive = improvement
    
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
            'interpretation': "Insufficient non-zero differences (n < 10)"
        }
    
    try:
        statistic, p_value = stats.wilcoxon(ranks_before_np, ranks_after_np, alternative='two-sided')
        n = len(non_zero_diff)
        z_score = stats.norm.ppf(1 - p_value / 2) if p_value > 0 else 0
        effect_size = abs(z_score) / np.sqrt(n)
        
        if effect_size < 0.3:
            effect_interp = "small"
        elif effect_size < 0.5:
            effect_interp = "medium"
        else:
            effect_interp = "large"
        
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
    """Print Wilcoxon test results."""
    print("\n" + "=" * 80)
    print(f"WILCOXON SIGNED-RANK TEST: {results['name']}")
    print("=" * 80)
    n = results['n_samples']
    print(f"\nSample Statistics:")
    print(f"  Total samples:           {n:>10}")
    print(f"  Improved (rank down):    {results['n_improved']:>10} ({100*results['n_improved']/n:.2f}%)")
    print(f"  Worsened (rank up):      {results['n_worsened']:>10} ({100*results['n_worsened']/n:.2f}%)")
    print(f"  Unchanged:               {results['n_unchanged']:>10} ({100*results['n_unchanged']/n:.2f}%)")
    print(f"\nRank Statistics:")
    print(f"  Median rank (before):    {results['median_before']:>10.2f}")
    print(f"  Median rank (after):     {results['median_after']:>10.2f}")
    print(f"  Mean rank change:        {results['mean_change']:>10.4f}")
    print(f"\nTest Results:")
    if results['statistic'] is not None:
        print(f"  Test statistic (W):      {results['statistic']:>10.2f}")
        print(f"  P-value:                 {results['p_value']:>10.4e}")
        print(f"  Effect size (r):         {results['effect_size']:>10.4f}")
        print(f"  Significant (p<0.05):    {'Yes' if results['significant'] else 'No':>10}")
    print(f"\nInterpretation: {results['interpretation']}")
    print("=" * 80)


# ============================================================================
# PART 7: EVALUATION PIPELINE
# ============================================================================

def evaluate_model(model, test_list, num_rels, num_nodes, all_ans_list, longtail_entities,
                   calibrator=None, apply_calibration=False, dataset_name="Test",
                   track_rank_changes=False, collect_ranks=False):
    """Evaluate model with optional rank collection for significance testing."""
    ranks_raw, ranks_filter = [], []
    ranks_raw_lt, ranks_filter_lt = [], []
    all_test_triples, all_final_scores, all_longtail_masks, all_ans_collected = [], [], [], []
    total_rank_changes = {'boosted': 0, 'lowered': 0, 'unchanged': 0, 'skipped': 0, 'total': 0}
    
    # For Wilcoxon test
    all_individual_ranks_raw, all_individual_ranks_filter = [], []
    all_individual_ranks_raw_lt, all_individual_ranks_filter_lt = [], []
    
    device = torch.device('cpu')
    model.eval()
    
    for time_idx, test_snap in enumerate(tqdm(test_list, desc=f"Evaluating {dataset_name}")):
        test_triples_input = torch.LongTensor(test_snap).to(device)
        test_triples, final_score = model.predict(test_triples_input, num_rels)
        
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
            eval_score = calibrator.calibrate_scores(test_triples, final_score, num_rels,
                                                      targets=targets, track_changes=track_rank_changes)
            if track_rank_changes:
                snap_stats = calibrator.get_rank_change_stats()
                if snap_stats:
                    for k in total_rank_changes:
                        total_rank_changes[k] += snap_stats.get(k, 0)
        else:
            eval_score = final_score
        
        _, _, rank_raw, rank_filter = get_total_rank(test_triples, eval_score, all_ans_list[time_idx])
        ranks_raw.append(rank_raw)
        ranks_filter.append(rank_filter)
        
        if collect_ranks:
            all_individual_ranks_raw.append(rank_raw)
            all_individual_ranks_filter.append(rank_filter)
        
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
            _, _, rank_raw_lt, rank_filter_lt = get_total_rank(lt_test_triples, lt_eval_score, lt_all_ans)
            ranks_raw_lt.append(rank_raw_lt)
            ranks_filter_lt.append(rank_filter_lt)
            if collect_ranks:
                all_individual_ranks_raw_lt.append(rank_raw_lt)
                all_individual_ranks_filter_lt.append(rank_filter_lt)
    
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
    
    if collect_ranks:
        results['all_ranks_raw'] = torch.cat(all_individual_ranks_raw) if all_individual_ranks_raw else torch.tensor([])
        results['all_ranks_filter'] = torch.cat(all_individual_ranks_filter) if all_individual_ranks_filter else torch.tensor([])
        results['all_ranks_raw_lt'] = torch.cat(all_individual_ranks_raw_lt) if all_individual_ranks_raw_lt else torch.tensor([])
        results['all_ranks_filter_lt'] = torch.cat(all_individual_ranks_filter_lt) if all_individual_ranks_filter_lt else torch.tensor([])
    
    return results


# ============================================================================
# PART 8: PRINTING UTILITIES
# ============================================================================

def print_metrics_table(title, raw_mrr, raw_hits, filter_mrr, filter_hits, num_queries=None):
    print("\n" + "=" * 70)
    print(title)
    if num_queries is not None:
        print(f"Queries: {num_queries}")
    print("=" * 70)
    print(f"{'Metric':<15} {'Raw':>20} {'Filtered':>20}")
    print("-" * 70)
    print(f"{'MRR':<15} {raw_mrr:>20.6f} {filter_mrr:>20.6f}")
    print(f"{'Hits@1':<15} {raw_hits[0]:>20.6f} {filter_hits[0]:>20.6f}")
    print(f"{'Hits@3':<15} {raw_hits[1]:>20.6f} {filter_hits[1]:>20.6f}")
    print(f"{'Hits@10':<15} {raw_hits[2]:>20.6f} {filter_hits[2]:>20.6f}")
    print("=" * 70)


def print_comparison_table(title, before, after):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)
    print("\n>>> OVERALL")
    print(f"{'Metric':<12} {'Bef(Raw)':>12} {'Bef(Filt)':>12} {'Aft(Raw)':>12} {'Aft(Filt)':>12}")
    print("-" * 60)
    for i, m in enumerate(['MRR', 'Hits@1', 'Hits@3', 'Hits@10']):
        if m == 'MRR':
            print(f"{m:<12} {before['mrr_raw']:>12.6f} {before['mrr']:>12.6f} {after['mrr_raw']:>12.6f} {after['mrr']:>12.6f}")
        else:
            idx = i - 1
            print(f"{m:<12} {before['hits_raw'][idx]:>12.6f} {before['hits'][idx]:>12.6f} {after['hits_raw'][idx]:>12.6f} {after['hits'][idx]:>12.6f}")
    
    if before.get('mrr_lt', 0) > 0:
        print("\n>>> LONG-TAIL")
        print(f"{'Metric':<12} {'Bef(Raw)':>12} {'Bef(Filt)':>12} {'Aft(Raw)':>12} {'Aft(Filt)':>12}")
        print("-" * 60)
        for i, m in enumerate(['MRR', 'Hits@1', 'Hits@3', 'Hits@10']):
            if m == 'MRR':
                print(f"{m:<12} {before['mrr_lt_raw']:>12.6f} {before['mrr_lt']:>12.6f} {after['mrr_lt_raw']:>12.6f} {after['mrr_lt']:>12.6f}")
            else:
                idx = i - 1
                print(f"{m:<12} {before['hits_lt_raw'][idx]:>12.6f} {before['hits_lt'][idx]:>12.6f} {after['hits_lt_raw'][idx]:>12.6f} {after['hits_lt'][idx]:>12.6f}")
    print("=" * 90)


def print_rank_change_stats(rc, total):
    print("\n" + "=" * 70)
    print("RANK CHANGE STATISTICS")
    print("=" * 70)
    modified = rc['boosted'] + rc['lowered'] + rc['unchanged']
    print(f"Total: {total}, Skipped: {rc['skipped']} ({100*rc['skipped']/total:.1f}%), Modified: {modified} ({100*modified/total:.1f}%)")
    if modified > 0:
        print(f"  Boosted: {rc['boosted']} ({100*rc['boosted']/modified:.1f}%), Lowered: {rc['lowered']} ({100*rc['lowered']/modified:.1f}%), Unchanged: {rc['unchanged']} ({100*rc['unchanged']/modified:.1f}%)")
    print("=" * 70)


# ============================================================================
# PART 9: RESULTS SAVING
# ============================================================================

def save_results_to_file(args, before, after, calib_params, lt_stats, wilcoxon_results, grid_info):
    filename = f"1_convtranse_results_{args.dataset}_hard_gpu{args.gpu}.txt"
    filepath = os.path.join("./", filename)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(filepath, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("TKG EVALUATION - HARD MODE POST-PROCESSING + WILCOXON TEST\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Timestamp: {timestamp}\n\n")
        
        f.write("-" * 80 + "\nCONFIGURATION\n" + "-" * 80 + "\n")
        f.write(f"Dataset: {args.dataset}, GPU: {args.gpu}, LT threshold: {args.longtail_threshold}\n")
        f.write(f"Early stop ratio: {args.early_stop_ratio}\n\n")
        
        f.write("-" * 80 + "\nLONG-TAIL STATISTICS\n" + "-" * 80 + "\n")
        f.write(f"LT entities: {lt_stats['num_lt_entities']}/{lt_stats['num_entities']} ({lt_stats['lt_percentage']:.2f}%)\n\n")
        
        f.write("-" * 80 + "\nGRID SEARCH INFO\n" + "-" * 80 + "\n")
        f.write(f"Early stopped: {grid_info.get('early_stopped', False)}\n")
        f.write(f"Iterations: {grid_info.get('iterations_searched', 'N/A')}/{grid_info.get('total_iterations', 'N/A')}\n\n")
        
        # Before results
        f.write("-" * 80 + "\nBEFORE POST-PROCESSING\n" + "-" * 80 + "\n")
        f.write(f"Overall - MRR: {before['mrr']:.6f}, Hits@1/3/10: {before['hits'][0]:.6f}/{before['hits'][1]:.6f}/{before['hits'][2]:.6f}\n")
        f.write(f"LT - MRR: {before['mrr_lt']:.6f}, Hits@1/3/10: {before['hits_lt'][0]:.6f}/{before['hits_lt'][1]:.6f}/{before['hits_lt'][2]:.6f}\n")
        f.write(f"Queries: {before['total_queries']} total, {before['lt_queries']} LT\n\n")
        
        # After results
        f.write("-" * 80 + "\nAFTER POST-PROCESSING (HARD MODE)\n" + "-" * 80 + "\n")
        f.write(f"Overall - MRR: {after['mrr']:.6f}, Hits@1/3/10: {after['hits'][0]:.6f}/{after['hits'][1]:.6f}/{after['hits'][2]:.6f}\n")
        f.write(f"LT - MRR: {after['mrr_lt']:.6f}, Hits@1/3/10: {after['hits_lt'][0]:.6f}/{after['hits_lt'][1]:.6f}/{after['hits_lt'][2]:.6f}\n\n")
        
        # Improvements
        f.write("-" * 80 + "\nIMPROVEMENT (AFTER - BEFORE)\n" + "-" * 80 + "\n")
        f.write(f"Overall MRR: {after['mrr']-before['mrr']:+.6f}, LT MRR: {after['mrr_lt']-before['mrr_lt']:+.6f}\n\n")
        
        # Rank changes
        if after.get('rank_changes'):
            rc = after['rank_changes']
            f.write("-" * 80 + "\nRANK CHANGES\n" + "-" * 80 + "\n")
            f.write(f"Skipped: {rc['skipped']}, Boosted: {rc['boosted']}, Lowered: {rc['lowered']}, Unchanged: {rc['unchanged']}\n\n")
        
        # Wilcoxon results
        f.write("-" * 80 + "\nWILCOXON SIGNED-RANK TEST RESULTS\n" + "-" * 80 + "\n\n")
        for w in wilcoxon_results:
            f.write(f">>> {w['name']}\n")
            f.write(f"  Samples: {w['n_samples']}, Improved: {w['n_improved']}, Worsened: {w['n_worsened']}, Unchanged: {w['n_unchanged']}\n")
            f.write(f"  Median: {w['median_before']:.2f} -> {w['median_after']:.2f}, Mean change: {w['mean_change']:.4f}\n")
            if w['statistic'] is not None:
                f.write(f"  W={w['statistic']:.2f}, p={w['p_value']:.4e}, r={w['effect_size']:.4f}, Sig: {'Yes' if w['significant'] else 'No'}\n")
            f.write(f"  {w['interpretation']}\n\n")
        
        # Calibrator params
        f.write("-" * 80 + "\nCALIBRATOR PARAMETERS\n" + "-" * 80 + "\n")
        for k, v in calib_params.items():
            f.write(f"{k}: {v}\n")
        
        f.write("\n" + "=" * 80 + "\nEND OF RESULTS\n" + "=" * 80 + "\n")
    
    print(f"\n[INFO] Results saved to: {filepath}")
    return filepath


# ============================================================================
# PART 10: MAIN EXPERIMENT
# ============================================================================

def run_experiment(args):
    print("\n" + "=" * 70)
    print("LOADING DATA")
    print("=" * 70)
    data = load_data(args.dataset)
    train_list, train_times = split_by_time(data.train)
    valid_list, valid_times = split_by_time(data.valid)
    test_list, test_times = split_by_time(data.test)
    
    num_nodes, num_rels = data.num_nodes, data.num_rels
    num_times = len(train_times) + len(valid_times) + len(test_times)
    time_interval = train_times[1] - train_times[0] if len(train_times) > 1 else 1
    
    entity_freq = count_entity_frequencies(data.train, num_rels)
    longtail_entities = identify_longtail_entities(entity_freq, args.longtail_threshold, num_nodes)
    
    lt_stats = {
        'num_lt_entities': len(longtail_entities),
        'num_entities': num_nodes,
        'lt_percentage': 100.0 * len(longtail_entities) / num_nodes
    }
    print(f"\nLT entities: {len(longtail_entities)} ({lt_stats['lt_percentage']:.2f}%)")
    
    all_ans_valid = load_all_answers_for_time_filter(data.valid, num_rels, num_nodes)
    all_ans_test = load_all_answers_for_time_filter(data.test, num_rels, num_nodes)
    
    # Model checkpoint
    model_name = f"simple_decoder_{args.dataset}-ly2-his12-dp0.2_0.2_0.2_0.2-gpu{args.gpu}-decoder_only"
    model_path = os.path.join('./models/', model_name)
    print(f"\nModel: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found: {model_path}")
        return None, None
    
    # Create model
    device = torch.device('cpu')
    model = RecurrentRGCN(num_nodes, num_rels, num_times, time_interval, args.n_hidden,
                          args.input_dropout, args.hidden_dropout, args.feat_dropout, args.layer_norm)
    model.to(device)
    
    # Load checkpoint
    print("\nLoading checkpoint...")
    ckpt = torch.load(model_path, map_location=device)
    state_dict = ckpt['state_dict']
    # Filter compatible keys
    model_dict = model.state_dict()
    compatible = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(compatible)
    model.load_state_dict(model_dict)
    print(f"Loaded {len(compatible)}/{len(state_dict)} parameters from epoch {ckpt.get('epoch', '?')}")
    
    # Create calibrator
    print("\n[INFO] HARD mode calibration (DATA LEAKAGE WARNING)")
    calibrator = HardModeCalibrator(longtail_entities, num_nodes, num_rels)
    calibrator.learn_from_training(data.train, entity_freq)
    
    # STEP 1: Validation
    print("\n" + "=" * 70)
    print("STEP 1: VALIDATION SET")
    print("=" * 70)
    val_results = evaluate_model(model, valid_list, num_rels, num_nodes, all_ans_valid, longtail_entities)
    print_metrics_table("VALIDATION - Overall", val_results['mrr_raw'], val_results['hits_raw'],
                        val_results['mrr'], val_results['hits'], val_results['total_queries'])
    if val_results['mrr_lt'] > 0:
        print_metrics_table("VALIDATION - Long-tail", val_results['mrr_lt_raw'], val_results['hits_lt_raw'],
                            val_results['mrr_lt'], val_results['hits_lt'], val_results['lt_queries'])
    
    # Learn calibration
    val_triples = torch.cat(val_results['all_triples'], dim=0)
    val_scores = torch.cat(val_results['all_scores'], dim=0)
    calibrator.learn_from_validation(val_triples, val_scores, val_triples[:, 2], args.early_stop_ratio)
    grid_info = calibrator.grid_search_info
    
    # STEP 2: Test BEFORE
    print("\n" + "=" * 70)
    print("STEP 2: TEST SET (BEFORE)")
    print("=" * 70)
    test_before = evaluate_model(model, test_list, num_rels, num_nodes, all_ans_test, longtail_entities,
                                  collect_ranks=True)
    print_metrics_table("TEST (BEFORE) - Overall", test_before['mrr_raw'], test_before['hits_raw'],
                        test_before['mrr'], test_before['hits'], test_before['total_queries'])
    if test_before['mrr_lt'] > 0:
        print_metrics_table("TEST (BEFORE) - Long-tail", test_before['mrr_lt_raw'], test_before['hits_lt_raw'],
                            test_before['mrr_lt'], test_before['hits_lt'], test_before['lt_queries'])
    
    # STEP 3: Test AFTER
    print("\n" + "=" * 70)
    print("STEP 3: TEST SET (AFTER - HARD MODE)")
    print("=" * 70)
    test_after = evaluate_model(model, test_list, num_rels, num_nodes, all_ans_test, longtail_entities,
                                 calibrator=calibrator, apply_calibration=True,
                                 track_rank_changes=True, collect_ranks=True)
    print_metrics_table("TEST (AFTER) - Overall", test_after['mrr_raw'], test_after['hits_raw'],
                        test_after['mrr'], test_after['hits'], test_after['total_queries'])
    if test_after['mrr_lt'] > 0:
        print_metrics_table("TEST (AFTER) - Long-tail", test_after['mrr_lt_raw'], test_after['hits_lt_raw'],
                            test_after['mrr_lt'], test_after['hits_lt'], test_after['lt_queries'])
    
    print_comparison_table("COMPARISON: BEFORE vs AFTER", test_before, test_after)
    if test_after['rank_changes']:
        print_rank_change_stats(test_after['rank_changes'], test_after['total_queries'])
    
    # STEP 4: Wilcoxon tests
    print("\n" + "=" * 70)
    print("STEP 4: WILCOXON SIGNED-RANK TESTS")
    print("=" * 70)
    
    wilcoxon_results = []
    
    # Overall Raw
    w_overall_raw = compute_wilcoxon_test(test_before['all_ranks_raw'], test_after['all_ranks_raw'],
                                           "Overall Entity Prediction (Raw Ranks)")
    wilcoxon_results.append(w_overall_raw)
    print_wilcoxon_results(w_overall_raw)
    
    # Overall Filtered
    w_overall_filter = compute_wilcoxon_test(test_before['all_ranks_filter'], test_after['all_ranks_filter'],
                                              "Overall Entity Prediction (Filtered Ranks)")
    wilcoxon_results.append(w_overall_filter)
    print_wilcoxon_results(w_overall_filter)
    
    # Long-tail Raw
    if len(test_before['all_ranks_raw_lt']) > 0:
        w_lt_raw = compute_wilcoxon_test(test_before['all_ranks_raw_lt'], test_after['all_ranks_raw_lt'],
                                          "Long-tail Entity Prediction (Raw Ranks)")
        wilcoxon_results.append(w_lt_raw)
        print_wilcoxon_results(w_lt_raw)
        
        # Long-tail Filtered
        w_lt_filter = compute_wilcoxon_test(test_before['all_ranks_filter_lt'], test_after['all_ranks_filter_lt'],
                                             "Long-tail Entity Prediction (Filtered Ranks)")
        wilcoxon_results.append(w_lt_filter)
        print_wilcoxon_results(w_lt_filter)
    
    # Save results
    calib_params = calibrator.get_params_dict()
    save_results_to_file(args, test_before, test_after, calib_params, lt_stats, wilcoxon_results, grid_info)
    
    return test_before, test_after


# ============================================================================
# PART 11: ARGUMENT PARSER AND MAIN
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='TKG Eval with HARD Mode + Wilcoxon Test')
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID for checkpoint name")
    parser.add_argument("-d", "--dataset", type=str, default='ICEWS14',
                        choices=['ICEWS18', 'ICEWS14', 'ICEWS05-15'])
    parser.add_argument("--longtail-threshold", type=int, default=20)
    parser.add_argument("--early-stop-ratio", type=float, default=0.5,
                        help="Early stop after this fraction without improvement (0<x<=1)")
    parser.add_argument("--n-hidden", type=int, default=200)
    parser.add_argument("--input-dropout", type=float, default=0.2)
    parser.add_argument("--hidden-dropout", type=float, default=0.2)
    parser.add_argument("--feat-dropout", type=float, default=0.2)
    parser.add_argument("--layer-norm", action='store_true', default=True)
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
    print("TKG EVALUATION: HARD MODE + WILCOXON SIGNIFICANCE TEST")
    print("=" * 70)
    print(f"Dataset: {args.dataset}, GPU: {args.gpu}, LT threshold: {args.longtail_threshold}")
    print(f"Early stop ratio: {args.early_stop_ratio}")
    print("=" * 70 + "\n")
    
    run_experiment(args)


if __name__ == '__main__':
    main()