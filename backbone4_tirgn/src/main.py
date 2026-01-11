import argparse
import itertools
import os
import sys
import time
import pickle

import dgl
import numpy as np
import torch
from tqdm import tqdm
import random
sys.path.append("..")
from rgcn import utils
from rgcn.utils import build_sub_graph
from src.rrgcn import RecurrentRGCN
from src.hyperparameter_range import *
import torch.nn.modules.rnn
from collections import defaultdict
from rgcn.knowledge_graph import _read_triplets_as_list
import scipy.sparse as sp


def count_entity_frequencies(train_data, num_rels):
    """
    Count entity frequencies in training data, considering inverse relations.
    
    Args:
        train_data: training quadruples (s, r, o, t)
        num_rels: number of relations (before adding inverse)
    
    Returns:
        entity_freq: dictionary mapping entity_id to frequency count
    """
    entity_freq = defaultdict(int)
    
    for quad in train_data:
        s, r, o, t = quad
        # Count original triple
        entity_freq[s] += 1
        entity_freq[o] += 1
        # Count inverse triple (o, r+num_rels, s, t)
        entity_freq[o] += 1
        entity_freq[s] += 1
    
    return entity_freq


def identify_longtail_entities(entity_freq, threshold, num_entities):
    """
    Identify long-tail entities based on frequency threshold.
    
    Args:
        entity_freq: dictionary of entity frequencies
        threshold: frequency threshold for long-tail classification
        num_entities: total number of entities
    
    Returns:
        longtail_entities: set of entity ids that are long-tail
    """
    longtail_entities = set()
    
    for entity_id in range(num_entities):
        # If entity doesn't appear in training or appears <= threshold times
        if entity_freq.get(entity_id, 0) <= threshold:
            longtail_entities.add(entity_id)
    
    return longtail_entities


def test(model, history_list, test_list, num_rels, num_nodes, use_cuda, all_ans_list, all_ans_list_r, 
         model_name, static_graph, time_list, history_time_nogt, mode, longtail_entities):
    """
    Test function with separate overall and long-tail entity evaluation.
    """
    # Lists for overall metrics
    ranks_raw, ranks_filter, mrr_raw_list, mrr_filter_list = [], [], [], []
    brier_scores_ent = []
    ece_scores_ent = []
    
    # Lists for long-tail metrics
    ranks_raw_lt, ranks_filter_lt = [], []
    brier_scores_ent_lt = []
    ece_scores_ent_lt = []
    
    # Set proper device
    if use_cuda:
        device = torch.device('cuda:{}'.format(args.gpu))
    else:
        device = torch.device('cpu')

    idx = 0
    if mode == "test":
        # test mode: load parameter form file
        checkpoint = torch.load(model_name, map_location=device)
        print("Load Model name: {}. Using best epoch : {}".format(model_name, checkpoint['epoch']))
        print("\n"+"-"*10+"start testing"+"-"*10+"\n")
        model.load_state_dict(checkpoint['state_dict'])

    model.eval()
    # do not have inverse relation in test input
    input_list = [snap for snap in history_list[-args.test_history_len:]]

    if args.multi_step:
        all_tail_seq = sp.load_npz(
            '../data/{}/history/tail_history_{}.npz'.format(args.dataset, history_time_nogt))
        all_rel_seq = sp.load_npz(
            '../data/{}/history/rel_history_{}.npz'.format(args.dataset, history_time_nogt))

    for time_idx, test_snap in enumerate(tqdm(test_list)):
        history_glist = [build_sub_graph(num_nodes, num_rels, g, use_cuda, args.gpu) for g in input_list]
        test_triples_input = torch.LongTensor(test_snap).to(device)

        # get history
        histroy_data = test_triples_input
        inverse_histroy_data = histroy_data[:, [2, 1, 0, 3]]
        inverse_histroy_data[:, 1] = inverse_histroy_data[:, 1] + num_rels
        histroy_data = torch.cat([histroy_data, inverse_histroy_data])
        histroy_data = histroy_data.cpu().numpy()
        
        if args.multi_step:
            seq_idx = histroy_data[:, 0] * num_rels * 2 + histroy_data[:, 1]
            tail_seq = torch.Tensor(all_tail_seq[seq_idx].todense())
            one_hot_tail_seq = tail_seq.masked_fill(tail_seq != 0, 1)
            rel_seq_idx = histroy_data[:, 0] * num_nodes + histroy_data[:, 2]
            rel_seq = torch.Tensor(all_rel_seq[rel_seq_idx].todense())
            one_hot_rel_seq = rel_seq.masked_fill(rel_seq != 0, 1)
        else:
            all_tail_seq = sp.load_npz(
                '../data/{}/history/tail_history_{}.npz'.format(args.dataset, time_list[time_idx]))
            seq_idx = histroy_data[:, 0] * num_rels * 2 + histroy_data[:, 1]
            tail_seq = torch.Tensor(all_tail_seq[seq_idx].todense())
            one_hot_tail_seq = tail_seq.masked_fill(tail_seq != 0, 1)
            all_rel_seq = sp.load_npz(
                '../data/{}/history/rel_history_{}.npz'.format(args.dataset, time_list[time_idx]))
            rel_seq_idx = histroy_data[:, 0] * num_nodes + histroy_data[:, 2]
            rel_seq = torch.Tensor(all_rel_seq[rel_seq_idx].todense())
            one_hot_rel_seq = rel_seq.masked_fill(rel_seq != 0, 1)
            
        if use_cuda:
            one_hot_tail_seq = one_hot_tail_seq.to(device)
            one_hot_rel_seq = one_hot_rel_seq.to(device)

        test_triples, final_score, final_r_score = model.predict(history_glist, num_rels, static_graph, 
                                                                  test_triples_input, one_hot_tail_seq, 
                                                                  one_hot_rel_seq, use_cuda)

        # Separate long-tail queries
        # A query is long-tail if either subject or object is long-tail
        longtail_mask = []
        for triple in test_triples.cpu():
            s, r, o, t = triple
            # Check if subject or object is long-tail
            # For original triples (r < num_rels), check s
            # For inverse triples (r >= num_rels), check o
            if r < num_rels:
                is_longtail = (s.item() in longtail_entities) or (o.item() in longtail_entities)
            else:
                is_longtail = (s.item() in longtail_entities) or (o.item() in longtail_entities)
            longtail_mask.append(is_longtail)
        longtail_mask = torch.tensor(longtail_mask)
        
        # Overall entity prediction metrics
        mrr_filter_snap, mrr_snap, rank_raw, rank_filter, brier_ent, ece_ent = utils.get_total_rank(
            test_triples, final_score, all_ans_list[time_idx], eval_bz=1000, rel_predict=0)
        
        # Store overall metrics
        ranks_raw.append(rank_raw)
        ranks_filter.append(rank_filter)
        mrr_raw_list.append(mrr_snap)
        mrr_filter_list.append(mrr_filter_snap)
        brier_scores_ent.append(brier_ent)
        ece_scores_ent.append(ece_ent)
        
        # Long-tail entity prediction metrics (if there are any long-tail queries)
        if longtail_mask.any():
            lt_test_triples = test_triples[longtail_mask]
            lt_final_score = final_score[longtail_mask]
            
            # Create filtered answer dict for long-tail queries
            lt_all_ans = {}
            for i, is_lt in enumerate(longtail_mask):
                if is_lt:
                    triple = test_triples[i].cpu()
                    h, r = triple[0].item(), triple[1].item()
                    if h in all_ans_list[time_idx] and r in all_ans_list[time_idx][h]:
                        if h not in lt_all_ans:
                            lt_all_ans[h] = {}
                        lt_all_ans[h][r] = all_ans_list[time_idx][h][r]
            
            mrr_filter_snap_lt, mrr_snap_lt, rank_raw_lt, rank_filter_lt, brier_ent_lt, ece_ent_lt = utils.get_total_rank(
                lt_test_triples, lt_final_score, lt_all_ans, eval_bz=1000, rel_predict=0)
            
            ranks_raw_lt.append(rank_raw_lt)
            ranks_filter_lt.append(rank_filter_lt)
            brier_scores_ent_lt.append(brier_ent_lt)
            ece_scores_ent_lt.append(ece_ent_lt)

        # reconstruct history graph list
        if args.multi_step:
            if not args.relation_evaluation:    
                predicted_snap = utils.construct_snap(test_triples, num_nodes, num_rels, final_score, args.topk)
            else:
                predicted_snap = utils.construct_snap_r(test_triples, num_nodes, num_rels, final_r_score, args.topk)
            if len(predicted_snap):
                input_list.pop(0)
                input_list.append(predicted_snap)
        else:
            input_list.pop(0)
            input_list.append(test_snap)
        idx += 1
    
    # Calculate overall metrics
    mrr_raw, hit_result_raw = utils.stat_ranks(ranks_raw, "raw_ent")
    mrr_filter, hit_result_filter = utils.stat_ranks(ranks_filter, "filter_ent")
    avg_brier_ent = np.mean(brier_scores_ent)
    avg_ece_ent = np.mean(ece_scores_ent)
    
    # Calculate long-tail metrics (if there are any)
    if ranks_raw_lt:
        mrr_raw_lt, hit_result_raw_lt = utils.stat_ranks(ranks_raw_lt, "raw_ent_longtail")
        mrr_filter_lt, hit_result_filter_lt = utils.stat_ranks(ranks_filter_lt, "filter_ent_longtail")
        avg_brier_ent_lt = np.mean(brier_scores_ent_lt)
        avg_ece_ent_lt = np.mean(ece_scores_ent_lt)
    else:
        mrr_raw_lt, mrr_filter_lt = 0.0, 0.0
        hit_result_raw_lt = hit_result_filter_lt = [torch.tensor(0.0)] * 3
        avg_brier_ent_lt, avg_ece_ent_lt = 0.0, 0.0
    
    # Print results
    print("\n" + "="*70)
    print("ENTITY PREDICTION RESULTS")
    print("="*70)
    
    print("\n>>> Overall Entity Prediction:")
    print("MRR (filtered): {:.6f}".format(mrr_filter))
    hits = [1, 3, 10]
    for hit_i, hit in enumerate(hits):
        print("Hits (filtered) @ {}: {:.6f}".format(hit, hit_result_filter[hit_i].item()))
    print("Brier Score (raw): {:.6f} (range [0, 2], lower is better)".format(avg_brier_ent))
    print("ECE (raw): {:.6f} (range [0, 1], lower is better)".format(avg_ece_ent))
    
    if ranks_raw_lt:
        print("\n>>> Long-tail Entity Prediction:")
        print("MRR (filtered): {:.6f}".format(mrr_filter_lt))
        for hit_i, hit in enumerate(hits):
            print("Hits (filtered) @ {}: {:.6f}".format(hit, hit_result_filter_lt[hit_i].item()))
        print("Brier Score (raw): {:.6f} (range [0, 2], lower is better)".format(avg_brier_ent_lt))
        print("ECE (raw): {:.6f} (range [0, 1], lower is better)".format(avg_ece_ent_lt))
    else:
        print("\n>>> Long-tail Entity Prediction: No long-tail queries found in test set")
    
    print("="*70 + "\n")
    
    return (mrr_filter, hit_result_filter, avg_brier_ent, avg_ece_ent,
            mrr_filter_lt, hit_result_filter_lt, avg_brier_ent_lt, avg_ece_ent_lt)


def run_experiment(args, history_len=None, n_layers=None, dropout=None, n_bases=None, angle=None, history_rate=None):
    # load configuration for grid search the best configuration
    if history_len:
        args.train_history_len = history_len
        args.test_history_len = history_len
    if n_layers:
        args.n_layers = n_layers
    if dropout:
        args.dropout = dropout
    if n_bases:
        args.n_bases = n_bases
    if angle:
        args.angle = angle
    if history_rate:
        args.history_rate = history_rate

    # load graph data
    print("loading graph data")
    data = utils.load_data(args.dataset)
    train_list, train_times = utils.split_by_time(data.train)
    valid_list, valid_times = utils.split_by_time(data.valid)
    test_list, test_times = utils.split_by_time(data.test)

    num_nodes = data.num_nodes
    num_rels = data.num_rels
    
    # Print dataset statistics
    print("\n" + "="*70)
    print("DATASET STATISTICS (Original)")
    print("="*70)
    print("Number of unique entities: {}".format(num_nodes))
    print("Number of unique relations: {}".format(num_rels))
    print("Number of unique timestamps: {}".format(len(train_times) + len(valid_times) + len(test_times)))
    print("Number of events: {} (train: {}, valid: {}, test: {})".format(
        len(data.train) + len(data.valid) + len(data.test),
        len(data.train), len(data.valid), len(data.test)))
    
    print("\n" + "="*70)
    print("DATASET STATISTICS (With Inverse Relations)")
    print("="*70)
    print("Number of unique entities: {} (unchanged)".format(num_nodes))
    print("Number of unique relations: {} (original: {}, inverse: {})".format(num_rels * 2, num_rels, num_rels))
    print("Number of unique timestamps: {} (unchanged)".format(len(train_times) + len(valid_times) + len(test_times)))
    print("Number of events: {} (each original event counted twice)".format(
        2 * (len(data.train) + len(data.valid) + len(data.test))))
    
    # Count entity frequencies in training data
    entity_freq = count_entity_frequencies(data.train, num_rels)
    longtail_entities = identify_longtail_entities(entity_freq, args.longtail_threshold, num_nodes)
    
    print("\n" + "="*70)
    print("LONG-TAIL ENTITY ANALYSIS")
    print("="*70)
    print("Long-tail threshold: {}".format(args.longtail_threshold))
    print("Number of long-tail entities: {} ({:.2f}% of all entities)".format(
        len(longtail_entities), 100.0 * len(longtail_entities) / num_nodes))
    print("Number of regular entities: {} ({:.2f}% of all entities)".format(
        num_nodes - len(longtail_entities), 100.0 * (num_nodes - len(longtail_entities)) / num_nodes))
    print("="*70 + "\n")
    
    if args.dataset == "ICEWS14s":
        num_times = len(train_list) + len(valid_list) + len(test_list) + 1
    else:
        num_times = len(train_list) + len(valid_list) + len(test_list)
    time_interval = train_times[1]-train_times[0]
    print("num_times", num_times, "--------------", time_interval)
    history_val_time_nogt = valid_times[0]
    history_test_time_nogt = test_times[0]
    if args.multi_step:
        print("val only use global history before:", history_val_time_nogt)
        print("test only use global history before:", history_test_time_nogt)

    all_ans_list_test = utils.load_all_answers_for_time_filter(data.test, num_rels, num_nodes, False)
    all_ans_list_r_test = utils.load_all_answers_for_time_filter(data.test, num_rels, num_nodes, True)
    all_ans_list_valid = utils.load_all_answers_for_time_filter(data.valid, num_rels, num_nodes, False)
    all_ans_list_r_valid = utils.load_all_answers_for_time_filter(data.valid, num_rels, num_nodes, True)

    model_name = "gl_rate_{}-{}-{}-{}-ly{}-dilate{}-his{}-weight_{}-discount_{}-angle_{}-dp{}_{}_{}_{}-gpu{}-{}"\
        .format(args.history_rate, args.dataset, args.encoder, args.decoder, args.n_layers, args.dilate_len, args.train_history_len, args.weight, args.discount, args.angle,
                args.dropout, args.input_dropout, args.hidden_dropout, args.feat_dropout, args.gpu, args.save)
    model_state_file = os.path.join('../models/', model_name)
    print("Sanity Check: stat name : {}".format(model_state_file))
    print("Sanity Check: Is cuda available ? {}".format(torch.cuda.is_available()))

    # Only use CUDA if available, otherwise CPU (never MPS)
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    if use_cuda:
        print("Using CUDA on GPU: {}".format(args.gpu))
        device = torch.device('cuda:{}'.format(args.gpu))
    else:
        print("Using CPU")
        args.gpu = -1
        device = torch.device('cpu')

    if args.add_static_graph:
        static_triples = np.array(_read_triplets_as_list("../data/" + args.dataset + "/e-w-graph.txt", {}, {}, load_time=False))
        num_static_rels = len(np.unique(static_triples[:, 1]))
        num_words = len(np.unique(static_triples[:, 2]))
        static_triples[:, 2] = static_triples[:, 2] + num_nodes 
        static_node_id = torch.from_numpy(np.arange(num_words + data.num_nodes)).view(-1, 1).long().to(device)
    else:
        num_static_rels, num_words, static_triples, static_graph = 0, 0, [], None

    # create stat
    model = RecurrentRGCN(args.decoder,
                          args.encoder,
                        num_nodes,
                        num_rels,
                        num_static_rels,
                        num_words,
                        num_times,
                        time_interval,
                        args.n_hidden,
                        args.opn,
                        args.history_rate,
                        sequence_len=args.train_history_len,
                        num_bases=args.n_bases,
                        num_basis=args.n_basis,
                        num_hidden_layers=args.n_layers,
                        dropout=args.dropout,
                        self_loop=args.self_loop,
                        skip_connect=args.skip_connect,
                        layer_norm=args.layer_norm,
                        input_dropout=args.input_dropout,
                        hidden_dropout=args.hidden_dropout,
                        feat_dropout=args.feat_dropout,
                        aggregation=args.aggregation,
                        weight=args.weight,
                        discount=args.discount,
                        angle=args.angle,
                        use_static=args.add_static_graph,
                        entity_prediction=args.entity_prediction,
                        relation_prediction=args.relation_prediction,
                        use_cuda=use_cuda,
                        gpu = args.gpu,
                        analysis=args.run_analysis)

    if use_cuda:
        torch.cuda.set_device(args.gpu)
        model.to(device)

    if args.add_static_graph:
        static_graph = build_sub_graph(len(static_node_id), num_static_rels, static_triples, use_cuda, args.gpu)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    if args.test and os.path.exists(model_state_file):
        results = test(model,
                      train_list+valid_list, 
                      test_list, 
                      num_rels, 
                      num_nodes, 
                      use_cuda, 
                      all_ans_list_test, 
                      all_ans_list_r_test, 
                      model_state_file, 
                      static_graph,
                      test_times,
                      history_test_time_nogt,
                      "test",
                      longtail_entities)
        return results
    elif args.test and not os.path.exists(model_state_file):
        print("--------------{} not exist, Change mode to train and generate stat for testing----------------\n".format(model_state_file))
    else:
        print("----------------------------------------start training----------------------------------------\n")
        best_mrr = 0
        for epoch in range(args.n_epochs):
            model.train()
            losses = []
            losses_e = []
            losses_r = []
            losses_static = []

            idx = [_ for _ in range(len(train_list))]
            random.shuffle(idx)

            for train_sample_num in tqdm(idx):
                if train_sample_num == 0: continue
                output = train_list[train_sample_num:train_sample_num+1]
                if train_sample_num - args.train_history_len<0:
                    input_list = train_list[0: train_sample_num]
                else:
                    input_list = train_list[train_sample_num - args.train_history_len: train_sample_num]

                # generate history graph
                history_glist = [build_sub_graph(num_nodes, num_rels, snap, use_cuda, args.gpu) for snap in input_list]
                output = [torch.from_numpy(_).long().to(device) for _ in output]

                # history load
                histroy_data = output[0]
                inverse_histroy_data = histroy_data[:, [2, 1, 0, 3]]
                inverse_histroy_data[:, 1] = inverse_histroy_data[:, 1] + num_rels
                histroy_data = torch.cat([histroy_data, inverse_histroy_data])
                histroy_data = histroy_data.cpu().numpy()
                # tail
                all_tail_seq = sp.load_npz(
                    '../data/{}/history/tail_history_{}.npz'.format(args.dataset, train_times[train_sample_num]))
                seq_idx = histroy_data[:, 0] * num_rels * 2 + histroy_data[:, 1]
                tail_seq = torch.Tensor(all_tail_seq[seq_idx].todense())
                one_hot_tail_seq = tail_seq.masked_fill(tail_seq != 0, 1)
                # rel
                all_rel_seq = sp.load_npz(
                    '../data/{}/history/rel_history_{}.npz'.format(args.dataset, train_times[train_sample_num]))
                rel_seq_idx = histroy_data[:, 0] * num_nodes + histroy_data[:, 2]
                rel_seq = torch.Tensor(all_rel_seq[rel_seq_idx].todense())
                one_hot_rel_seq = rel_seq.masked_fill(rel_seq != 0, 1)
                if use_cuda:
                    one_hot_tail_seq = one_hot_tail_seq.to(device)
                    one_hot_rel_seq = one_hot_rel_seq.to(device)

                loss_e, loss_r, loss_static = model.get_loss(history_glist, output[0], static_graph, one_hot_tail_seq, one_hot_rel_seq, use_cuda)
                loss = args.task_weight*loss_e + (1-args.task_weight)*loss_r + loss_static

                losses.append(loss.item())
                losses_e.append(loss_e.item())
                losses_r.append(loss_r.item())
                losses_static.append(loss_static.item())

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
                optimizer.step()
                optimizer.zero_grad()

            print("Epoch {:04d} | Ave Loss: {:.4f} | entity-relation-static:{:.4f}-{:.4f}-{:.4f} Best MRR {:.4f} | Model {} "
                  .format(epoch, np.mean(losses), np.mean(losses_e), np.mean(losses_r), np.mean(losses_static), best_mrr, model_name))

            # validation
            if epoch and epoch % args.evaluate_every == 0:
                results = test(model,
                              train_list, 
                              valid_list, 
                              num_rels, 
                              num_nodes, 
                              use_cuda, 
                              all_ans_list_valid, 
                              all_ans_list_r_valid, 
                              model_state_file, 
                              static_graph,
                              valid_times,
                              history_val_time_nogt,
                              mode="train",
                              longtail_entities=longtail_entities)
                mrr_filter = results[0]  # Overall filtered MRR for entity prediction
                
                if not args.relation_evaluation:  # entity prediction evaluation
                    if mrr_filter < best_mrr:
                        if epoch >= args.n_epochs:
                            break
                    else:
                        best_mrr = mrr_filter
                        torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, model_state_file)
        
        results = test(model,
                      train_list+valid_list,
                      test_list, 
                      num_rels, 
                      num_nodes, 
                      use_cuda, 
                      all_ans_list_test, 
                      all_ans_list_r_test, 
                      model_state_file, 
                      static_graph,
                      test_times,
                      history_test_time_nogt,
                      mode="test",
                      longtail_entities=longtail_entities)
    
    return results


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    
    # Disable MPS (Apple Silicon GPU) - only use CUDA or CPU
    import os
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    if hasattr(torch.backends, 'mps'):
        torch.backends.mps.is_available = lambda: False
    
    parser = argparse.ArgumentParser(description='TIRGN')

    parser.add_argument("--gpu", type=int, default=0,
                        help="gpu")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="batch-size")
    parser.add_argument("-d", "--dataset", type=str, default='ICEWS14',
                        help="dataset to use")
    parser.add_argument("--test", action='store_true', default=False,
                        help="load stat from dir and directly test")
    parser.add_argument("--run-analysis", action='store_true', default=False,
                        help="print log info")
    parser.add_argument("--run-statistic", action='store_true', default=False,
                        help="statistic the result")
    parser.add_argument("--multi-step", action='store_true', default=False,
                        help="do multi-steps inference without ground truth")
    parser.add_argument("--topk", type=int, default=50,
                        help="choose top k entities as results when do multi-steps without ground truth")
    parser.add_argument("--add-static-graph",  action='store_true', default=True,
                        help="use the info of static graph")
    parser.add_argument("--add-rel-word", action='store_true', default=False,
                        help="use words in relaitons")
    parser.add_argument("--relation-evaluation", action='store_true', default=False,
                        help="save model accordding to the relation evalution")
    parser.add_argument("--longtail-threshold", type=int, default=20,
                        help="frequency threshold to determine long-tail entities")

    # configuration for encoder RGCN stat
    parser.add_argument("--weight", type=float, default=0.5,
                        help="weight of static constraint")
    parser.add_argument("--task-weight", type=float, default=0.7,
                        help="weight of entity prediction task")
    parser.add_argument("--discount", type=float, default=1,
                        help="discount of weight of static constraint")
    parser.add_argument("--angle", type=int, default=14,
                        help="evolution speed")

    parser.add_argument("--encoder", type=str, default="convgcn",
                        help="method of encoder")
    parser.add_argument("--aggregation", type=str, default="none",
                        help="method of aggregation")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="dropout probability")
    parser.add_argument("--skip-connect", action='store_true', default=False,
                        help="whether to use skip connect in a RGCN Unit")
    parser.add_argument("--n-hidden", type=int, default=200,
                        help="number of hidden units")
    parser.add_argument("--opn", type=str, default="sub",
                        help="opn of compgcn")

    parser.add_argument("--n-bases", type=int, default=100,
                        help="number of weight blocks for each relation")
    parser.add_argument("--n-basis", type=int, default=100,
                        help="number of basis vector for compgcn")
    parser.add_argument("--n-layers", type=int, default=2,
                        help="number of propagation rounds")
    parser.add_argument("--self-loop", action='store_true', default=True,
                        help="perform layer normalization in every layer of gcn ")
    parser.add_argument("--layer-norm", action='store_true', default=True,
                        help="perform layer normalization in every layer of gcn ")
    parser.add_argument("--relation-prediction", action='store_true', default=True,
                        help="add relation prediction loss")
    parser.add_argument("--entity-prediction", action='store_true', default=True,
                        help="add entity prediction loss")
    parser.add_argument("--split_by_relation", action='store_true', default=False,
                        help="do relation prediction")

    # configuration for stat training
    parser.add_argument("--n-epochs", type=int, default=3,
                        help="number of minimum training epochs on each time step")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate")
    parser.add_argument("--grad-norm", type=float, default=1.0,
                        help="norm to clip gradient to")

    # configuration for evaluating
    parser.add_argument("--evaluate-every", type=int, default=1,
                        help="perform evaluation every n epochs")

    # configuration for decoder
    parser.add_argument("--decoder", type=str, default="timeconvtranse",
                        help="method of decoder")
    parser.add_argument("--input-dropout", type=float, default=0.2,
                        help="input dropout for decoder ")
    parser.add_argument("--hidden-dropout", type=float, default=0.2,
                        help="hidden dropout for decoder")
    parser.add_argument("--feat-dropout", type=float, default=0.2,
                        help="feat dropout for decoder")

    # configuration for sequences stat
    parser.add_argument("--train-history-len", type=int, default=9,
                        help="history length")
    parser.add_argument("--test-history-len", type=int, default=9,
                        help="history length for test")
    parser.add_argument("--dilate-len", type=int, default=1,
                        help="dilate history graph")

    # configuration for optimal parameters
    parser.add_argument("--grid-search", action='store_true', default=False,
                        help="perform grid search for best configuration")
    parser.add_argument("-tune", "--tune", type=str, default="history_len,n_layers,dropout,n_bases,angle,history_rate",
                        help="stat to use")
    parser.add_argument("--num-k", type=int, default=500,
                        help="number of triples generated")

    # configuration for global history
    parser.add_argument("--history-rate", type=float, default=0.3,
                        help="history rate")

    parser.add_argument("--save", type=str, default="checkpoint",
                        help="number of save")

    args = parser.parse_args()
    print(args)
    
    if args.grid_search:
        out_log = '../results/{}.{}.gs'.format(args.dataset, args.encoder+"-"+args.decoder+"-"+args.save)
        o_f = open(out_log, 'w')
        print("** Grid Search **")
        o_f.write("** Grid Search **\n")
        hyperparameters = args.tune.split(',')

        if args.tune == '' or len(hyperparameters) < 1:
            print("No hyperparameter specified.")
            sys.exit(0)
        if args.dataset == "ICEWS14s":
            hp_range_ = hp_range
        if args.dataset == "WIKI":
            hp_range_ = hp_range_WIKI
        if args.dataset == "YAGO":
            hp_range_ = hp_range_YAGO
        if args.dataset == "ICEWS18":
            hp_range_ = hp_range_ICEWS18
        if args.dataset == "ICEWS05-15":
            hp_range_ = hp_range_ICEWS05_15
        if args.dataset == "GDELT":
            hp_range_ = hp_range_GDELT
        grid = hp_range_[hyperparameters[0]]
        for hp in hyperparameters[1:]:
            grid = itertools.product(grid, hp_range_[hp])
        hits_at_1s = {}
        hits_at_10s = {}
        mrrs = {}
        grid = list(grid)
        print('* {} hyperparameter combinations to try'.format(len(grid)))
        o_f.write('* {} hyperparameter combinations to try\n'.format(len(grid)))
        o_f.close()

        for i, grid_entry in enumerate(list(grid)):
            o_f = open(out_log, 'a')

            if not (type(grid_entry) is list or type(grid_entry) is list):
                grid_entry = [grid_entry]
            grid_entry = utils.flatten(grid_entry)
            print('\n\n* Hyperparameter Set {}:'.format(i))
            o_f.write('* Hyperparameter Set {}:\n'.format(i))
            signature = ''
            print(grid_entry)
            o_f.write("\t".join([str(_) for _ in grid_entry]) + "\n")
            
            args.test = False
            args.multi_step = False
            results = run_experiment(args, grid_entry[0], grid_entry[1], grid_entry[2], grid_entry[3], grid_entry[4], grid_entry[5])
            (mrr_filter, hit_result_filter, avg_brier_ent, avg_ece_ent,
             mrr_filter_lt, hit_result_filter_lt, avg_brier_ent_lt, avg_ece_ent_lt) = results
            
            hits = [1, 3, 10]
            o_f.write(">>> Overall Entity Prediction:\n")
            o_f.write("MRR (filtered): {:.6f}\n".format(mrr_filter))
            for hit_i, hit in enumerate(hits):
                o_f.write("Hits (filtered) @ {}: {:.6f}\n".format(hit, hit_result_filter[hit_i].item()))
            o_f.write("Brier Score (raw): {:.6f}\n".format(avg_brier_ent))
            o_f.write("ECE (raw): {:.6f}\n".format(avg_ece_ent))
            
            o_f.write("\n>>> Long-tail Entity Prediction:\n")
            if mrr_filter_lt > 0:
                o_f.write("MRR (filtered): {:.6f}\n".format(mrr_filter_lt))
                for hit_i, hit in enumerate(hits):
                    o_f.write("Hits (filtered) @ {}: {:.6f}\n".format(hit, hit_result_filter_lt[hit_i].item()))
                o_f.write("Brier Score (raw): {:.6f}\n".format(avg_brier_ent_lt))
                o_f.write("ECE (raw): {:.6f}\n".format(avg_ece_ent_lt))
            else:
                o_f.write("No long-tail queries found in test set\n")
            
            # No ground truth results
            args.test = True
            args.topk = 0
            args.multi_step = True
            results = run_experiment(args, grid_entry[0], grid_entry[1], grid_entry[2], grid_entry[3], grid_entry[4], grid_entry[5])
            (mrr_filter, hit_result_filter, avg_brier_ent, avg_ece_ent,
             mrr_filter_lt, hit_result_filter_lt, avg_brier_ent_lt, avg_ece_ent_lt) = results
            
            o_f.write("\nNo ground truth result:\n")
            o_f.write(">>> Overall Entity Prediction:\n")
            o_f.write("MRR (filtered): {:.6f}\n".format(mrr_filter))
            for hit_i, hit in enumerate(hits):
                o_f.write("Hits (filtered) @ {}: {:.6f}\n".format(hit, hit_result_filter[hit_i].item()))
            o_f.write("Brier Score (raw): {:.6f}\n".format(avg_brier_ent))
            o_f.write("ECE (raw): {:.6f}\n".format(avg_ece_ent))
            
            o_f.write("\n>>> Long-tail Entity Prediction:\n")
            if mrr_filter_lt > 0:
                o_f.write("MRR (filtered): {:.6f}\n".format(mrr_filter_lt))
                for hit_i, hit in enumerate(hits):
                    o_f.write("Hits (filtered) @ {}: {:.6f}\n".format(hit, hit_result_filter_lt[hit_i].item()))
                o_f.write("Brier Score (raw): {:.6f}\n".format(avg_brier_ent_lt))
                o_f.write("ECE (raw): {:.6f}\n".format(avg_ece_ent_lt))
            else:
                o_f.write("No long-tail queries found in test set\n")
            o_f.close()

    # single run
    else:
        run_experiment(args)
    sys.exit()