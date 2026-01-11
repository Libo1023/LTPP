import argparse
import os
import sys
import numpy as np
import torch
from tqdm import tqdm
import random
from collections import defaultdict

sys.path.append("..")
from rgcn import utils
from rgcn.utils import build_sub_graph
from src.rrgcn import RecurrentRGCN
from rgcn.knowledge_graph import _read_triplets_as_list


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


def test(model, history_list, test_list, num_rels, num_nodes, use_cuda, all_ans_list, 
         model_name, static_graph, mode, longtail_entities):
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
    if use_cuda and torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(args.gpu))
    else:
        device = torch.device('cpu')

    idx = 0
    if mode == "test":
        # test mode: load parameter from file
        checkpoint = torch.load(model_name, map_location=device)
        print("Load Model name: {}. Using best epoch : {}".format(model_name, checkpoint['epoch']))
        print("\n"+"-"*10+"start testing"+"-"*10+"\n")
        model.load_state_dict(checkpoint['state_dict'])

    model.eval()
    # do not have inverse relation in test input
    input_list = [snap for snap in history_list[-args.test_history_len:]]

    for time_idx, test_snap in enumerate(tqdm(test_list)):
        history_glist = [build_sub_graph(num_nodes, num_rels, g, use_cuda, args.gpu) 
                        for g in input_list]
        test_triples_input = torch.LongTensor(test_snap).to(device)

        # Predict
        test_triples, final_score = model.predict(history_glist, num_rels, static_graph, 
                                                  test_triples_input, use_cuda)

        # Separate long-tail queries
        # A query is long-tail if either subject or object is long-tail
        longtail_mask = []
        for triple in test_triples.cpu():
            s, r, o, t = triple
            # Check if subject or object is long-tail
            # For original triples (r < num_rels), check s
            # For inverse triples (r >= num_rels), check o (which is actually the original subject)
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

        # Reconstruct history graph list
        if args.multi_step:
            predicted_snap = utils.construct_snap(test_triples, num_nodes, num_rels, 
                                                 final_score, args.topk)
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
    print("Brier Score: {:.6f} (range [0, 2], lower is better)".format(avg_brier_ent))
    print("ECE: {:.6f} (range [0, 1], lower is better)".format(avg_ece_ent))
    
    if ranks_raw_lt:
        print("\n>>> Long-tail Entity Prediction:")
        print("MRR (filtered): {:.6f}".format(mrr_filter_lt))
        for hit_i, hit in enumerate(hits):
            print("Hits (filtered) @ {}: {:.6f}".format(hit, hit_result_filter_lt[hit_i].item()))
        print("Brier Score: {:.6f} (range [0, 2], lower is better)".format(avg_brier_ent_lt))
        print("ECE: {:.6f} (range [0, 1], lower is better)".format(avg_ece_ent_lt))
    else:
        print("\n>>> Long-tail Entity Prediction: No long-tail queries found in test set")
    
    print("="*70 + "\n")
    
    return (mrr_filter, hit_result_filter, avg_brier_ent, avg_ece_ent,
            mrr_filter_lt, hit_result_filter_lt, avg_brier_ent_lt, avg_ece_ent_lt)


def run_experiment(args):
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

    all_ans_list_test = utils.load_all_answers_for_time_filter(data.test, num_rels, 
                                                                num_nodes, False)
    all_ans_list_valid = utils.load_all_answers_for_time_filter(data.valid, num_rels, 
                                                                 num_nodes, False)

    model_name = "{}-{}-{}-ly{}-his{}-angle{}-dp{}_{}_{}-gpu{}"\
        .format(args.dataset, args.encoder, args.decoder, args.n_layers, 
                args.train_history_len, args.angle,
                args.dropout, args.input_dropout, args.hidden_dropout, args.gpu)
    model_state_file = os.path.join('../models/', model_name)
    print("Sanity Check: model name : {}".format(model_state_file))
    print("Sanity Check: Is cuda available ? {}".format(torch.cuda.is_available()))

    # Only use CUDA if available, otherwise CPU
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    if use_cuda:
        print("Using CUDA on GPU: {}".format(args.gpu))
        device = torch.device('cuda:{}'.format(args.gpu))
    else:
        print("Using CPU")
        args.gpu = -1
        device = torch.device('cpu')

    # Load static graph if needed
    if args.add_static_graph:
        static_triples = np.array(_read_triplets_as_list(
            "../data/" + args.dataset + "/e-w-graph.txt", {}, {}, load_time=False))
        num_static_rels = len(np.unique(static_triples[:, 1]))
        num_words = len(np.unique(static_triples[:, 2]))
        static_triples[:, 2] = static_triples[:, 2] + num_nodes 
        static_node_id = torch.from_numpy(np.arange(num_words + data.num_nodes)).view(-1, 1).long().to(device)
    else:
        num_static_rels, num_words, static_triples, static_graph = 0, 0, [], None

    # Create model
    model = RecurrentRGCN(args.decoder,
                          args.encoder,
                          num_nodes,
                          num_rels,
                          num_static_rels,
                          num_words,
                          args.n_hidden,
                          args.opn,
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
                          relation_prediction=False,  # RE-GCN focuses on entity prediction
                          use_cuda=use_cuda,
                          gpu=args.gpu,
                          analysis=args.run_analysis)

    if use_cuda:
        torch.cuda.set_device(args.gpu)
        model.to(device)

    if args.add_static_graph:
        static_graph = build_sub_graph(len(static_node_id), num_static_rels, 
                                       static_triples, use_cuda, args.gpu)

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
                      model_state_file, 
                      static_graph,
                      "test",
                      longtail_entities)
    elif args.test and not os.path.exists(model_state_file):
        print("--------------{} not exist, Change mode to train and generate model for testing----------------\n".format(model_state_file))
    else:
        print("----------------------------------------start training----------------------------------------\n")
        best_mrr = 0
        for epoch in range(args.n_epochs):
            model.train()
            losses = []
            losses_e = []
            losses_static = []

            idx = [_ for _ in range(len(train_list))]
            random.shuffle(idx)

            for train_sample_num in tqdm(idx):
                if train_sample_num == 0: 
                    continue
                output = train_list[train_sample_num:train_sample_num+1]
                if train_sample_num - args.train_history_len < 0:
                    input_list = train_list[0: train_sample_num]
                else:
                    input_list = train_list[train_sample_num - args.train_history_len: train_sample_num]

                # Generate history graph
                history_glist = [build_sub_graph(num_nodes, num_rels, snap, use_cuda, args.gpu) 
                               for snap in input_list]
                output = [torch.from_numpy(_).long().to(device) for _ in output]

                # Get loss
                loss_e, loss_r, loss_static = model.get_loss(history_glist, output[0], 
                                                            static_graph, use_cuda)
                loss = loss_e + loss_static  # Only entity prediction loss

                losses.append(loss.item())
                losses_e.append(loss_e.item())
                losses_static.append(loss_static.item())

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
                optimizer.step()
                optimizer.zero_grad()

            print("Epoch {:04d} | Ave Loss: {:.4f} | entity-static:{:.4f}-{:.4f} | Best MRR {:.4f} | Model {} "
                  .format(epoch, np.mean(losses), np.mean(losses_e), 
                         np.mean(losses_static), best_mrr, model_name))

            # validation
            if epoch and epoch % args.evaluate_every == 0:
                results = test(model,
                              train_list, 
                              valid_list, 
                              num_rels, 
                              num_nodes, 
                              use_cuda, 
                              all_ans_list_valid, 
                              model_state_file, 
                              static_graph,
                              mode="train",
                              longtail_entities=longtail_entities)
                mrr_filter = results[0]  # Overall filtered MRR
                
                if mrr_filter < best_mrr:
                    if epoch >= args.n_epochs:
                        break
                else:
                    best_mrr = mrr_filter
                    torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, 
                             model_state_file)
        
        # Final test
        results = test(model,
                      train_list+valid_list,
                      test_list, 
                      num_rels, 
                      num_nodes, 
                      use_cuda, 
                      all_ans_list_test, 
                      model_state_file, 
                      static_graph,
                      mode="test",
                      longtail_entities=longtail_entities)
    
    return results


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    
    parser = argparse.ArgumentParser(description='RE-GCN')

    parser.add_argument("--gpu", type=int, default=0,
                        help="gpu")
    parser.add_argument("-d", "--dataset", type=str, default='ICEWS14s',
                        help="dataset to use")
    parser.add_argument("--test", action='store_true', default=False,
                        help="load model from dir and directly test")
    parser.add_argument("--run-analysis", action='store_true', default=False,
                        help="print log info")
    parser.add_argument("--multi-step", action='store_true', default=False,
                        help="do multi-steps inference without ground truth")
    parser.add_argument("--topk", type=int, default=10,
                        help="choose top k entities as results when do multi-steps without ground truth")
    parser.add_argument("--add-static-graph",  action='store_true', default=True,
                        help="use the info of static graph")
    parser.add_argument("--longtail-threshold", type=int, default=5,
                        help="frequency threshold to determine long-tail entities")

    # configuration for encoder RGCN
    parser.add_argument("--weight", type=float, default=0.5,
                        help="weight of static constraint")
    parser.add_argument("--discount", type=float, default=1,
                        help="discount of weight of static constraint")
    parser.add_argument("--angle", type=int, default=10,
                        help="evolution speed")

    parser.add_argument("--encoder", type=str, default="uvrgcn",
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
                        help="perform self loop in gcn")
    parser.add_argument("--layer-norm", action='store_true', default=False,
                        help="perform layer normalization in every layer of gcn")
    parser.add_argument("--entity-prediction", action='store_true', default=True,
                        help="add entity prediction loss")

    # configuration for training
    parser.add_argument("--n-epochs", type=int, default=500,
                        help="number of minimum training epochs on each time step")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate")
    parser.add_argument("--grad-norm", type=float, default=1.0,
                        help="norm to clip gradient to")

    # configuration for evaluating
    parser.add_argument("--evaluate-every", type=int, default=1,
                        help="perform evaluation every n epochs")

    # configuration for decoder
    parser.add_argument("--decoder", type=str, default="convtranse",
                        help="method of decoder")
    parser.add_argument("--input-dropout", type=float, default=0.2,
                        help="input dropout for decoder")
    parser.add_argument("--hidden-dropout", type=float, default=0.2,
                        help="hidden dropout for decoder")
    parser.add_argument("--feat-dropout", type=float, default=0.2,
                        help="feat dropout for decoder")

    # configuration for sequences
    parser.add_argument("--train-history-len", type=int, default=3,
                        help="history length")
    parser.add_argument("--test-history-len", type=int, default=3,
                        help="history length for test")
    parser.add_argument("--dilate-len", type=int, default=1,
                        help="dilate history graph")

    args = parser.parse_args()
    print(args)
    
    run_experiment(args)
    sys.exit()