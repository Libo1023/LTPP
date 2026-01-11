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
from src.rrgcn import RecurrentRGCN


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
         model_name, mode, longtail_entities):
    """Test function - entity prediction only with long-tail analysis"""
    # Lists for overall metrics
    ranks_raw, ranks_filter, mrr_raw_list, mrr_filter_list = [], [], [], []
    brier_scores_ent = []
    ece_scores_ent = []
    
    # Lists for long-tail metrics
    ranks_raw_lt, ranks_filter_lt = [], []
    brier_scores_ent_lt = []
    ece_scores_ent_lt = []
    
    # Counters for queries
    total_queries_original = 0
    total_queries_with_inverse = 0
    longtail_queries_original = 0
    longtail_queries_with_inverse = 0
    
    # Set proper device
    if use_cuda:
        device = torch.device('cuda:{}'.format(args.gpu))
    else:
        device = torch.device('cpu')

    if mode == "test":
        checkpoint = torch.load(model_name, map_location=device)
        print("Load Model name: {}. Using best epoch : {}".format(model_name, checkpoint['epoch']))
        print("\n"+"-"*10+"start testing"+"-"*10+"\n")
        model.load_state_dict(checkpoint['state_dict'])

    model.eval()

    # Create dummy history (not used since we're not doing message passing)
    dummy_history = []

    for time_idx, test_snap in enumerate(tqdm(test_list)):
        test_triples_input = torch.LongTensor(test_snap).to(device)
        
        # Count original queries
        num_original_queries = len(test_triples_input)
        total_queries_original += num_original_queries

        # Create dummy vocabularies (not used)
        dummy_vocab = torch.ones(len(test_triples_input) * 2, num_nodes).to(device)

        # Predict (model uses random embeddings internally)
        test_triples, final_score, _ = model.predict(
            dummy_history, num_rels, None, test_triples_input, dummy_vocab, dummy_vocab, use_cuda
        )
        
        # Count queries with inverse relations
        total_queries_with_inverse += len(test_triples)

        # Separate long-tail queries
        longtail_mask = []
        longtail_mask_original = []
        
        for i, triple in enumerate(test_triples.cpu()):
            s, r, o, t = triple
            # Check if subject or object is long-tail
            is_longtail = (s.item() in longtail_entities) or (o.item() in longtail_entities)
            longtail_mask.append(is_longtail)
            
            # Track original (non-inverse) queries separately
            if r.item() < num_rels:  # Original relation (not inverse)
                if i < num_original_queries:  # Should match first half
                    if is_longtail:
                        longtail_mask_original.append(True)
                    else:
                        longtail_mask_original.append(False)
        
        longtail_mask = torch.tensor(longtail_mask)
        
        # Count long-tail queries for this batch
        longtail_queries_original += sum(longtail_mask_original)
        longtail_queries_with_inverse += longtail_mask.sum().item()

        # Entity prediction metrics
        mrr_filter_snap, mrr_snap, rank_raw, rank_filter, brier_ent, ece_ent = utils.get_total_rank(
            test_triples, final_score, all_ans_list[time_idx], eval_bz=1000, rel_predict=0
        )

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
                lt_test_triples, lt_final_score, lt_all_ans, eval_bz=1000, rel_predict=0
            )
            
            ranks_raw_lt.append(rank_raw_lt)
            ranks_filter_lt.append(rank_filter_lt)
            brier_scores_ent_lt.append(brier_ent_lt)
            ece_scores_ent_lt.append(ece_ent_lt)

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
    
    # Print query statistics
    print("\n" + "="*70)
    print("TEST QUERY STATISTICS")
    print("="*70)
    print("Total queries (original, before inverse): {}".format(total_queries_original))
    print("Total queries (with inverse relations): {}".format(total_queries_with_inverse))
    print("Long-tail queries (original, before inverse): {} ({:.2f}%)".format(
        longtail_queries_original, 
        100.0 * longtail_queries_original / total_queries_original if total_queries_original > 0 else 0))
    print("Long-tail queries (with inverse relations): {} ({:.2f}%)".format(
        longtail_queries_with_inverse,
        100.0 * longtail_queries_with_inverse / total_queries_with_inverse if total_queries_with_inverse > 0 else 0))
    print("="*70)
    
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
    # Initialize return values
    mrr_filter = None
    hit_result_filter = None
    brier_ent = None
    ece_ent = None
    mrr_filter_lt = None
    hit_result_filter_lt = None
    brier_ent_lt = None
    ece_ent_lt = None

    # Load graph data
    print("loading graph data")
    data = utils.load_data(args.dataset)
    train_list, train_times = utils.split_by_time(data.train)
    valid_list, valid_times = utils.split_by_time(data.valid)
    test_list, test_times = utils.split_by_time(data.test)

    num_nodes = data.num_nodes
    num_rels = data.num_rels
    num_times = len(train_list) + len(valid_list) + len(test_list)
    time_interval = train_times[1] - train_times[0]
    
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
    print("Long-tail threshold: {} (entities appearing <= {} times in training)".format(
        args.longtail_threshold, args.longtail_threshold))
    print("Number of long-tail entities: {} ({:.2f}% of all entities)".format(
        len(longtail_entities), 100.0 * len(longtail_entities) / num_nodes))
    print("Number of regular entities: {} ({:.2f}% of all entities)".format(
        num_nodes - len(longtail_entities), 100.0 * (num_nodes - len(longtail_entities)) / num_nodes))
    print("="*70 + "\n")
    
    print("num_nodes: {}, num_rels: {}, num_times: {}".format(num_nodes, num_rels, num_times))

    # Only entity prediction answers needed
    all_ans_list_test = utils.load_all_answers_for_time_filter(data.test, num_rels, num_nodes, False)
    all_ans_list_valid = utils.load_all_answers_for_time_filter(data.valid, num_rels, num_nodes, False)

    model_name = "simple_decoder_{}-ly{}-his{}-dp{}_{}_{}_{}-gpu{}-{}".format(
        args.dataset, args.n_layers, args.train_history_len, args.dropout, 
        args.input_dropout, args.hidden_dropout, args.feat_dropout, args.gpu, args.save
    )
    model_state_file = os.path.join('../models/', model_name)
    print("Sanity Check: model name: {}".format(model_state_file))
    print("Sanity Check: Is cuda available? {}".format(torch.cuda.is_available()))

    # Only use CUDA if available, otherwise CPU
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    if use_cuda:
        print("Using CUDA on GPU: {}".format(args.gpu))
        device = torch.device('cuda:{}'.format(args.gpu))
    else:
        print("Using CPU")
        args.gpu = -1
        device = torch.device('cpu')

    # Create simplified model (no GNN/RNN, only decoder)
    model = RecurrentRGCN(
        args.decoder,
        args.encoder,
        num_nodes,
        num_rels,
        0,  # num_static_rels (not used)
        0,  # num_words (not used)
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
        use_static=False,  # No static graph
        entity_prediction=True,
        relation_prediction=False,  # No relation prediction
        use_cuda=use_cuda,
        gpu=args.gpu,
        analysis=args.run_analysis
    )

    if use_cuda:
        torch.cuda.set_device(args.gpu)
        model.to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    if args.test and os.path.exists(model_state_file):
        results = test(
            model, train_list+valid_list, test_list, num_rels, num_nodes, 
            use_cuda, all_ans_list_test, model_state_file, "test", longtail_entities
        )
        return results
    elif args.test and not os.path.exists(model_state_file):
        print("--------------{} not exist, Change mode to train----------------\n".format(model_state_file))
    else:
        print("----------------------------------------start training----------------------------------------\n")
        best_mrr = 0
        for epoch in range(args.n_epochs):
            model.train()
            losses = []

            idx = [_ for _ in range(len(train_list))]
            random.shuffle(idx)

            for train_sample_num in tqdm(idx):
                if train_sample_num == 0:
                    continue
                    
                output = train_list[train_sample_num:train_sample_num+1]
                output = [torch.from_numpy(_).long().to(device) for _ in output]

                # Create dummy vocabularies (not used)
                dummy_vocab = torch.ones(len(output[0]) * 2, num_nodes).to(device)

                # Compute loss (model uses random embeddings internally)
                loss_e, _, _ = model.get_loss(None, output[0], None, dummy_vocab, dummy_vocab, use_cuda)

                losses.append(loss_e.item())

                loss_e.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
                optimizer.step()
                optimizer.zero_grad()

            print("Epoch {:04d} | Ave Loss: {:.4f} | Best MRR {:.4f} | Model {}".format(
                epoch, np.mean(losses), best_mrr, model_name
            ))

            # Validation
            if epoch and epoch % args.evaluate_every == 0:
                results = test(
                    model, train_list, valid_list, num_rels, num_nodes,
                    use_cuda, all_ans_list_valid, model_state_file, "train", longtail_entities
                )
                mrr_filter = results[0]
                
                if mrr_filter < best_mrr:
                    if epoch >= args.n_epochs:
                        break
                else:
                    best_mrr = mrr_filter
                    torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, model_state_file)
        
        # Final test
        results = test(
            model, train_list+valid_list, test_list, num_rels, num_nodes,
            use_cuda, all_ans_list_test, model_state_file, "test", longtail_entities
        )
    
    return results


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    
    # Disable MPS
    import os
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    if hasattr(torch.backends, 'mps'):
        torch.backends.mps.is_available = lambda: False
    
    parser = argparse.ArgumentParser(description='Simplified TimeConvTransE Decoder Only')

    parser.add_argument("--gpu", type=int, default=0, help="gpu")
    parser.add_argument("--batch-size", type=int, default=1, help="batch-size")
    parser.add_argument("-d", "--dataset", type=str, default='ICEWS14', help="dataset to use")
    parser.add_argument("--test", action='store_true', default=False, help="load model and test")
    parser.add_argument("--run-analysis", action='store_true', default=False, help="print log info")
    parser.add_argument("--longtail-threshold", type=int, default=20,
                        help="frequency threshold to determine long-tail entities")

    # Decoder configuration
    parser.add_argument("--decoder", type=str, default="timeconvtranse", help="decoder method")
    parser.add_argument("--encoder", type=str, default="convgcn", help="encoder (not used)")
    parser.add_argument("--input-dropout", type=float, default=0.2, help="input dropout for decoder")
    parser.add_argument("--hidden-dropout", type=float, default=0.2, help="hidden dropout for decoder")
    parser.add_argument("--feat-dropout", type=float, default=0.2, help="feat dropout for decoder")

    # Model configuration (many not used but kept for compatibility)
    parser.add_argument("--weight", type=float, default=0.5, help="not used")
    parser.add_argument("--task-weight", type=float, default=0.7, help="not used")
    parser.add_argument("--discount", type=float, default=1, help="not used")
    parser.add_argument("--angle", type=int, default=14, help="not used")
    parser.add_argument("--aggregation", type=str, default="none", help="not used")
    parser.add_argument("--dropout", type=float, default=0.2, help="dropout probability")
    parser.add_argument("--skip-connect", action='store_true', default=False, help="not used")
    parser.add_argument("--n-hidden", type=int, default=200, help="hidden dimension")
    parser.add_argument("--opn", type=str, default="sub", help="not used")
    parser.add_argument("--n-bases", type=int, default=100, help="not used")
    parser.add_argument("--n-basis", type=int, default=100, help="not used")
    parser.add_argument("--n-layers", type=int, default=2, help="not used")
    parser.add_argument("--self-loop", action='store_true', default=True, help="not used")
    parser.add_argument("--layer-norm", action='store_true', default=True, help="perform layer normalization")
    parser.add_argument("--history-rate", type=float, default=0.3, help="not used")

    # Training configuration
    parser.add_argument("--n-epochs", type=int, default=30, help="number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--grad-norm", type=float, default=1.0, help="gradient clipping")
    parser.add_argument("--evaluate-every", type=int, default=1, help="evaluate every n epochs")

    # History configuration (not used)
    parser.add_argument("--train-history-len", type=int, default=9, help="not used")
    parser.add_argument("--test-history-len", type=int, default=9, help="not used")
    parser.add_argument("--dilate-len", type=int, default=1, help="not used")

    parser.add_argument("--save", type=str, default="checkpoint", help="save name")

    args = parser.parse_args()
    print(args)
    
    run_experiment(args)
    sys.exit()