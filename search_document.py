import torch
from transformers import AutoTokenizer, AutoModel
import datasets
import logging
import os
from arguments import get_index_parser
from tqdm import tqdm
import faiss  # type: ignore
from models.dpr import mDPRBase
from util.dataset import read_queries, QueryDataset
from util.util import query_tokenizer, set_seed, test_trec_eval
from collections import defaultdict
import json
import time

from datasets import disable_caching
disable_caching()

logging.basicConfig(level = logging.INFO)
logger = logging.getLogger()

def batch_search(model, tokenizer, args, query_loader, ds, buffer=1000):
    model.eval()
    runs = defaultdict(dict)
    with torch.no_grad():
        for item in tqdm(query_loader, desc=f'batch search ...'):
            qids, query = item
            q_ids, q_mask = query_tokenizer(query, args, tokenizer)
            q_reps = model.query(q_ids, q_mask).detach().cpu().numpy()
            scores, ranklists = ds.get_nearest_examples_batch(args.index_name, q_reps, k=args.topK + buffer)
            for qid, plist, slist in zip(qids, ranklists, scores):
                for pid, sc in zip(plist[args.pid_name], slist):
                    seg = pid[::-1].find('-')
                    docid = pid[::-1][seg+1:][::-1]
                    if docid in runs[qid]:
                        continue
                    else:
                        runs[qid][docid] = sc
                assert len(runs[qid]) >= args.topK, f"increase buffer size to find {args.topK} documents."
    return runs

def main(args):
    set_seed(args.seed)
    args.rank = 0 # single gpu, set rank to 0
    args.device = torch.cuda.current_device()
    os.makedirs(args.output_dir, exist_ok=True)

    args.num_langs = len(args.langs)

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model_name)
    except:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model_name, from_slow=True)

    if args.use_pooler:
        base_encoder = AutoModel.from_pretrained(args.base_model_name, add_pooling_layer=True)
    else:
        base_encoder = AutoModel.from_pretrained(args.base_model_name, add_pooling_layer=False)

    model = mDPRBase(base_encoder, args)
    model.to(args.device)

    # load checkpoint
    if args.checkpoint is not None:
        model.load(args.checkpoint)
    logger.info("model loaded")
    
    # read collection
    ds = datasets.load_from_disk(args.collection)
    args.pid_name = ds.column_names[0]
    logger.info("dataset loaded")

    # load faiss index
    logger.info("loading faiss index ...")
    ds.load_faiss_index(args.index_name, args.faiss_index, device=args.device)
    # make sure the metric is correct.
    assert ds.get_index(args.index_name).faiss_index.metric_type == faiss.METRIC_INNER_PRODUCT 

    # read query
    queries = read_queries(args.test_queries)
    query_list = [[qid, qtxt] for qid, qtxt in queries.items()]
    dataset_query = QueryDataset(query_list)
    query_loader = torch.utils.data.DataLoader(
            dataset_query,
            batch_size=args.batch_size,
            drop_last=False, shuffle=False)
    # encode query
    start_time = time.time()
    runs = batch_search(model, tokenizer, args, query_loader, ds)
    logger.info(f"batch search finished, {round(time.time() - start_time, 3) / len(query_list)} sec/query.")

    # write to file
    runf = os.path.join(args.output_dir, f'test.run')
    with open(runf, 'wt') as runfile:
        for qid in runs:
            scores = list(sorted(runs[qid].items(), key=lambda x: (x[1], x[0]), reverse=True))[:args.topK]
            for i, (did, score) in enumerate(scores):
                runfile.write(f'{qid} 0 {did} {i+1} {score} run\n')
    
    if args.test_qrel:
        # evaludation
        trec_out = test_trec_eval(args.test_qrel, runf, args.metrics, args.trec_eval)
        
        # write trec_eval output into a file
        trec_eval_outfile = os.path.join(args.output_dir, f'test.trec_eval')
        trec_file = open(trec_eval_outfile, 'w')
        for line in trec_out:
            trec_file.write(line + '\n')
        json.dump(vars(args), trec_file)
        trec_file.close()
    
    logger.info("done!")

if __name__ == '__main__':
    parser = get_index_parser()
    args = parser.parse_args()
    main(args)