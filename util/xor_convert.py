import logging
import argparse
import json, os
from collections import defaultdict
from tqdm import tqdm
from dataset import get_num_lines

logging.basicConfig(level = logging.INFO)
logger = logging.getLogger()

def read_collection(args):
    docs = {}
    with open(args.collection, "r") as f:
        for line in tqdm(f, total=get_num_lines(args.collection), desc="read docs"):
            data = line.rstrip("\n").split("\t")
            assert len(data) >= 2, data
            docid, doctxt = data[:2]
            docs[docid] = doctxt.strip()
    return docs

def read_queries(path):
    queries = []
    with open(path, 'r') as f:
        for line in tqdm(f, total=get_num_lines(path), desc='read queries'):
            data = json.loads(line.rstrip('\n'))
            queries.append(data)
    return queries


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    
    # read collection
    docs = read_collection(args)
    logger.info("dataset loaded")

    
    ranklists = defaultdict(list)
    with open(args.runf, "r") as f:
        for line in f:
            qid, _, did, rank, _, _ = line.rstrip("\n").split()
            ranklists[qid].append([did, int(rank)])
    
    for qid in ranklists:
        ranklists[qid] = sorted(ranklists[qid], key=lambda x: x[1])
    logger.info("ranking data loaded")

    queries = read_queries(args.test_queries)
    results = []
    for query in queries:
        res = {"id": query["id"], "lang": query["lang"]}
        qid = query["id"]
        ctxs = []
        for did, _ in ranklists[qid]:
            ctxs.append(docs[did])
        res["ctxs"] = ctxs
        results.append(res)
    
    # write to file
    runf = os.path.join(args.output_dir, f"test_xor_format.run")
    with open(runf, "w") as runfile:
        json.dump(results, runfile)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection",default=None,type=str,help="collection in tsv file")
    parser.add_argument("--runf", default=None, type=str, required=True, help="run file")
    parser.add_argument("--output_dir",default=None,type=str,required=True,help="output directory")
    parser.add_argument("--test_queries", default=None, type=str, help="test query file")
    args = parser.parse_args()
    main(args)