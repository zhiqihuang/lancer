from collections import defaultdict
import os, re, json
from typing import List, Tuple
import pickle
import random
from torch.utils.data import Dataset, DataLoader
import mmap
from tqdm import tqdm
from datasets import load_dataset
HF_TOKEN = "<your_hf_token>"

# reads the number of lines in a file
def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines

def read_queries(path):
    queries = {}
    file_type = path.split('.')[-1]
    assert file_type in ['tsv', 'jsonl'], "unsupported query file formt"

    with open(path, 'r') as f:
        for line in tqdm(f, total=get_num_lines(path), desc='read queries'):
            if file_type == 'tsv':
                data = line.rstrip('\n').split('\t')
                assert len(data) == 2
                qid, qtxt = data
                queries[qid] = qtxt
            else:
                data = json.loads(line.rstrip('\n'))
                qid, qtxt = data["id"], data["question"]
                queries[qid] = qtxt
    return queries

def read_qidpidtriples(file_path):
    qidpidtriples = []
    with open(file_path, 'r') as file:
        for line in tqdm(file, total=get_num_lines(file_path), desc='loading qidpidtriples'):
            line = line.strip()
            qid, pos_pid, neg_pid = line.split('\t')
            qidpidtriples.append((int(qid), int(pos_pid), int(neg_pid)))
    return qidpidtriples

def load_pickle(path):
    data_dict = {}
    with open(path, 'rb') as f:
        data_dict = pickle.load(f)
    return data_dict

class TriplesDataset(Dataset):
    def __init__(self, qidpidtriples, query_dataset, doc_dataset):
        self.query = query_dataset
        self.doc = doc_dataset
        self.qidpidtriples = qidpidtriples
        assert list(self.doc.keys()) == list(self.query.keys())
        self.langs = sorted(list(self.doc.keys()))
        
    def __len__(self):
        return len(self.qidpidtriples)

    def __getitem__(self, idx):
        qtxt, pos_dtxt, neg_dtxt = [], [], []
        qid, pos_pid, neg_pid = self.qidpidtriples[idx]
        for lang in self.langs:
            qtxt.append(self.query[lang][qid])
            pos_dtxt.append(self.doc[lang][pos_pid])
            neg_dtxt.append(self.doc[lang][neg_pid])
        return qtxt, pos_dtxt, neg_dtxt    



class QueryDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        content = self.dataset[idx]
        assert len(content) == 2
        qid, qtxt = content
        return [qid, qtxt]

def format_query(query: str, prefix: str = '') -> str:
    return f'{prefix} {query.strip()}'.strip()

def format_passage(text: str, title: str = '', prefix: str = '') -> str:
    return f'{prefix} {title.strip()} {text.strip()}'.strip()

class TevatronTrainDataset(Dataset):
    def __init__(self, train_n_passages=2, query_prefix='', passage_prefix=''):
        self.train_data = load_dataset("Tevatron/msmarco-passage", split='train')
        self.train_n_passages = train_n_passages
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, item) -> Tuple[str, List[str]]:
        group = self.train_data[item]
        query = group['query']
        group_positives = group['positive_passages']
        group_negatives = group['negative_passages']

        formated_query = [format_query(query, self.query_prefix)]

        pos_psg = random.choice(group_positives)
        
        formated_pos_passages = [format_passage(pos_psg['text'], pos_psg['title'], self.passage_prefix)]

        negative_size = self.train_n_passages-1
        if len(group_negatives) < negative_size:
            negs = random.choices(group_negatives, k=negative_size)
        else:
            _offset = random.randint(0, negative_size)
            negs = [x for x in group_negatives]
            random.shuffle(negs)
            negs = negs * 2
            negs = negs[_offset: _offset + negative_size]

        formated_neg_passages = []
        for neg_psg in negs:
            formated_neg_passages.append(format_passage(neg_psg['text'], neg_psg['title'], self.passage_prefix))

        return formated_query, formated_pos_passages, formated_neg_passages

def get_train_tevatron(args):
    if args.add_prefix:
        train_dataset = TevatronTrainDataset(args.train_n_passages, query_prefix='query:', passage_prefix='passage:')
    else:
        train_dataset = TevatronTrainDataset(args.train_n_passages)
    # # load query and document datasets
    # tevatron_msmarco = load_dataset("Tevatron/msmarco-passage")
    # qidpidtriples = []
    # invalid = 0
    # for data in tqdm(tevatron_msmarco['train'], desc='loading qidpidtriples'):
    #     if len([int(psg['docid']) for psg in data['negative_passages']]) >= args.train_n_passages-1:
    #         qidpidtriples.append([int(data['query_id']), [int(psg['docid']) for psg in data['positive_passages']], [int(psg['docid']) for psg in data['negative_passages']]])
    #     else:
    #         invalid += 1
    # print(f"number of invalid: {invalid}, due to the number of negative passages.")
    # query_dataset, doc_dataset = {}, {}
    # lang = "english"
    # doc_dataset[lang] = load_pickle(args.data_dir + f"/collection-{lang}.pickle")
    # query_dataset[lang] = load_pickle(args.data_dir + f"/queries-{lang}.pickle")
    # # create datasets
    # train_dataset = TevatronoTriplesDataset(qidpidtriples, query_dataset, doc_dataset, args.train_n_passages)
    return train_dataset
