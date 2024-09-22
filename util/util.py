#!/usr/bin/env python
# coding: utf-8
import subprocess
import json
from collections import defaultdict
import random
import torch
from torch import nn

import torch.nn.functional as F
import numpy as np
import wandb # type: ignore
import os
from contextlib import contextmanager

import pytrec_eval # type: ignore

def check_cuda():
    is_available = torch.cuda.is_available()
    if is_available:
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        print(f"cuda is available, GPU name {device_name}")
    return is_available

class NullContextManager(object):
    def __init__(self, dummy_resource=None):
        self.dummy_resource = dummy_resource
    def __enter__(self):
        return self.dummy_resource
    def __exit__(self, *args):
        pass

class MixedPrecisionManager():
    def __init__(self, activated):

        self.activated = activated

        if self.activated:
            self.scaler = torch.cuda.amp.GradScaler()

    def context(self):
        return torch.cuda.amp.autocast() if self.activated else NullContextManager()

    def backward(self, loss):
        if self.activated:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def step(self, model, optimizer, grad_clip=True):
        if self.activated:
            self.scaler.unscale_(optimizer)
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            self.scaler.step(optimizer)
            self.scaler.update()
            optimizer.zero_grad()
        else:
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            optimizer.zero_grad()

def setup_wandb(args):
    wandb.login(key = '<wandb_key>')
    if hasattr( args, 'is_master'):
        if args.is_master:
            run = wandb.init(project='m2mdpr', entity='zhiqi', config=args, settings=wandb.Settings(code_dir="."))
            run.name = args.job_name
        else:
            run = None
    else:
        run = wandb.init(project='m2mdpr', entity='zhiqi', config=args, settings=wandb.Settings(code_dir="."))
        run.name = args.job_name
    return run

def set_seed(seed):
    '''
    some cudnn methods can be random even after fixing the seed
    unless you tell it to be deterministic
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def trec_eval(qrelf, runf, metric, trec_eval_f):
    output = subprocess.check_output([trec_eval_f, '-m', metric, qrelf, runf]).decode().rstrip()
    output = output.replace('\t', ' ').split('\n')
    assert len(output) == 1
    return float(output[0].split()[2])

def test_trec_eval(qrelf, runf, metrics, trec_eval_f):
    measure = []
    for m in metrics.split(','):
        measure.append('-m')
        measure.append(m.strip())
    output = subprocess.check_output([trec_eval_f] + measure + [qrelf, runf]).decode().rstrip()
    output = output.split('\n')
    eval_out = []
    for line in output:
        eval_out.append(line)
    return eval_out

def query_tokenizer(qtxt, args, tokenizer, padding='longest', pad_to_max=False):
    toks = tokenizer(
        qtxt,
        padding=padding,
        return_tensors='pt',
        max_length = args.query_maxlen,
        truncation=True
    )
    ids, mask = toks['input_ids'], toks['attention_mask']
    if pad_to_max:
        cur_length = ids.size(-1)
        ids = F.pad(ids, (0, args.doc_maxlen-cur_length), "constant", 0)
        mask = F.pad(mask, (0, args.doc_maxlen-cur_length), "constant", 0)
    return ids.to(args.device), mask.to(args.device)

def doc_tokenizer(dtxt, args, tokenizer, padding='longest'):
    toks = tokenizer(
        dtxt,
        padding=padding,
        return_tensors='pt',
        max_length = args.doc_maxlen,
        truncation=True
    )
    ids, mask = toks['input_ids'], toks['attention_mask']
    return ids.to(args.device), mask.to(args.device)

def get_module(model):
    return model.module if hasattr(model, "module") else model

class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss

    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """
    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum
    

class LaReQaEval(object):
    def __init__(self, args):
        self.lan_list = ["ar", "de", "el", "en", "es", "hi", "ru", "th", "tr", "vi", "zh"]

        self.questions = {}
        for lan in self.lan_list:
            self.questions[lan] = defaultdict(list)

        # key: qid, embeds
        self.candidates = defaultdict(list)

        # memorize the corresponding index of each language in "candidates"
        self.index = {}

        self.xquard_qrels = defaultdict(dict)

        self.xquard_evaluator = None

        self.read_data(args)
    
    def read_data(self, args):
        # load dataset
        dataset_name = "xquad-r"
        # dataset_name = "mlqa-r"

        start_ = 0
        self.candidates["sentences"] = []
        self.candidates["context"] = []
        self.candidates["qid"] = []

        for lan in self.lan_list:
            self.questions[lan]["qid"] = []
            self.questions[lan]["question"] = []
            f = open(f"/gypsum/work1/allan/zhiqihuang/m2mdpr/data/{dataset_name}/{lan}.json")
            data = json.load(f)["data"]
            for entry in data:
                for para in entry["paragraphs"]:
                    n_sents = len(para["sentences"])
                    context = [para["context"]]*n_sents            
                    qid_list = [qs["id"] for qs in para["qas"]]
                    q_list = [qs["question"] for qs in para["qas"]]
                    sent_qid_list = [set() for _ in range(n_sents)]
                    for qs in para["qas"]:
                        a_start = qs["answers"][0]["answer_start"]
                        for i in range(n_sents):
                            if a_start >= para["sentence_breaks"][i][0] and a_start <= para["sentence_breaks"][i][1]:
                                sent_qid_list[i].add(qs["id"])
                                break
                    self.candidates["sentences"].extend(para["sentences"])
                    self.candidates["context"].extend(context)
                    self.candidates["qid"].extend(sent_qid_list)
                    self.questions[lan]["qid"].extend(qid_list)
                    self.questions[lan]["question"].extend(q_list)
            self.index[lan] = [start_, len(self.candidates["sentences"])]
            start_ = len(self.candidates["sentences"])
            f.close()

        self.candidates["did"] = ['' for _ in range(len(self.candidates["qid"]))]
        docid = 0
        for lan, (s_, e_) in self.index.items():
            for i in range(s_, e_):
                self.candidates["did"][i] = f'{lan}-{docid}'
                docid += 1

        if args.add_prefix:
            for lan in self.lan_list:
                self.questions[lan]["question"] = ["query: " + txt for txt in self.questions[lan]["question"]]
            for i, txt in enumerate(self.candidates["sentences"]):
                self.candidates["sentences"][i] = "passage: " + txt

        
        for lan in self.lan_list:
            n_q = len(self.questions[lan]["qid"])
            for i in range(n_q):
                qid = self.questions[lan]['qid'][i]
                for id_set, did in zip(self.candidates["qid"], self.candidates["did"]):
                    if qid in id_set:
                        lan_qid = f"{lan}-{qid}"
                        self.xquard_qrels[lan_qid][did] = int(1)

        self.xquard_evaluator = pytrec_eval.RelevanceEvaluator(self.xquard_qrels, {"map_cut.100", "ndcg_cut.10", "recip_rank"})
        
        for lan in self.lan_list:
            for k, v in self.questions[lan].items():
                if k != "qid":
                    self.questions[lan][k] = np.asarray(v)

        for k, v in self.candidates.items():
            if k != "qid" and k != "did":
                self.candidates[k] = np.asarray(v)

    def encode(self, args, model, tokenizer, fitter=None):
        # reset embeds
        for lan in self.lan_list:
            self.questions[lan]["embeds"] = []
        self.candidates["embeds"] = []

        # encode dataset
        model.eval()
        with torch.no_grad():

            for lan in self.lan_list:
                q_lan = self.questions[lan]["question"]
                for i in range(0, len(q_lan), args.batch_size):
                    toks = tokenizer(q_lan[i:i+args.batch_size].tolist(), padding="longest", return_tensors="pt", max_length=40, truncation=True).to('cuda')
                    if fitter:
                        embeds_ = model.query(toks["input_ids"], toks["attention_mask"], fitter=fitter).detach().cpu().numpy()
                    else:
                        embeds_ = model.query(toks["input_ids"], toks["attention_mask"]).detach().cpu().numpy()
                    self.questions[lan]["embeds"].append(embeds_)

            for i in range(0, len(self.candidates["sentences"]), args.batch_size):
                toks = tokenizer(self.candidates["sentences"][i:i+args.batch_size].tolist(), padding="longest", return_tensors="pt", max_length=180, truncation=True).to('cuda')
                if fitter:
                    embeds_ = model.doc(toks["input_ids"], toks["attention_mask"], fitter=fitter).detach().cpu().numpy()
                else:
                    embeds_ = model.doc(toks["input_ids"], toks["attention_mask"]).detach().cpu().numpy()
                self.candidates["embeds"].append(embeds_)

            for lan in self.lan_list:
                self.questions[lan]["embeds"] = np.concatenate(self.questions[lan]["embeds"])
            self.candidates["embeds"] = np.concatenate(self.candidates["embeds"])
    
    def mlir_eval(self, args, model, tokenizer, fitter=None):
        eval_metrics = {}
        self.encode(args, model, tokenizer, fitter)
        xquad_runs_ranklist = defaultdict(dict)
        for lan in self.lan_list:
            dots = np.matmul(self.questions[lan]["embeds"], np.transpose(self.candidates["embeds"]))
            n_q = len(self.questions[lan]["qid"])
            for i in range(n_q):
                qid = self.questions[lan]["qid"][i]
                lan_qid = f"{lan}-{qid}"
                for j in (-dots[i, :]).argsort()[:100]:
                    sc = dots[i, :][j]
                    did = self.candidates['did'][j]
                    xquad_runs_ranklist[lan_qid][did] = float(sc)
    
        results = self.xquard_evaluator.evaluate(xquad_runs_ranklist)
        for measure in sorted(results[list(results.keys())[0]].keys()):
            eval_metrics[measure] = pytrec_eval.compute_aggregated_measure(measure,[query_measures[measure] for query_measures in results.values()])
        model.train()
        return eval_metrics