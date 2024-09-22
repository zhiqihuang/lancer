#!/usr/bin/env python
# coding: utf-8
import torch
import os
import datasets
from datasets import interleave_datasets
import torch.nn.functional as F
from models.leace import get_score
from models.dpr import mDPRBase
from transformers import AutoTokenizer, AutoModel
from arguments import get_train_parser
from tqdm import tqdm
from util.dataset import get_train_tevatron
from util.util import MixedPrecisionManager, LaReQaEval, doc_tokenizer, set_seed, setup_wandb
os.environ["TOKENIZERS_PARALLELISM"] = "true"

def add_prefix(example, idx):
    if idx % 2:
        example["text"] = 'query: ' + example["text"]
    else:
        example["text"] = 'passage: ' + example["text"]
    return example

def mc4_render(langs, labels, split='train', size=0, text_prefix=False):
    ds_group = []
    for lang in langs:
        ds = datasets.load_from_disk(f"/path_to_c4_ds/{lang}/{split}")
        if size and len(ds) > size:
            ds = ds.select(range(size))
        if labels:
            y = [int(labels[lang])] * len(ds)
            ds = ds.add_column("label", y)
        if text_prefix:
            ds = ds.map(add_prefix, with_indices=True)
        ds_group.append(ds)
    ds = interleave_datasets(ds_group, stopping_strategy="all_exhausted")
    return ds

def mc4_loader(langs, split='train', size=3000000, batch_size=100, return_loader=True, text_prefix=False):
    labels = {lang: i for i, lang in enumerate(sorted(langs))}
    ds = mc4_render(langs, labels, split, size, text_prefix)
    if return_loader:
        ds = torch.utils.data.DataLoader(ds, drop_last=True, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    return ds

def get_ir_datasets(args):
    dataset_train = get_train_tevatron(args)
    train_loader = torch.utils.data.DataLoader(
            dataset_train,
            drop_last=True,
            batch_size=args.batch_size,
            shuffle=True, pin_memory=True)
    return train_loader

def lingustic_recover(model, args, tokenizer, dc_recover_loader, dc_dev_loader):
    model.eval()
    dev_res = {}
    X_train, y_train = [], []
    X_dev, y_dev = [], []
    amp = MixedPrecisionManager(args.fp16)
    with torch.no_grad():
        with amp.context():
            for batch in dc_recover_loader:
                lingual_ids, lingual_mask = doc_tokenizer(batch['text'], args, tokenizer)
                lingual_labels = batch['label'].long().to(args.device)
                features = model.doc(lingual_ids, lingual_mask)
                X_train.append(features)
                y_train.append(lingual_labels)
            X_train = torch.cat(X_train).cpu().numpy()
            y_train = torch.cat(y_train).cpu().numpy().astype(int)
        
            for batch in dc_dev_loader:
                lingual_ids, lingual_mask = doc_tokenizer(batch['text'], args, tokenizer)
                lingual_labels = batch['label'].long().to(args.device)
                features = model.doc(lingual_ids, lingual_mask)
                X_dev.append(features.cpu())
                y_dev.append(lingual_labels.cpu())
            X_dev = torch.cat(X_dev).numpy()
            y_dev = torch.cat(y_dev).numpy().astype(int)
    loss_val, train_score, dev_score = get_score(X_train, y_train, X_dev, y_dev)
    dev_res['recover_loss_val'] = loss_val
    dev_res['recover_train_score'] = train_score
    dev_res['recover_dev_score'] = dev_score
    model.train()
    return dev_res

def train(model, tokenizer, args, ir_train_loader, dc_train_loader, dc_dev_loader, dc_recover_loader, wandb_run=None):
    
    model.train()
    optim = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    amp = MixedPrecisionManager(args.fp16)

    steps = 0
    train_rank_loss = 0.0
    train_dist_loss = 0.0
    best_dev_ndcg = 0.0
    
    dc_train_iter = iter(dc_train_loader)
    lareqa_eval = LaReQaEval(args)
    
    for epoch in range(args.num_train_epochs):
        for item in tqdm(ir_train_loader, desc=f'train at epoch {epoch}'):
            queries, pos_docs, neg_docs = item
            queries, pos_docs, neg_docs = list(map(list, zip(*queries))), list(map(list, zip(*pos_docs))) ,list(map(list, zip(*neg_docs)))
            queries = sum(queries, [])
            pos_docs = sum(pos_docs, [])
            neg_docs = sum(neg_docs, [])
            docs = pos_docs + neg_docs
            
            try:
                batch = next(dc_train_iter)
            except StopIteration:
                dc_train_iter = iter(dc_train_loader)
                batch = next(dc_train_iter)

            lingual_labels = F.one_hot(batch['label'].long().to(args.device), num_classes=args.num_langs)
            input_sections = [len(queries), len(docs), len(batch['text'])]
            combined_text = queries + docs + batch['text']
            input_ids, attention_mask = doc_tokenizer(combined_text, args, tokenizer)
            
            steps += 1

            with amp.context():
                ensemble_features = model.doc(input_ids, attention_mask)
                
                q_reps, d_reps, features = ensemble_features.split(input_sections, dim=0)
                scores = model.score(q_reps, d_reps)
                rank_loss = model.loss_fct(scores, model.labels[:scores.size(0)])
                
                mean_labels = lingual_labels.float().mean(dim=0, keepdim=True)
                mean_features = features.mean(dim=0, keepdim=True)
            
                features_centered = features - mean_features
                labels_centered = lingual_labels - mean_labels

                var_labels = torch.var(lingual_labels.float(), dim=0, keepdim=True)
                var_features = torch.var(features, dim=0, keepdim=True)
            
                cov_matrix = (features_centered.T @ labels_centered) / (features_centered.shape[0] - 1)

                correlation_matrix = cov_matrix / (torch.sqrt(var_features.t() * var_labels) + 1e-5)
                dist_loss = torch.mean(torch.abs(correlation_matrix))

            amp.backward(rank_loss + dist_loss)

            if steps % args.gradient_accumulation_steps == 0:
                amp.step(model, optim)
            
            train_rank_loss += rank_loss.item()
            train_dist_loss += dist_loss.item()
            
            if steps % args.logging_steps == 0 and steps % args.gradient_accumulation_steps == 0:
                train_rank_loss = train_rank_loss / args.logging_steps
                train_dist_loss = train_dist_loss / args.logging_steps
                dev_res = lingustic_recover(model, args, tokenizer, dc_recover_loader, dc_dev_loader)
                mlir_res = lareqa_eval.mlir_eval(args, model, tokenizer)
                if mlir_res['ndcg_cut_10'] > best_dev_ndcg:
                    best_dev_ndcg = mlir_res['ndcg_cut_10']
                    model.save(os.path.join(args.output_dir, 'weights.p'))
                wandb_run.log({
                    'train_rank_loss': train_rank_loss,
                    'train_dist_loss': train_dist_loss,
                    'plain_recover_loss': dev_res['recover_loss_val'],
                    'plain_dev_recover_accuracy': dev_res['recover_dev_score'] * 100,
                    'plain_train_recover_accuracy': dev_res['recover_train_score'] * 100,
                    'mlir_ndcg@10': mlir_res['ndcg_cut_10'],
                })
                train_rank_loss = 0.0
                train_dist_loss = 0.0

        # save checkpoint after training finish:
        model.save(os.path.join(args.output_dir, f'checkpoint_at_{epoch+1}.pth'))

def main(args):
    set_seed(args.seed)
    args.device = torch.cuda.current_device()
    wandb_obj = setup_wandb(args)
    os.makedirs(args.output_dir, exist_ok=True)

    args.num_langs = len(args.langs)
    
    base_encoder = AutoModel.from_pretrained(args.base_model_name, add_pooling_layer=args.use_pooler)

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model_name)
    except:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model_name, from_slow=True)

    assert tokenizer.is_fast
    
    model = mDPRBase(base_encoder, args)
    model.to(args.device)

    # load checkpoint
    if args.checkpoint:
        model.load(args.checkpoint)
    
    # train
    ir_train_loader = get_ir_datasets(args)
    dc_batch_size = 256

    dc_train_loader = mc4_loader(args.langs, split='train', size=args.num_train, batch_size=dc_batch_size, return_loader=True)
    dc_recover_loader = mc4_loader(args.langs, split='train', size=5000, batch_size=args.batch_size*100, return_loader=True, text_prefix=args.add_prefix)
    dc_dev_loader = mc4_loader(args.langs, split='test', size=1000, batch_size=args.batch_size*100, return_loader=True, text_prefix=args.add_prefix)
    
    train(model, tokenizer, args, ir_train_loader, dc_train_loader, dc_dev_loader, dc_recover_loader, wandb_obj)

if __name__ == '__main__':
    parser = get_train_parser()
    args = parser.parse_args()
    main(args)