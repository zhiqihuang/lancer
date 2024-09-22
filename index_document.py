import torch
from transformers import AutoTokenizer, AutoModel
import logging
import os
import faiss # type: ignore
from arguments import get_index_parser
import pickle
from models.dpr import mDPRBase
from datasets import load_dataset
from util.util import set_seed

from datasets import disable_caching
disable_caching()

logging.basicConfig(level = logging.INFO)
logger = logging.getLogger()

def doc_to_psg_tokenizer(examples, tokenizer, psg_maxlen, stride):
    passage_data = []
    pids = examples["pid"]
    docs = examples["text"]
    assert tokenizer.is_fast, "tokenizer must be fast tokenizer"
    for pid, dtxt in zip(pids, docs):
        toks = tokenizer(dtxt, 
            padding="max_length", 
            return_tensors="pt", 
            max_length=psg_maxlen,
            truncation=True, 
            return_overflowing_tokens=True, 
            stride=stride)
        for seg, (ids, mask) in enumerate(zip(toks["input_ids"], toks["attention_mask"])):
            passage_data.append({"seg_id":pid + f"-{seg}", "input_ids": ids, "attention_mask":mask})
    return {"segments": passage_data}

def read_collection(args):
    # docs = {"pid":[], "text":[]}
    # with open(args.collection, "r") as f:
    #     for line in tqdm(f, total=get_num_lines(args.collection), desc="read docs"):
    #         data = line.rstrip("\n").split("\t")
    #         assert len(data) >= 2, data
    #         docid, doctxt = data[:2]
    #         docs["pid"].append(docid)
    #         docs["text"].append(doctxt)
    # ds = datasets.Dataset.from_dict(docs)
    # return ds
    ds = load_dataset("csv", delimiter="\t", header=None, names=['pid', 'text'], usecols=[0, 1], data_files=args.collection)
    return ds['train']

def save_vectors(ds, args, buffer_size=500000):
    n = len(ds)
    for i in range(0, n, buffer_size):
        docids = ds[i:i+buffer_size]["seg_id"]
        vectors = ds[i:i+buffer_size][args.index_name]
        with open(os.path.join(args.output_dir, f"{args.index_name}-{i//buffer_size}.id"), "wb") as f:
            pickle.dump(docids, f)
        with open(os.path.join(args.output_dir, f"{args.index_name}-{i//buffer_size}.vec"), "wb") as f:
            pickle.dump(vectors, f)

def indexing(model, args, ds):
    model.eval()
    with torch.no_grad():
        def encode(examples):
            d_ids = torch.tensor(examples["input_ids"], dtype=torch.int64).to(args.device)
            d_mask = torch.tensor(examples["attention_mask"], dtype=torch.int64).to(args.device)
            npys = model.doc(d_ids, d_mask).cpu().numpy()
            return {args.index_name: npys}

        ds_with_embeddings = ds.map(encode, batched=True, batch_size=args.batch_size, remove_columns=["attention_mask", "input_ids"])

        logger.info("build index ...")
        ds_with_embeddings.add_faiss_index(column=args.index_name, metric_type=faiss.METRIC_INNER_PRODUCT) # !important metric_type=faiss.METRIC_INNER_PRODUCT
        
        logger.info("save index ...")
        ds_with_embeddings.save_faiss_index(args.index_name, os.path.join(args.output_dir, f"{args.index_name}.faiss"))
        
        if args.save_vectors:
            logger.info("save vectors ...")
            save_vectors(ds_with_embeddings, args)

        logger.info("save passage ids to huggingface dataset ...")
        hf_ds = os.path.join(args.output_dir, "hf_ds")
        os.makedirs(hf_ds, exist_ok=True)
        ds_with_embeddings.drop_index(args.index_name)
        removed_columns = [col for col in ds_with_embeddings.column_names if col != "seg_id"]
        ds_ids = ds_with_embeddings.remove_columns(removed_columns) # only keep "seg_id" column
        ds_ids.save_to_disk(hf_ds)

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
    ds = read_collection(args)
    logger.info("dataset loaded")

    # tokenize+split documents into passages
    logger.info("split document into passages ...")
    fn_kwargs={"psg_maxlen": args.doc_maxlen, "stride":args.stride, "tokenizer":tokenizer}
    psg_ds = ds.map(doc_to_psg_tokenizer, fn_kwargs=fn_kwargs, batched=True, remove_columns=ds.column_names, batch_size=args.batch_size)
    psg_ds = psg_ds.flatten()
    psg_ds = psg_ds.rename_column("segments.seg_id", "seg_id")
    psg_ds = psg_ds.rename_column("segments.input_ids", "input_ids")
    psg_ds = psg_ds.rename_column("segments.attention_mask", "attention_mask")
    
    # indexing
    logger.info("begin indexing ...")
    indexing(model, args, psg_ds)

    logger.info("done!")

if __name__ == "__main__":
    parser = get_index_parser()
    args = parser.parse_args()
    main(args)