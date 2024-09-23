# lancer
**(WIP)** Codebase for "Language Concept Erasure for Language-invariant Dense Retrieval" EMNLP 2024

### Training
The training of lancer framework needs `msmarco passage ranking` and `mc4` multilingual datasets.
The following command 

```
python3 train_mdpr_lancer_cor.py \
        --seed 42 \
        --job_name mdpr_labse_lancer_cor \
        --base_model_name sentence-transformers/LaBSE \
        --output_dir /path/to/output/mdpr_labse_lancer_cor \
        --langs arabic bangla chinese english finnish french german hindi indonesian japanese korean persian russian spanish telugu thai
        --num_train 3000000 \
        --batch_size 8 \
        --train_n_passages 8 \
        --gradient_accumulation_steps 128 \
        --learning_rate 2e-5 \
        --logging_steps 6400 \
        --query_maxlen 40 \
        --doc_maxlen 180 \
        --num_train_epochs 4 \
        --use_pooler \
        --temperature 10.0 \
        --normalize \
        --fp16
```
