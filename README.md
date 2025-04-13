# IPA Tokenizer

## Installation and Setup

To download the mapping file, run the following command in your terminal:

```bash
gdown https://drive.google.com/uc?id=1XzLzIvsPZ6wvCOHj_uhnwVJD4rIVnm_T

```

To test the tokenizer

Example:
```bash
python train.py --task sst2 --tokenizer bpe --model transformer --vocab_file thu.csv --max_len 64 --batch_size 16 --epochs 3 --lr 0.0005 --patience 5
