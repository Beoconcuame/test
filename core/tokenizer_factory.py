import os
import sys
import logging
from typing import List, Dict, Any, Tuple

from datasets import load_dataset

from tokenizer.unigramtokenizer import UnigramTokenizer
from tokenizer.bpetokenizer import BPETokenizer
from tokenizer.wordpiecetokenizer import WordPieceTokenizer
from tokenizer.ipatokenizer2 import IpaTokenizer
from tokenizer.chartokenizer import CharTokenizer
from tokenizer.bytetokenizer import ByteTokenizer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

_SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<CLS>", "<SEP>"]


def _extract_sentences(task: str, dataset_path: str, split: str = "train") -> List[str]:
    ds = load_dataset(
        "glue",
        task,
        split=split,
        cache_dir=dataset_path if os.path.isdir(dataset_path) else None
    )
    if "sentence" in ds.column_names:
        return [s for s in ds["sentence"] if isinstance(s, str) and s]
    if {"question", "sentence"}.issubset(ds.column_names):
        return [q + " " + s for q, s in zip(ds["question"], ds["sentence"]) if isinstance(q, str) and isinstance(s, str)]
    if {"sentence1", "sentence2"}.issubset(ds.column_names):
        return [s1 + " " + s2 for s1, s2 in zip(ds["sentence1"], ds["sentence2"]) if isinstance(s1, str) and isinstance(s2, str)]
    if {"premise", "hypothesis"}.issubset(ds.column_names):
        return [p + " " + h for p, h in zip(ds["premise"], ds["hypothesis"]) if isinstance(p, str) and isinstance(h, str)]
    cols = ds.features
    str_cols = [c for c, t in cols.items() if getattr(t, "dtype", None) == "string"]
    logger.warning("No standard text columns; using fallback cols %s", str_cols)
    sents: List[str] = []
    for c in str_cols:
        sents.extend([s for s in ds[c] if isinstance(s, str) and s])
    if not sents:
        raise ValueError(f"No sentences extracted for task {task}")
    return sents


def create_tokenizer(cfg: Dict[str, Any]) -> Tuple[Any, int]:
    name         = cfg["tokenizer"]
    task         = cfg["task"]
    dataset_path = cfg["dataset_path"]
    max_len      = cfg["max_len"]
    vocab_size   = cfg.get("vocab_size", 30000)
    run_mode     = cfg.get("run_mode", "train")
    model_dir    = cfg.get("tokenizer_model_dir", "trained_tokenizers")
    os.makedirs(model_dir, exist_ok=True)

    logger.info("Preparing tokenizer '%s' for task '%s'", name, task)
    model_path = os.path.join(model_dir, f"{name}_{task}_{vocab_size}.model")

    if name == "ipa":
        vf = cfg.get("vocab_file")
        if not vf or not os.path.exists(vf):
            logger.critical("IPA vocab file '%s' not found.", vf)
            sys.exit(1)
        tok = IpaTokenizer(vf, empty_token="<EMPTY>", unknown_token="<UNK>")
        return tok, tok.get_vocab_size()

    if name == "char":
        tok = CharTokenizer()
        return tok, tok.get_vocab_size()

    if name == "byte":
        tok = ByteTokenizer()
        return tok, tok.get_vocab_size()

    if name in {"bpe", "unigram", "wordpiece"}:
        TokClass = {"bpe": BPETokenizer, "unigram": UnigramTokenizer, "wordpiece": WordPieceTokenizer}[name]
        if os.path.exists(model_path):
            tok = TokClass.load(model_path)
            logger.info("Loaded existing %s tokenizer from %s", name, model_path)
        else:
            if run_mode != "train":
                logger.critical("Tokenizer file %s not found and run_mode!='train'", model_path)
                sys.exit(1)
            sentences = _extract_sentences(task, dataset_path)
            logger.info("Training new %s tokenizer (vocab_size=%d)...", name, vocab_size)
            # BPETokenizer.__init__ expects train_sentences first
            tok = TokClass(sentences, _SPECIAL_TOKENS, vocab_size)
            try:
                tok.save(model_path)
                logger.info("Saved %s tokenizer to %s", name, model_path)
            except Exception as e:
                logger.error("Failed to save tokenizer: %s", e)
        return tok, tok.get_vocab_size()

    logger.critical("Unknown tokenizer '%s'", name)
    sys.exit(1)
