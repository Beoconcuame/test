import os
import sys
import logging
from typing import List, Tuple, Dict, Any

from datasets import load_dataset
from tokenizers import Tokenizer, normalizers
from tokenizers.normalizers import NFD, StripAccents, Lowercase
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import BertProcessing
from tokenizers.models import BPE, Unigram, WordPiece
from tokenizers.trainers import BpeTrainer, UnigramTrainer, WordPieceTrainer

from tokenizer.ipatokenizer2 import IpaTokenizer
from tokenizer.chartokenizer import CharTokenizer
from tokenizer.bytetokenizer import ByteTokenizer

logger = logging.getLogger(__name__)

_SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<CLS>", "<SEP>", ]
_PAD, _UNK, _CLS, _SEP = _SPECIAL_TOKENS



def _extract_sentences(task: str, dataset_path: str, split: str = "train") -> List[str]:
    ds = load_dataset("glue", task, split=split, cache_dir=dataset_path if os.path.isdir(dataset_path) else None)
    cols: Dict[str, Any] = ds.features
    if "sentence" in ds.column_names:
        return ds["sentence"]
    if {"question", "sentence"}.issubset(ds.column_names):
        return ds["question"] + ds["sentence"]
    if {"sentence1", "sentence2"}.issubset(ds.column_names):
        return ds["sentence1"] + ds["sentence2"]
    if {"premise", "hypothesis"}.issubset(ds.column_names):
        return ds["premise"] + ds["hypothesis"]

    # Fallback: concatenate all string columns
    str_cols = [c for c, t in cols.items() if getattr(t, "dtype", None) == "string"]
    logger.warning("No standard text columns; using %s", str_cols)
    sents: List[str] = []
    for c in str_cols:
        sents.extend([s for s in ds[c] if isinstance(s, str) and s])
    if not sents:
        raise ValueError(f"No sentences extracted for task {task}")
    return sents


def _train_hf_tokenizer(name: str, sentences: List[str], vocab_size: int, max_len: int) -> Tokenizer:
    models = {
        "bpe": (BPE(unk_token=_UNK), BpeTrainer),
        "unigram": (Unigram(), UnigramTrainer),
        "wordpiece": (WordPiece(unk_token=_UNK), WordPieceTrainer),
    }
    model, trainer_cls = models[name]
    trainer = trainer_cls(vocab_size=vocab_size, special_tokens=_SPECIAL_TOKENS, unk_token=_UNK if name == "unigram" else None)
    tok = Tokenizer(model)
    tok.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])
    tok.pre_tokenizer = Whitespace()
    tok.train_from_iterator(sentences, trainer=trainer)

    # Padding / truncation / postproc
    pad_id = tok.token_to_id(_PAD)
    tok.enable_padding(pad_id=pad_id, pad_token=_PAD, length=max_len, direction="right")
    tok.enable_truncation(max_length=max_len)
    cls_id, sep_id = tok.token_to_id(_CLS), tok.token_to_id(_SEP)
    if cls_id is not None and sep_id is not None:
        tok.post_processor = BertProcessing(sep=(_SEP, sep_id), cls=(_CLS, cls_id))
    return tok



def create_tokenizer(cfg: Dict[str, Any]) -> Tuple[Any, int]:
    name = cfg["tokenizer"]
    task = cfg["task"]
    dataset_path = cfg["dataset_path"]
    max_len = cfg["max_len"]
    vocab_size_cfg = cfg.get("vocab_size", 30_000)
    run_mode = cfg.get("run_mode", "train")
    model_dir = cfg.get("tokenizer_model_dir", "trained_tokenizers")
    os.makedirs(model_dir, exist_ok=True)

    logger.info("Preparing tokenizer '%s'", name)
    if name == "ipa":
        vocab_file = cfg.get("vocab_file")
        if not vocab_file or not os.path.exists(vocab_file):
            logger.critical("IPA vocabulary file '%s' not found.", vocab_file)
            sys.exit(1)
        tok = IpaTokenizer(vocab_file, empty_token="<EMPTY>", unknown_token="<UNK>")
        size = len(getattr(tok, "token2idx", getattr(tok, "vocab", {})))
        return tok, size


    if name == "char":
        tok = CharTokenizer()
        return tok, tok.get_vocab_size()
    if name == "byte":
        tok = ByteTokenizer()
        return tok, tok.get_vocab_size()


    if name not in {"bpe", "unigram", "wordpiece"}:
        logger.critical("Unknown tokenizer '%s'", name)
        sys.exit(1)

    model_path = os.path.join(model_dir, f"{name}_{task}_{vocab_size_cfg}.json")

    if os.path.exists(model_path):
        tok = Tokenizer.from_file(model_path)
        # Ensure pad/trunc settings match current max_len
        pad_id = tok.token_to_id(_PAD)
        if pad_id is not None:
            tok.enable_padding(pad_id=pad_id, pad_token=_PAD, length=max_len, direction="right")
        if tok.truncation is None or tok.truncation.get("max_length") != max_len:
            tok.enable_truncation(max_length=max_len)
        return tok, tok.get_vocab_size()

    if run_mode != "train":
        logger.critical("Tokenizer file %s not found and run_mode!='train'", model_path)
        sys.exit(1)

    logger.info("Training new %s tokenizer (vocab=%d)...", name, vocab_size_cfg)
    sentences = _extract_sentences(task, dataset_path)
    tok = _train_hf_tokenizer(name, sentences, vocab_size_cfg, max_len)
    try:
        tok.save(model_path)
        logger.info("Saved tokenizer to %s", model_path)
    except Exception as e:
        logger.error("Cannot save tokenizer: %s", e)
    return tok, tok.get_vocab_size()
