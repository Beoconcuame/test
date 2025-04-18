import torch
from torch.utils.data import Dataset
import os
import logging
from datasets import load_dataset, disable_caching
import numpy as np
disable_caching()
from tokenizer.unigramtokenizer import UnigramTokenizer
from tokenizer.bpetokenizer import BPETokenizer
from tokenizer.wordpiecetokenizer import WordPieceTokenizer
from tokenizer.ipatokenizer2 import IpaTokenizer
from tokenizer.chartokenizer import CharTokenizer



logger = logging.getLogger(__name__)

class GLUEDataset(Dataset):
    """PyTorch Dataset for GLUE tasks supporting custom tokenizers."""

    def __init__(self, split: str, tokenizer, max_len: int, task: str, dataset_path: str | None = None):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.task = task
        self.split = split

        self.is_unigram_tok = isinstance(tokenizer, UnigramTokenizer)
        self.is_bpe_tok = isinstance(tokenizer, BPETokenizer)
        self.is_wordpiece_tok = isinstance(tokenizer, WordPieceTokenizer)
        self.is_ipa_tok = IpaTokenizer is not None and isinstance(tokenizer, IpaTokenizer)
        self.is_char_tok = CharTokenizer is not None and isinstance(tokenizer, CharTokenizer)

        logger.info(
            f"Dataset init ▸ task={task} | split={split} | max_len={max_len} | tok={type(tokenizer).__name__}"
        )

        try:
            actual_split = split
            self.dataset = load_dataset(
                "glue",
                task,
                split=actual_split,
                cache_dir=dataset_path if dataset_path and os.path.isdir(dataset_path) else None,
            )
            logger.info(f"Loaded {len(self.dataset)} samples for {task}/{actual_split}")
        except Exception as e:
            logger.error(f"Failed to load GLUE set ({task}/{split}): {e}", exc_info=True)
            raise

        self.text_keys = self._get_text_keys(task)
        self.label_key = "label"
        if task == "stsb":
            self.num_classes = 1
        else:
            self.num_classes = (
                self.dataset.features[self.label_key].num_classes if self.label_key in self.dataset.features else 2
            )

    @staticmethod
    def _get_text_keys(task: str) -> list[str]:
        task2keys = {
            "cola": ["sentence"],
            "sst2": ["sentence"],
            "mrpc": ["sentence1", "sentence2"],
            "qqp": ["question1", "question2"],
            "stsb": ["sentence1", "sentence2"],
            "mnli": ["premise", "hypothesis"],
            "qnli": ["question", "sentence"],
            "rte": ["sentence1", "sentence2"],
            "wnli": ["sentence1", "sentence2"],
        }
        if task not in task2keys:
            raise ValueError(f"Unknown GLUE task: {task}")
        return task2keys[task]

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        sample = self.dataset[idx]
        text = sample[self.text_keys[0]]
        text_pair = sample[self.text_keys[1]] if len(self.text_keys) > 1 else None
        combined = text if text_pair is None else f"{text} {text_pair}"

        if hasattr(self.tokenizer, "tokenize_text"):
            ids = self.tokenizer.tokenize_text(combined, seq_length=self.max_len)
        else:
            raise TypeError(f"Tokenizer {type(self.tokenizer).__name__} lacks `tokenize_text`.")

        pad_id = getattr(self.tokenizer, "pad_id", 0)
        ids = ids + [pad_id] * (self.max_len - len(ids)) if len(ids) < self.max_len else ids[: self.max_len]

        input_ids = torch.tensor(ids, dtype=torch.long)
        attention_mask = torch.tensor([1 if t != pad_id else 0 for t in ids], dtype=torch.long)

        if self.is_ipa_tok and hasattr(self.tokenizer, "customize_positions"):
            pos = self.tokenizer.customize_positions(combined, seq_length=self.max_len)
            pos = pos + [getattr(self.tokenizer, "pad_pos_id", 0)] * (self.max_len - len(pos)) if len(pos) < self.max_len else pos[: self.max_len]
            aux = torch.tensor(pos, dtype=torch.long)
        else:
            aux = attention_mask

        if self.task == "stsb":
            label_val = sample.get(self.label_key, float("nan"))
            label = torch.tensor(label_val, dtype=torch.float)
        else:
            label_val = sample.get(self.label_key, -1)
            label = torch.tensor(int(label_val), dtype=torch.long)

        return input_ids, aux, label
