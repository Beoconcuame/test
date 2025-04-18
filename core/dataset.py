import torch
from torch.utils.data import Dataset
import os
import logging
from datasets import load_dataset, disable_caching

from tokenizer.unigramtokenizer import UnigramTokenizer
from tokenizer.bpetokenizer import BPETokenizer
from tokenizer.wordpiecetokenizer import WordPieceTokenizer
from tokenizer.ipatokenizer2 import IpaTokenizer
from tokenizer.chartokenizer import CharTokenizer
from tokenizer.bytetokenizer import ByteTokenizer

disable_caching()
logger = logging.getLogger(__name__)

class GLUEDataset(Dataset):
    """PyTorch Dataset for GLUE tasks supporting only custom tokenizers."""

    _ALLOWED_TOKENIZERS = (
        UnigramTokenizer,
        BPETokenizer,
        WordPieceTokenizer,
        IpaTokenizer,
        CharTokenizer,
        ByteTokenizer,
    )

    def __init__(self, split: str, tokenizer, max_len: int, task: str, dataset_path: str | None = None):
        if not isinstance(tokenizer, self._ALLOWED_TOKENIZERS):
            raise TypeError(
                f"Tokenizer phải là instance của một trong các lớp {', '.join(c.__name__ for c in self._ALLOWED_TOKENIZERS)}; "
                f"nhưng nhận được {type(tokenizer).__name__}"
            )
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.task = task
        self.split = split

        self.is_ipa_tok = isinstance(tokenizer, IpaTokenizer)

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
        self.num_classes = 1 if task == "stsb" else (
            self.dataset.features[self.label_key].num_classes
            if self.label_key in self.dataset.features else 2
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

        ids = self.tokenizer.tokenize_text(combined, seq_length=self.max_len)

        pad_id = getattr(self.tokenizer, "pad_id", 0)
        if len(ids) < self.max_len:
            ids = ids + [pad_id] * (self.max_len - len(ids))
        else:
            ids = ids[: self.max_len]

        input_ids = torch.tensor(ids, dtype=torch.long)
        attention_mask = torch.tensor([1 if t != pad_id else 0 for t in ids], dtype=torch.long)

        if self.is_ipa_tok:
            pos = self.tokenizer.customize_positions(combined, seq_length=self.max_len)
            pad_pos_id = getattr(self.tokenizer, "pad_pos_id", 0)
            if len(pos) < self.max_len:
                pos = pos + [pad_pos_id] * (self.max_len - len(pos))
            else:
                pos = pos[: self.max_len]
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
