import torch
from torch.utils.data import Dataset
import os
import sys
import logging
from datasets import load_dataset, disable_caching
import numpy as np

try:
    from tokenizer.ipatokenizer2 import IpaTokenizer
    from tokenizer.chartokenizer import CharTokenizer
    from tokenizers import Tokenizer as HFTokenizer
    HF_TOKENIZERS_AVAILABLE = True
except ImportError:
    print("Warning: Could not import some tokenizers. Logic might be affected.")
    IpaTokenizer = None
    CharTokenizer = None
    HFTokenizer = None
    HF_TOKENIZERS_AVAILABLE = False

logger = logging.getLogger(__name__)

class GLUEDataset(Dataset):
    """
    PyTorch Dataset class for loading and processing GLUE benchmark tasks.
    Handles different tokenizers and prepares batches with input IDs,
    auxiliary input (attention mask or custom positions), and labels.
    """
    def __init__(self, split, tokenizer, max_len, task, dataset_path):
        """Initializes the GLUEDataset with tokenizer settings and loads the GLUE dataset."""
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.task = task
        self.split = split
        self.is_ipa_tokenizer = IpaTokenizer is not None and isinstance(self.tokenizer, IpaTokenizer)
        self.is_hf_tokenizer = HFTokenizer is not None and isinstance(self.tokenizer, HFTokenizer)
        self.is_char_tokenizer = CharTokenizer is not None and isinstance(self.tokenizer, CharTokenizer)
        logger.info(f"Initializing GLUEDataset: task={task}, split={split}, max_len={max_len}, tokenizer={type(tokenizer).__name__}, is_ipa={self.is_ipa_tokenizer}, is_hf={self.is_hf_tokenizer}, is_char={self.is_char_tokenizer}")

        logger.info(f"Loading GLUE dataset: task='{task}', split='{split}', cache_dir='{dataset_path}'")
        try:
            actual_split_name = split
            if task == 'mnli' and split.startswith('validation'):
                pass
            elif task != 'mnli' and split == 'test':
                logger.warning(f"Loading GLUE test split {task}. Labels may be missing.")
            self.dataset = load_dataset("glue", task, split=actual_split_name, cache_dir=dataset_path if os.path.isdir(dataset_path) else None)
            logger.info(f"Loaded {len(self.dataset)} samples for {task}/{actual_split_name}.")
            logger.debug(f"First 3 samples: {self.dataset[:3]}")
        except Exception as e:
            logger.error(f"Failed to load dataset for {task}/{split}: {e}", exc_info=True)
            raise e

        self.text_keys = self._get_text_keys(task)
        logger.info(f"Using text keys for task '{task}': {self.text_keys}")

        self.label_key = 'label'
        if task == 'stsb':
            self.num_classes = 1
        else:
            if self.label_key not in self.dataset.features:
                logger.warning(f"Label column '{self.label_key}' not found in split '{split}'. Assuming test set.")
                self.num_classes = 2
            else:
                self.num_classes = self.dataset.features[self.label_key].num_classes

    def _get_text_keys(self, task):
        """Returns the appropriate text column keys for the given GLUE task."""
        key_map = {
            "cola": ["sentence"],
            "sst2": ["sentence"],
            "mrpc": ["sentence1", "sentence2"],
            "qqp": ["question1", "question2"],
            "stsb": ["sentence1", "sentence2"],
            "mnli": ["premise", "hypothesis"],
            "qnli": ["question", "sentence"],
            "rte": ["sentence1", "sentence2"],
            "wnli": ["sentence1", "sentence2"]
        }
        if task not in key_map:
            raise ValueError(f"Task '{task}' not recognized.")
        return key_map[task]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """Retrieves and processes a single sample by tokenizing text and preparing label and auxiliary input."""
        try:
            item = self.dataset[idx]
            label = item.get(self.label_key, -1 if self.task != 'stsb' else float('nan'))
            text = item.get(self.text_keys[0], "")
            text_pair = item.get(self.text_keys[1]) if len(self.text_keys) > 1 else None
            combined_text = text if not text_pair else text + " " + text_pair

            input_ids = None
            attention_mask = None
            custom_positions = None
            aux_input = None

            if self.is_ipa_tokenizer:
                if not hasattr(self.tokenizer, 'tokenize_text') or not hasattr(self.tokenizer, 'customize_positions'):
                    raise AttributeError("IpaTokenizer must have 'tokenize_text' and 'customize_positions'.")
                input_ids_list = self.tokenizer.tokenize_text(combined_text, seq_length=self.max_len)
                input_ids = torch.tensor(input_ids_list, dtype=torch.long)
                positions_list = self.tokenizer.customize_positions(combined_text, seq_length=self.max_len)
                if len(positions_list) != self.max_len:
                    pad_val = getattr(self.tokenizer, 'pad_pos_id', 0)
                    positions_list = positions_list[:self.max_len] + [pad_val] * (self.max_len - len(positions_list))
                custom_positions = torch.tensor(positions_list, dtype=torch.long)
                aux_input = custom_positions

            elif self.is_hf_tokenizer:
                try:
                    encoding = self.tokenizer.encode(
                        sequence=text,
                        pair=text_pair,
                        add_special_tokens=True
                    )
                    input_ids = torch.tensor(encoding.ids, dtype=torch.long)
                    attention_mask = torch.tensor(encoding.attention_mask, dtype=torch.long)
                    aux_input = attention_mask
                except Exception as hf_e:
                    logger.error(f"Item {idx} [HF]: Error during HF tokenizer encode: {hf_e}", exc_info=True)
                    raise hf_e

            elif self.is_char_tokenizer:
                if not hasattr(self.tokenizer, 'tokenize_text'):
                    raise AttributeError("CharTokenizer must have 'tokenize_text'.")
                input_ids_list = self.tokenizer.tokenize_text(combined_text, seq_length=self.max_len)
                input_ids = torch.tensor(input_ids_list, dtype=torch.long)
                pad_token_id = getattr(self.tokenizer, 'pad_id', 0)
                attention_mask_list = [1 if token_id != pad_token_id else 0 for token_id in input_ids_list]
                attention_mask = torch.tensor(attention_mask_list, dtype=torch.long)
                aux_input = attention_mask

            else:
                logger.debug(f"Item {idx}: Handling other tokenizer type: {type(self.tokenizer).__name__}")
                # Ở nhánh này, thay vì gọi encode, ta sử dụng phương thức tokenize_text
                if hasattr(self.tokenizer, 'tokenize_text') and callable(getattr(self.tokenizer, 'tokenize_text')):
                    ids_list = self.tokenizer.tokenize_text(combined_text, seq_length=self.max_len)
                    pad_id = getattr(self.tokenizer, 'pad_id', 0)
                    if len(ids_list) < self.max_len:
                        ids_list += [pad_id] * (self.max_len - len(ids_list))
                    elif len(ids_list) > self.max_len:
                        ids_list = ids_list[:self.max_len]
                    input_ids = torch.tensor(ids_list, dtype=torch.long)
                    attention_mask_list = [1 if token_id != pad_id else 0 for token_id in ids_list]
                    attention_mask = torch.tensor(attention_mask_list, dtype=torch.long)
                    aux_input = attention_mask
                else:
                    raise AttributeError(f"Tokenizer {type(self.tokenizer).__name__} missing required 'tokenize_text' method.")

            if aux_input is None:
                logger.warning(f"Item {idx}: Failed to generate auxiliary input. Creating default attention mask.")
                aux_input = torch.ones_like(input_ids)
            if input_ids.shape != aux_input.shape:
                raise ValueError(f"Item {idx}: Final shape mismatch input_ids ({input_ids.shape}) vs aux_input ({aux_input.shape}).")

            if self.task == 'stsb':
                label_tensor = torch.tensor(label if label != -1 else float('nan'), dtype=torch.float)
            else:
                label_tensor = torch.tensor(int(label), dtype=torch.long)

            return input_ids, aux_input, label_tensor

        except Exception as e:
            logger.error(f"Error processing item {idx}: {e}", exc_info=True)
            dummy_ids = torch.zeros(self.max_len, dtype=torch.long)
            dummy_aux = torch.zeros(self.max_len, dtype=torch.long)
            dummy_label = torch.tensor(-100 if self.task != 'stsb' else float('nan'), dtype=torch.long if self.task != 'stsb' else torch.float)
            return dummy_ids, dummy_aux, dummy_label
