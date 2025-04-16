import sys
import os
try:
    from tokenizers import Tokenizer
    from tokenizers import normalizers
    from tokenizers.normalizers import NFD, StripAccents, Lowercase
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.processors import BertProcessing
    from tokenizers.models import BPE, Unigram, WordPiece
    from tokenizers.trainers import BpeTrainer, UnigramTrainer, WordPieceTrainer
    HF_TOKENIZERS_AVAILABLE = True
except ImportError:
    HF_TOKENIZERS_AVAILABLE = False
    print("CRITICAL Warning: Hugging Face 'tokenizers' library not found. BPE, Unigram, WordPiece tokenizers will not be available.")

from datasets import load_dataset, disable_caching
import logging

logger = logging.getLogger(__name__)

try:
    from tokenizer.ipatokenizer2 import IpaTokenizer
    from tokenizer.chartokenizer import CharTokenizer
    from tokenizer.bytetokenizer import ByteTokenizer
except ImportError as e:
    logger.error(f"CRITICAL: Error importing custom tokenizer classes: {e}.", exc_info=True)
    sys.exit(1)

def get_sentences_for_training(task, dataset_path, split="train"):
    """
    Load and return sentences from the GLUE dataset for tokenizer training.
    
    Args:
        task (str): The GLUE task name.
        dataset_path (str): Path to the dataset cache.
        split (str, optional): Dataset split to use (default "train").

    Returns:
        list: A list of sentences extracted from the dataset.
    """
    try:
        logger.info(f"Loading dataset 'glue/{task}' split '{split}' from path '{dataset_path}' for tokenizer training...")
        glue_data = load_dataset("glue", task, split=split, cache_dir=dataset_path if os.path.isdir(dataset_path) else None)
        logger.info(f"Dataset '{task}/{split}' loaded successfully.")
        if "sentence" in glue_data.column_names: return glue_data["sentence"]
        if "question" in glue_data.column_names and "sentence" in glue_data.column_names: return glue_data["question"] + glue_data["sentence"]
        if "sentence1" in glue_data.column_names and "sentence2" in glue_data.column_names: return glue_data["sentence1"] + glue_data["sentence2"]
        if "premise" in glue_data.column_names and "hypothesis" in glue_data.column_names: return glue_data["premise"] + glue_data["hypothesis"]
        logger.warning(f"Could not find standard text columns for task '{task}'. Concatenating all string columns.")
        text_cols = [col for col, dtype in glue_data.features.items() if dtype.dtype == 'string']
        if not text_cols: raise ValueError(f"Cannot determine text columns for task {task}.")
        logger.info(f"Using columns for tokenizer training: {text_cols}")
        sentences = []
        [sentences.extend([s for s in glue_data[col] if isinstance(s, str) and s]) for col in text_cols]
        if not sentences: logger.warning(f"Extracted sentences for tokenizer training appear empty for task '{task}'.")
        return sentences
    except Exception as e:
        logger.error(f"Error loading dataset/extracting sentences for task {task}: {e}", exc_info=True)
        sys.exit(1)

def create_tokenizer(config):
    """
    Creates and returns a tokenizer instance based on the given configuration.
    
    Supports various tokenizer types including IPA, BPE, Unigram, WordPiece, Char, and Byte.
    For HF tokenizers, enables padding and truncation using the provided max_len.
    
    Args:
        config (dict): The configuration dictionary.

    Returns:
        tuple: A tuple (tokenizer, vocab_size) where tokenizer is the initialized tokenizer instance
               and vocab_size is its vocabulary size.
    """
    tokenizer_name = config['tokenizer']
    vocab_file = config.get('vocab_file')
    run_mode = config.get('run_mode', 'train')
    task = config['task']
    dataset_path = config['dataset_path']
    vocab_size_config = config.get('vocab_size', 30000)
    tokenizer_model_dir = config.get("tokenizer_model_dir", "trained_tokenizers")
    os.makedirs(tokenizer_model_dir, exist_ok=True)
    max_len = config['max_len']

    tokenizer = None
    vocab_size = None

    logger.info(f"Creating/Loading tokenizer: {tokenizer_name}")

    try:
        if tokenizer_name == "ipa":
            if not vocab_file or not os.path.exists(vocab_file):
                raise FileNotFoundError(f"vocab_file '{vocab_file}' required for IPA.")
            tokenizer = IpaTokenizer(vocab_file, empty_token="<EMPTY>", unknown_token="<UNK>")
            if hasattr(tokenizer, 'token2idx'):
                vocab_size = len(tokenizer.token2idx)
            elif hasattr(tokenizer, 'vocab'):
                vocab_size = len(tokenizer.vocab)
            else:
                raise AttributeError("IpaTokenizer missing 'token2idx' or 'vocab'.")
            logger.info(f"Initialized IpaTokenizer. Vocab size: {vocab_size}")

        elif tokenizer_name in ["bpe", "unigram", "wordpiece"]:
            if not HF_TOKENIZERS_AVAILABLE:
                raise ImportError("HF 'tokenizers' library required.")
            tokenizer_model_path = os.path.join(tokenizer_model_dir, f"{tokenizer_name}_{task}_{vocab_size_config}.json")
            special_tokens = ["<PAD>", "<UNK>", "<SOS>", "<EOS>", "<CLS>", "<SEP>", "<MASK>"]
            pad_token = "<PAD>"
            cls_token = "<CLS>"
            sep_token = "<SEP>"
            tokenizer_loaded = False
            if os.path.exists(tokenizer_model_path):
                logger.info(f"Attempting to load {tokenizer_name} model from: {tokenizer_model_path}")
                try:
                    tokenizer = Tokenizer.from_file(tokenizer_model_path)
                    vocab_size = tokenizer.get_vocab_size()
                    tokenizer_loaded = True
                    logger.info(f"Successfully loaded pre-trained {tokenizer_name}. Vocab size: {vocab_size}")
                except Exception as e:
                    logger.error(f"Error loading tokenizer from {tokenizer_model_path}: {e}. Retraining if in 'train' mode.", exc_info=True)
                    tokenizer_loaded = False

            if not tokenizer_loaded and run_mode == "train":
                logger.info(f"Training new {tokenizer_name} tokenizer...")
                train_sentences = get_sentences_for_training(task, dataset_path)
                if not train_sentences:
                    raise ValueError("No sentences for training.")
                unk_token = "<UNK>"
                if tokenizer_name == "bpe":
                    model_tok = BPE(unk_token=unk_token)
                    trainer = BpeTrainer(vocab_size=vocab_size_config, special_tokens=special_tokens)
                elif tokenizer_name == "unigram":
                    model_tok = Unigram()
                    trainer = UnigramTrainer(vocab_size=vocab_size_config, special_tokens=special_tokens, unk_token=unk_token)
                elif tokenizer_name == "wordpiece":
                    model_tok = WordPiece(unk_token=unk_token)
                    trainer = WordPieceTrainer(vocab_size=vocab_size_config, special_tokens=special_tokens)
                else:
                    raise ValueError(f"Invalid tokenizer name {tokenizer_name}")
                tokenizer_instance = Tokenizer(model_tok)
                tokenizer_instance.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])
                tokenizer_instance.pre_tokenizer = Whitespace()
                logger.info(f"Training {tokenizer_name} with vocab_size={vocab_size_config}...")
                tokenizer_instance.train_from_iterator(train_sentences, trainer=trainer)
                vocab_size = tokenizer_instance.get_vocab_size()
                logger.info(f"{tokenizer_name} tokenizer trained. Actual vocab size: {vocab_size}")
                pad_token_id = tokenizer_instance.token_to_id(pad_token)
                if pad_token_id is not None:
                    tokenizer_instance.enable_padding(pad_id=pad_token_id, pad_token=pad_token, length=max_len, direction='right')
                    logger.info(f"Enabled padding to max_length={max_len} with pad_token='{pad_token}' (ID: {pad_token_id})")
                else:
                    logger.warning("Could not find PAD token ID. Padding may not work correctly.")
                tokenizer_instance.enable_truncation(max_length=max_len, strategy='longest_first', stride=0)
                logger.info(f"Enabled truncation to max_length={max_len}")
                cls_token_id = tokenizer_instance.token_to_id(cls_token)
                sep_token_id = tokenizer_instance.token_to_id(sep_token)
                if cls_token_id is not None and sep_token_id is not None:
                    logger.info(f"Setting BertProcessing with SEP ('{sep_token}', {sep_token_id}) and CLS ('{cls_token}', {cls_token_id})")
                    tokenizer_instance.post_processor = BertProcessing(sep=(sep_token, sep_token_id), cls=(cls_token, cls_token_id))
                else:
                    logger.warning("CLS or SEP token ID not found. BertProcessing post-processor not set.")
                try:
                    tokenizer_instance.save(tokenizer_model_path)
                    logger.info(f"Saved trained tokenizer to {tokenizer_model_path}")
                except Exception as e:
                    logger.error(f"Error saving tokenizer: {e}")
                tokenizer = tokenizer_instance

            elif not tokenizer_loaded and run_mode != "train":
                raise FileNotFoundError(f"Cannot run '{run_mode}'. Pre-trained {tokenizer_name} model not found at '{tokenizer_model_path}'.")

            if tokenizer is not None:
                pad_token_id = tokenizer.token_to_id(pad_token)
                if pad_token_id is not None:
                    if tokenizer.padding is None or tokenizer.padding.get('length') != max_len:
                        tokenizer.enable_padding(pad_id=pad_token_id, pad_token=pad_token, length=max_len, direction='right')
                        logger.info(f"Enabled/updated padding for loaded tokenizer to max_length={max_len}")
                else:
                    logger.warning(f"PAD token ID not found in loaded tokenizer. Padding may fail.")
                if tokenizer.truncation is None or tokenizer.truncation.get('max_length') != max_len:
                    tokenizer.enable_truncation(max_length=max_len, strategy='longest_first', stride=0)
                    logger.info(f"Enabled/updated truncation for loaded tokenizer to max_length={max_len}")

        elif tokenizer_name == "char":
            tokenizer = CharTokenizer()
            if run_mode == 'train':
                logger.warning("CharTokenizer fitting logic might be needed.")
            if hasattr(tokenizer, 'get_vocab_size'):
                vocab_size = tokenizer.get_vocab_size()
            elif hasattr(tokenizer, 'char_index'):
                vocab_size = len(tokenizer.char_list)
            else:
                vocab_size = 256
                logger.warning("Estimating CharTokenizer vocab size as 256.")
        elif tokenizer_name == "byte":
            tokenizer = ByteTokenizer()
            if hasattr(tokenizer, 'get_vocab_size'):
                vocab_size = tokenizer.get_vocab_size()
            else:
                vocab_size = 257
                logger.warning(f"Estimating ByteTokenizer size: {vocab_size}")
        else:
            raise ValueError(f"Invalid tokenizer name '{tokenizer_name}'.")
    except FileNotFoundError as fnf:
        logger.error(f"CRITICAL: File not found: {fnf}")
        sys.exit(1)
    except AttributeError as ae:
        logger.error(f"CRITICAL: Missing attribute/method: {ae}")
        sys.exit(1)
    except ValueError as ve:
        logger.error(f"CRITICAL: Value error: {ve}")
        sys.exit(1)
    except ImportError as ie:
        logger.error(f"CRITICAL: Import error: {ie}. Is '{tokenizer_name}' supported or library installed?")
        sys.exit(1)
    except Exception as e:
        logger.error(f"CRITICAL: Failed to create tokenizer: {e}", exc_info=True)
        sys.exit(1)

    if tokenizer is None or vocab_size is None:
        logger.error(f"CRITICAL: Tokenizer/vocab_size is None for '{tokenizer_name}'.")
        sys.exit(1)

    logger.info(f"Tokenizer '{tokenizer_name}' ready. Actual vocab size: {vocab_size}")
    return tokenizer, vocab_size
