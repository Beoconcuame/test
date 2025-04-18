from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

class BPETokenizer:
    def __init__(self, train_sentences, special_tokens=["<UNK>", "<PAD>"], vocab_size=30000):
        self.tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
        self.tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(special_tokens=special_tokens, vocab_size=vocab_size)
        self.tokenizer.train_from_iterator(train_sentences, trainer=trainer)
        self.vocab = self.tokenizer.get_vocab()
        self.pad_id = self.vocab["<PAD>"]
        self.vocab_size = vocab_size

    def tokenize_text(self, text, seq_length=None):
        tokens = self.tokenizer.encode(text).ids
        if seq_length is not None:
            current_len = len(tokens)
            if current_len < seq_length:
                tokens = tokens + [self.pad_id] * (seq_length - current_len)
            elif current_len > seq_length:
                tokens = tokens[:seq_length]
        return tokens
    def get_vocab_size(self):
        return self.vocab_size
    
