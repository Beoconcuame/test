from tokenizers import Tokenizer
from tokenizers.models import Unigram
from tokenizers.trainers import UnigramTrainer
from tokenizers.pre_tokenizers import Whitespace

class UnigramTokenizer:
    def __init__(self, train_sentences, special_tokens=["<UNK>", "<PAD>"], vocab_size=30000):
        self.tokenizer = Tokenizer(Unigram())
        self.tokenizer.pre_tokenizer = Whitespace()
        trainer = UnigramTrainer(special_tokens=special_tokens, vocab_size=vocab_size)
        self.tokenizer.train_from_iterator(train_sentences, trainer=trainer)
        self.vocab = self.tokenizer.get_vocab()
        self.pad_id = self.vocab["<PAD>"]
        self.unk_id = self.vocab["<UNK>"]
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
