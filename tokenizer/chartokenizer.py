import string

class CharTokenizer:
    def __init__(self):
        self.char_list = list(string.printable) + ["<pad>", "<unk>"]
        self.char_to_id = {char: i for i, char in enumerate(self.char_list)}
        self.id_to_char = {i: char for i, char in enumerate(self.char_list)}
        self.pad_id = self.char_to_id["<pad>"]
        self.unk_id = self.char_to_id["<unk>"]

    def tokenize_text(self, text, seq_length=None):
        tokens = [self.char_to_id.get(char, self.unk_id) for char in text]
    
        if seq_length is not None:
            current_len = len(tokens)
            if current_len < seq_length:
                tokens = tokens + [self.pad_id] * (seq_length - current_len)
            elif current_len > seq_length:
                tokens = tokens[:seq_length]
        return tokens

    def get_vocab_size(self):
        """Trả về kích thước từ vựng của CharTokenizer."""
        return len(self.char_list)

    def __call__(self, text, seq_length=None):
        return self.tokenize_text(text, seq_length)
