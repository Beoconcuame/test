import string

class CharTokenizer:
    def __init__(self):
        self.char_list = list(string.printable) + ["<pad>", "<bos>", "<eos>", "<unk>"]
        self.char_to_id = {char: i for i, char in enumerate(self.char_list)}
        self.id_to_char = {i: char for i, char in enumerate(self.char_list)}
        self.pad_id = self.char_to_id["<pad>"]
        self.bos_id = self.char_to_id["<bos>"]
        self.eos_id = self.char_to_id["<eos>"]
        self.unk_id = self.char_to_id["<unk>"]

    def tokenize_text(self, text, seq_length=None):
        tokens = [self.char_to_id.get(char, self.unk_id) for char in text]
        tokens = [self.bos_id] + tokens + [self.eos_id]
        if seq_length is not None:
            current_len = len(tokens)
            if current_len < seq_length:
                tokens = tokens[:-1] + [self.pad_id] * (seq_length - current_len) + [self.eos_id]
            elif current_len > seq_length and seq_length >= 2:
                tokens = [tokens[0]] + tokens[1:seq_length-1] + [tokens[-1]]
        return tokens



    def __call__(self, text, seq_length=None):
        return self.tokenize_text(text, seq_length)