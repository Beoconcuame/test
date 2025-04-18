class ByteTokenizer:
    def __init__(self, special_tokens=["<PAD>", "<UNK>"]):
        self.special_tokens = special_tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.pad_id = 256
        self.unk_id = 257

    def tokenize_text(self, text, seq_length=None):
        try:
            byte_seq = list(text.encode('utf-8'))
        except Exception as e:
            byte_seq = [self.unk_id]
        tokens = byte_seq
        if seq_length is not None:
            current_len = len(tokens)
            if current_len < seq_length:
                tokens = tokens + [self.pad_id] * (seq_length - current_len)
            elif current_len > seq_length:
                tokens = tokens[:seq_length]
        return tokens

    def encode(self, text, seq_length=None, **kwargs):
        """
        Phương thức encode chuyển đổi chuỗi thành danh sách token.
        Nó hỗ trợ đối số 'max_length' để tương thích với các nơi gọi tokenizer.encode(..., max_length=...).
        """
        if 'max_length' in kwargs:
            seq_length = kwargs['max_length']
        return self.tokenize_text(text, seq_length)

    def get_vocab_size(self):
        return 258

    def __call__(self, text, seq_length=None, **kwargs):
        return self.encode(text, seq_length, **kwargs)
