class ByteTokenizer:
    def __init__(self, special_tokens=["<PAD>", "<UNK>"]):
        self.special_tokens = special_tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.pad_id = 0
        self.unk_id = 256

    def tokenize_text(self, text, seq_length=None):
        # Chuyển mỗi ký tự trong text thành số nguyên đại diện theo bảng mã UTF-8,
        # nếu không thuộc khoảng [0, 255] thì sử dụng unk_id
        byte_ids = [ord(char) if 0 <= ord(char) <= 255 else self.unk_id for char in text]
        tokens = byte_ids  # Không thêm BOS hay EOS
        
        # Nếu seq_length được chỉ định, thực hiện padding hoặc cắt danh sách token
        if seq_length is not None:
            current_len = len(tokens)
            if current_len < seq_length:
                tokens = tokens + [self.pad_id] * (seq_length - current_len)
            elif current_len > seq_length:
                tokens = tokens[:seq_length]
        return tokens

    def __call__(self, text, seq_length=None):
        return self.tokenize_text(text, seq_length)
