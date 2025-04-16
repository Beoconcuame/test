import unicodedata
import re
import csv
import stanza
from typing import List, Tuple, Optional
import os

# Tải và khởi tạo Stanza (dùng cho lemmatization nếu cần)
stanza.download('en')
nlp = stanza.Pipeline('en', processors='tokenize,mwt,pos,lemma')

def split_syllable_IPA(IPA: str) -> List[str]:
    """
    Chuẩn hóa chuỗi IPA và tách thành các âm tiết dựa vào một số ký tự định nghĩa.
    """
    IPA = IPA.replace("ţ", "t").replace("ɑ˞", "ɑːr").replace("ɔ˞", "ɔːr")\
             .replace("ɝ", "ɜːr").replace("ɚ", "ər")\
             .replace('̭', "").replace('̬', "").replace('˞', "")
    IPA = IPA.replace("ˌ", " ").replace("ˈ", " ")\
             .replace(".", " ").replace("·", " ").replace("-", "")
    return IPA.split()

def split_syllable_to_phoneme(syllable: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Tách một âm tiết thành (onset, nucleus, coda) dựa vào danh sách phoneme cho phụ âm và nguyên âm.
    Nếu không tìm được phần nào, trả về None cho phần đó.
    """
    # Các phụ âm (ưu tiên các cặp ký tự trước)
    consonants = [
        "tʃ", "dʒ", "hw", "kj", "lj", "nj", "sj",  
        "b", "h", "j", "k", "l", "m", "n", "ŋ", "t", 
        "tj", "θ", "θj", "v", "w", "z", "zj", "ʒ", "d", 
        "dj", "ð", "p", "r", "ʃ", "f", "ɡ", "s"
    ]
    # Danh sách nguyên âm (bao gồm "əʊ" khớp với test case)
    vowels = [
        "aɪər", "aʊər", "ɔɪər", "ɛər", "ɪər", 
        "ʊər", "eɪ", "aɪ", "aʊ", "oʊ", "əʊ", "ɔɪ", 
        "ɑːr", "ær", "ɒr", "ɛr", "ɪr", "ɔːr", 
        "ʊr", "ɜːr", "ʌr", "iə", "uə", "ər", 
        "oʊ", "əl", "ɑː", "ɒ", "æ", "ɛ", "ɜː", 
        "ɪ", "i", "iː", "ɔː", "ʊ", "uː", "u", 
        "ʌ", "ə", "e"
    ]
    def get_phoneme(syllable: str, phonemes: List[str]) -> Tuple[List[str], str]:
        selected = []
        while syllable:
            found = False
            for phon in phonemes:
                if syllable.startswith(phon):
                    selected.append(phon)
                    syllable = syllable[len(phon):]
                    found = True
                    break
            if not found:
                break
        return selected, syllable

    initials, rem = get_phoneme(syllable, consonants)
    nuclei, rem = get_phoneme(rem, vowels)
    finals, rem = get_phoneme(rem, consonants)

    onset = "".join(initials) if initials else None
    nucleus = "".join(nuclei) if nuclei else None
    coda = "".join(finals) if finals else None
    return onset, nucleus, coda

def convert_English_IPA_to_phoneme(IPA: str) -> List[Tuple[Optional[str], Optional[str], Optional[str]]]:
    """
    Chuyển chuỗi IPA thành danh sách các tuple (onset, nucleus, coda) cho từng âm tiết.
    """
    syllables = split_syllable_IPA(IPA)
    return [split_syllable_to_phoneme(syll) for syll in syllables]

class IpaTokenizer:
    def __init__(self, csv_file: str, empty_token: str = "<EMPTY>", unknown_token: str = "<UNK>", 
                 space_token: str = "<_>", pad_token: str = "<PAD>"):
        self.empty_token = empty_token
        self.unknown_token = unknown_token
        self.space_token = space_token
        self.pad_token = pad_token
        self.token2idx = {}
        self.idx2token = {}
        self.word_to_ipa = {}
        self.ipa_to_word = {}
        self.build_vocab_and_mapping(csv_file)

    def build_vocab_and_mapping(self, csv_file: str):
        """
        Xây dựng từ điển các token dựa vào file CSV với cột "word" và "ipa".
        Với mỗi IPA, ta tách theo các thành phần của âm tiết rồi thêm từng phần (nếu có) vào từ điển.
        """
        tokens_set = set()
        with open(csv_file, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                word = row["word"].strip().lower()
                ipa = row["ipa"].strip()
                if len(word) == 1:
                    self.word_to_ipa[word] = ipa
                    self.ipa_to_word[ipa] = word
                else:
                    if word in self.word_to_ipa:
                        if isinstance(self.word_to_ipa[word], list):
                            self.word_to_ipa[word].append(ipa)
                        else:
                            self.word_to_ipa[word] = [self.word_to_ipa[word], ipa]
                    else:
                        self.word_to_ipa[word] = ipa
                    self.ipa_to_word[ipa] = word
                components = convert_English_IPA_to_phoneme(ipa)
                for onset, nucleus, coda in components:
                    if onset is not None:
                        tokens_set.add(onset)
                    if nucleus is not None:
                        tokens_set.add(nucleus)
                    if coda is not None:
                        tokens_set.add(coda)
        
        tokens_set.add(self.space_token)
        tokens_set.add(self.unknown_token)
        tokens_set.add(self.pad_token)
        tokens_set.add(self.empty_token)
        
        tokens_list = sorted(tokens_set)
        self.token2idx = {token: idx for idx, token in enumerate(tokens_list)}
        self.idx2token = {idx: token for token, idx in self.token2idx.items()}

    def lemmatize_text(self, text: str) -> str:
        text = text.lower()
        doc = nlp(text)
        return " ".join(word.lemma for sent in doc.sentences for word in sent.words)

    def text_to_flat_phonemes(self, text: str) -> List[str]:
        """
        Chuyển đổi văn bản đầu vào thành danh sách các token IPA (theo thứ tự)
        – Nếu từ có mapping, ta chuyển qua IPA rồi “làm phẳng” từng âm tiết (loại bỏ các phần None).
        – Nếu từ không có mapping, mỗi ký tự được xem như token unknown.
        – Với khoảng trắng, token space sẽ được thêm vào.
        """
        phoneme_tokens = []
        idx = 0
        while idx < len(text):
            char = text[idx]
            if char.isspace():
                while idx < len(text) and text[idx].isspace():
                    phoneme_tokens.append(self.space_token)
                    idx += 1
            elif char.isdigit():
                while idx < len(text) and text[idx].isdigit():
                    token = text[idx] if text[idx] in self.token2idx else self.unknown_token
                    phoneme_tokens.append(token)
                    idx += 1
            elif char.isalpha():
                start = idx
                while idx < len(text) and text[idx].isalpha():
                    idx += 1
                word = text[start:idx].lower()
                if word in self.word_to_ipa:
                    ipa = self.word_to_ipa[word] if isinstance(self.word_to_ipa[word], str) else self.word_to_ipa[word][0]
                    components = convert_English_IPA_to_phoneme(ipa)
                    for tpl in components:
                        for elem in tpl:
                            if elem is not None:
                                phoneme_tokens.append(elem)
                else:
                    for _ in word:
                        phoneme_tokens.append(self.unknown_token)
            else:
                phoneme_tokens.append(self.unknown_token)
                idx += 1
        return phoneme_tokens

    def tokenize_text(self, text: str, seq_length: int = None) -> List[int]:
        """
        Chuyển đổi văn bản đầu vào thành danh sách các id dựa theo mapping của token IPA.
        Nếu seq_length được chỉ định, kết quả sẽ được padding/cắt cho phù hợp.
        """
        phoneme_tokens = self.text_to_flat_phonemes(text)
        token_ids = [self.token2idx.get(token, self.token2idx[self.unknown_token]) for token in phoneme_tokens]
        if seq_length is not None:
            current_len = len(token_ids)
            if current_len < seq_length:
                pad_id = self.token2idx[self.pad_token]
                token_ids.extend([pad_id] * (seq_length - current_len))
            elif current_len > seq_length:
                token_ids = token_ids[:seq_length]
        return token_ids

    def customize_positions(self, text: str, seq_length: int = None) -> List[int]:
        """
        Tạo danh sách vị trí cho mỗi token theo quy tắc:
          - 0: token khoảng trắng
          - 1: onset
          - 2: nucleus
          - 3: coda
          - 4: dùng cho chữ số hoặc token unknown (theo yêu cầu đối với chữ số không xác định)
        
        Sau đó, nếu seq_length được chỉ định và danh sách vị trí chưa đủ,
        padding thêm với giá trị 7 (đại diện cho <pad>).
        """
        positions = []
        idx = 0
        while idx < len(text):
            char = text[idx]
            if char.isspace():
                while idx < len(text) and text[idx].isspace():
                    positions.append(0)  # 0 cho khoảng trắng
                    idx += 1
            elif char.isdigit():
                # Với chữ số, theo yêu cầu gán 4
                while idx < len(text) and text[idx].isdigit():
                    positions.append(4)
                    idx += 1
            elif char.isalpha():
                start = idx
                while idx < len(text) and text[idx].isalpha():
                    idx += 1
                word = text[start:idx].lower()
                if word in self.word_to_ipa:
                    ipa = self.word_to_ipa[word] if isinstance(self.word_to_ipa[word], str) else self.word_to_ipa[word][0]
                    components = convert_English_IPA_to_phoneme(ipa)
                    for tpl in components:
                        # Gán theo thứ tự: onset -> 1, nucleus -> 2, coda -> 3
                        if tpl[0] is not None:
                            positions.append(1)
                        if tpl[1] is not None:
                            positions.append(2)
                        if tpl[2] is not None:
                            positions.append(3)
                else:
                    # Nếu không có mapping, mỗi ký tự được gán mặc định là 4 (unknown)
                    for _ in word:
                        positions.append(4)
            else:
                positions.append(4)
                idx += 1

        if seq_length is not None:
            current_len = len(positions)
            if current_len < seq_length:
                positions.extend([7] * (seq_length - current_len))
            elif current_len > seq_length:
                positions = positions[:seq_length]
        return positions

    def export_mapping(self, output_file: str):
        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["token", "index"])
            for token, idx in sorted(self.token2idx.items(), key=lambda x: x[1]):
                writer.writerow([token, idx])
        print(f"Output file: {output_file}")