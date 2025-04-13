import unicodedata
import re
import csv
import stanza
from typing import List, Tuple
import os

stanza.download('en')
nlp = stanza.Pipeline('en', processors='tokenize,mwt,pos,lemma')

def split_syllable_IPA(IPA: str) -> list[str]:
    # Chuẩn hóa IPA và tách thành các âm tiết
    IPA = IPA.replace("ţ", "t").replace("ɑ˞", "ɑːr").replace("ɔ˞", "ɔːr").replace("ɝ", "ɜːr").replace("ɚ", "ər").replace('̭', "").replace('̬', "").replace('˞', "")
    IPA = IPA.replace("ˌ", " ").replace("ˈ", " ").replace(".", " ").replace("·", " ").replace("-", "")
    return IPA.split()

def split_syllable_to_phoneme(syllable: str) -> tuple[str, str, str]:
    consonants = [
        "tʃ", "dʒ", "b", "h", "hw", "j", "k", "kj", 
        "l", "lj", "m", "n", "nj", "ŋ", "t", "tj", 
        "θ", "θj", "v", "w", "z", "zj", "ʒ", "d", 
        "dj", "ð", "p", "r", "ʃ", "f", "ɡ", "s", "sj"
    ]
    vowels = [
        "aɪər", "aʊər", "ɔɪər", "ɛər", "ɪər", 
        "ʊər", "eɪ", "aɪ", "aʊ", "oʊ", "ɔɪ", 
        "ɑːr", "ær", "ɒr", "ɛr", "ɪr", "ɔːr", 
        "ʊr", "ɜːr", "ʌr", "iə", "uə", "ər", 
        "oʊ", "əl", "ɑː", "ɒ", "æ", "ɛ", "ɜː", 
        "ɪ", "i", "iː", "ɔː", "ʊ", "uː", "u", 
        "ʌ", "ə", "e"
    ]

    def get_phoneme(syllable: str, phonemes: list[str]) -> list[str]:
        if not syllable:
            return [], syllable
        selected_phonemes = []
        while syllable:
            found = False
            for phoneme in phonemes:
                if syllable.startswith(phoneme):
                    selected_phonemes.append(phoneme)
                    syllable = syllable[len(phoneme):]
                    found = True
                    break
            if not found:
                break
        return selected_phonemes, syllable

    initials, syllable = get_phoneme(syllable, consonants)
    nuclei, syllable = get_phoneme(syllable, vowels)
    finals, syllable = get_phoneme(syllable, consonants)
    return ("".join(initials) or None, "".join(nuclei) or None, "".join(finals) or None)

def convert_English_IPA_to_phoneme(IPA: str) -> list[tuple[str, str, str]]:
    syllables = split_syllable_IPA(IPA)
    phonemes = []
    for syllable in syllables:
        phonemes.append(split_syllable_to_phoneme(syllable))
    return phonemes

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
                    self.word_to_ipa[word] = self.word_to_ipa.get(word, []) + [ipa]
                    self.ipa_to_word[ipa] = word
                components = convert_English_IPA_to_phoneme(ipa)
                for onset, nucleus, coda in components:
                    tokens_set.add(onset or self.empty_token)
                    tokens_set.add(nucleus or self.empty_token)
                    tokens_set.add(coda or self.empty_token)

        tokens_set.add(self.space_token)
        tokens_set.add(self.unknown_token)
        tokens_set.add(self.pad_token)
        
        tokens_list = sorted(tokens_set)
        self.token2idx = {token: idx for idx, token in enumerate(tokens_list)}
        self.idx2token = {idx: token for token, idx in self.token2idx.items()}

    def lemmatize_text(self, text: str) -> str:
        text = text.lower()
        doc = nlp(text)
        return " ".join(word.lemma for sent in doc.sentences for word in sent.words)

    def tokenize_text(self, text: str, seq_length: int = None) -> List[Tuple[int, int, int]]:
        syllables = []
        idx_empty = self.token2idx[self.empty_token]
        idx_unk = self.token2idx[self.unknown_token]

        i = 0
        while i < len(text):
            char = text[i]
            if char.isspace():
                while i < len(text) and text[i].isspace():
                    idx_space = self.token2idx[self.space_token]
                    syllables.append((idx_empty, idx_space, idx_empty))
                    i += 1
            elif char.isdigit():
                while i < len(text) and text[i].isdigit():
                    idx_digit = self.token2idx.get(text[i], idx_unk)
                    syllables.append((idx_empty, idx_digit, idx_empty))
                    i += 1
            elif char.isalpha():
                start = i
                while i < len(text) and text[i].isalpha():
                    i += 1
                word = text[start:i].lower()
                if word in self.word_to_ipa:
                    ipa = self.word_to_ipa[word] if isinstance(self.word_to_ipa[word], str) else self.word_to_ipa[word][0]
                    components = convert_English_IPA_to_phoneme(ipa)
                    for onset, nucleus, coda in components:
                        idx_onset = self.token2idx.get(onset or self.empty_token, idx_unk)
                        idx_nucleus = self.token2idx.get(nucleus or self.empty_token, idx_unk)
                        idx_coda = self.token2idx.get(coda or self.empty_token, idx_unk)
                        syllables.append((idx_onset, idx_nucleus, idx_coda))
                else:
                    for char in word:
                        syllables.append((idx_unk, idx_unk, idx_unk))
            else:
                syllables.append((idx_unk, idx_unk, idx_unk))
                i += 1

        if seq_length is not None:
            current_len = len(syllables)
            if current_len < seq_length:
                pad_tuple = (idx_empty, self.token2idx[self.pad_token], idx_empty)
                syllables.extend([pad_tuple] * (seq_length - current_len))
            elif current_len > seq_length:
                syllables = syllables[:seq_length]

        return syllables

    def export_mapping(self, output_file: str):
        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["token", "index"])
            for token, idx in sorted(self.token2idx.items(), key=lambda x: x[1]):
                writer.writerow([token, idx])
        print(f"Output file: {output_file}")