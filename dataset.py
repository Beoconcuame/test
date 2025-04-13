import os
import numpy as np
import torch
from torch.utils.data import Dataset
from datasets import load_from_disk, load_dataset

class GLUEDataset(Dataset):
    def __init__(self, split, tokenizer, max_len, task, dataset_path="dataset"):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.task = task.lower()

        # Lấy đường dẫn tuyệt đối cho dataset_path
        data_root = os.path.abspath(dataset_path)
        print("[DEBUG] Data root:", data_root)
        
        # Với MNLI, chuyển đổi split theo yêu cầu
        if self.task == "mnli":
            if split == "validation":
                split = "validation_matched"
            elif split == "test":
                split = "test_matched"
        
        # Xây dựng đường dẫn đến folder và file CSV theo mẫu
        folder_path = os.path.join(data_root, f"{self.task}_{split}")
        csv_path = os.path.join(data_root, f"{self.task}_{split}.csv")
        print("[DEBUG] Folder path:", folder_path)
        print("[DEBUG] CSV path:", csv_path)
        
        # Nếu folder đã được lưu bằng save_to_disk tồn tại, sử dụng load_from_disk
        if os.path.isdir(folder_path):
            print("[DEBUG] Loading dataset from disk folder:", folder_path)
            self.dataset = load_from_disk(folder_path)
        # Nếu không có folder, kiểm tra file CSV
        elif os.path.isfile(csv_path):
            print("[DEBUG] Loading dataset from CSV file:", csv_path)
            self.dataset = load_dataset("csv", data_files=csv_path, split="train")
        else:
            print("[DEBUG] Data unprocessed: downloading dataset from Hugging Face")
            # Tải dataset gốc từ Hugging Face
            self.dataset = load_dataset("glue", self.task, split=split)
            
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        # Xử lý các cấu trúc dữ liệu dựa trên các cột có sẵn trong item
        if "sentence" in item:
            text = item["sentence"].lower()
            token_indices = self.tokenizer.tokenize_text(text, seq_length=self.max_len)
        elif "sentence1" in item and ("sentence2" in item or "question" in item):
            text1 = item["sentence1"].lower()
            text2 = item["question"].lower() if "question" in item else item["sentence2"].lower()
            token_indices = self.tokenizer.tokenize_text(text1 + " " + text2, seq_length=self.max_len)
        elif "question1" in item and "question2" in item:
            text1 = item["question1"].lower()
            text2 = item["question2"].lower()
            token_indices = self.tokenizer.tokenize_text(text1 + " " + text2, seq_length=self.max_len)
        elif "premise" in item and "hypothesis" in item:
            text1 = item["premise"].lower()
            text2 = item["hypothesis"].lower()
            token_indices = self.tokenizer.tokenize_text(text1 + " " + text2, seq_length=self.max_len)
        else:
            raise ValueError(f"No supported structure: {item}")
        
        # Cắt hoặc padding token_indices cho phù hợp với max_len
        token_indices = token_indices[:self.max_len]
        if len(token_indices) < self.max_len:
            token_indices += [0] * (self.max_len - len(token_indices))
        token_indices = torch.tensor(token_indices, dtype=torch.long)
        
        label = item["label"]
        # Nếu task là stsb thì chuyển label sang float, còn lại dùng long
        label = torch.tensor(label, dtype=torch.float if self.task == "stsb" else torch.long)
        
        return token_indices, label
