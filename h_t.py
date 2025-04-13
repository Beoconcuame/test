# hyperparameter_tuning.py

import optuna
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
import yaml
from datasets import load_dataset

# Import các module từ dự án của bạn
from ipatokenizer import IpaTokenizer
from train import get_model, train_model, evaluate_model
from dataset import GLUEDataset

def objective(trial):
    # Đọc file cấu hình ban đầu
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # Sử dụng Optuna để gợi ý các siêu tham số:
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-2)
    dropout = trial.suggest_uniform("dropout", 0.1, 0.5)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    
    # Cập nhật các siêu tham số vào config (chỉ dùng trong quá trình tuning)
    config["lr"] = lr
    config["batch_size"] = batch_size
    # Lưu ý: Nếu bạn muốn truyền dropout vào mô hình, đảm bảo hàm tạo mô hình (get_model) hỗ trợ tham số dropout.
    
    # Thiết lập thiết bị và thông số task
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task = config.get("task", "cola")
    
    # Tạo tokenizer (ở đây sử dụng IpaTokenizer theo file config)
    tokenizer = IpaTokenizer(config["vocab_file"], empty_token="<EMPTY>", unknown_token="<UNK>")
    
    # Tạo dataset cho train và validation
    dataset_train = GLUEDataset(split="train", tokenizer=tokenizer, max_len=config["max_len"], task=task, dataset_path=config["dataset_path"])
    dataset_val = GLUEDataset(split="validation", tokenizer=tokenizer, max_len=config["max_len"], task=task, dataset_path=config["dataset_path"])
    
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset_val, batch_size=batch_size)
    
    # Xác định số từ (vocab) và số lớp đầu ra
    vocab_size = len(tokenizer.token2idx)
    num_classes = {"cola": 2, "sst2": 2, "mrpc": 2, "qqp": 2, "mnli": 3, "qnli": 2, "rte": 2, "stsb": 1}[task]
    
    # Lấy model theo loại đã định (dựa trên file config; ví dụ: transformer, bilstm, …)
    model = get_model(config["model"], vocab_size, device, num_classes=num_classes)
    
    # Nếu model có thuộc tính dropout, cập nhật giá trị từ trial
    if hasattr(model, "dropout"):
        model.dropout = torch.nn.Dropout(dropout)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scaler = GradScaler()
    criterion = torch.nn.MSELoss() if task == "stsb" else torch.nn.CrossEntropyLoss()
    
    best_val_metric = -float("inf")
    tuning_epochs = 5  # Chọn số epoch nhỏ để tiết kiệm thời gian trong quá trình tuning
    for epoch in range(tuning_epochs):
        train_loss = train_model(model, train_loader, optimizer, criterion, device, scaler, task)
        val_metric = evaluate_model(model, val_loader, device, task)
        best_val_metric = max(best_val_metric, val_metric)
        # Báo cáo kết quả cho trial; giúp Optuna biết trial này triển vọng hay không
        trial.report(val_metric, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    
    return best_val_metric

if __name__ == "__main__":
    # Tạo study với mục tiêu maximize (giả sử metric cao hơn nghĩa là tốt hơn)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)  # Số trial có thể tăng nếu bạn có thời gian và tài nguyên
    
    print("Best trial:")
    best_trial = study.best_trial
    print("  Best Metric: {:.4f}".format(best_trial.value))
    print("  Best Hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")
