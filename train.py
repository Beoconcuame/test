import argparse
import datetime
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler 
from sklearn.metrics import matthews_corrcoef, accuracy_score
from datasets import load_dataset
from scipy.stats import pearsonr
import logging
import os
import yaml

from ipatokenizer import IpaTokenizer
from bpetokenizer import BPETokenizer
from unigramtokenizer import UnigramTokenizer
from chartokenizer import CharTokenizer
from wordpiecetokenizer import WordPieceTokenizer
from bytetokenizer import ByteTokenizer  

from ebilstm import EnhancedBiLSTMClassifier
from bilstm import BiLSTMClassifier
from bigru import BiGRUClassifier
from transformer import TransformerClassifier
from textcnn import TextCNN

from dataset import GLUEDataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

def get_model(model_name, vocab_size, device, num_classes=2):
    models = {
        "bilstm": BiLSTMClassifier,
        "bigru": BiGRUClassifier,
        "transformer": TransformerClassifier,
        "textcnn": TextCNN,
        "ebilstm": EnhancedBiLSTMClassifier
    }
    model_class = models.get(model_name, BiLSTMClassifier)
    return model_class(vocab_size=vocab_size, num_classes=num_classes).to(device)

def train_model(model, train_loader, optimizer, criterion, device, scaler, task):
    model.train()
    total_loss = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast(device_type=device.type):
            outputs = model(inputs)
            if task == "stsb":
                outputs = outputs.squeeze(-1)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate_model(model, data_loader, device, task):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            if task == "stsb":
                preds.extend(outputs.squeeze(-1).cpu().numpy())
            else:
                predictions = torch.argmax(outputs, dim=1).cpu().numpy()
                preds.extend(predictions)
            trues.extend(labels.cpu().numpy())
    
    if task == "stsb":
        return pearsonr(preds, trues)[0]
    elif task == "cola":
        return matthews_corrcoef(trues, preds)
    else:
        return accuracy_score(trues, preds)

def get_args():
    parser = argparse.ArgumentParser(description="GLUE tasks training with YAML configuration")
    parser.add_argument("--config_file", type=str, default="config.yaml", help="Path to YAML configuration file")
    parser.add_argument("--run_mode", type=str, choices=["train", "validation", "test"])
    parser.add_argument("--task", type=str, choices=["cola", "sst2", "mrpc", "qqp", "stsb", "mnli", "qnli", "rte"])
    parser.add_argument("--tokenizer", type=str, choices=["ipa", "bpe", "unigram", "char", "wordpiece", "byte"])
    parser.add_argument("--model", type=str, choices=["bilstm", "bigru", "transformer", "textcnn", "ebilstm"])
    parser.add_argument("--vocab_file", type=str)
    parser.add_argument("--max_len", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--checkpoint_path", type=str)
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")
    return parser.parse_args()

def main():
    args = get_args()
    
    if os.path.exists(args.config_file):
        with open(args.config_file, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    else:
        print(f"Configuration file {args.config_file} not found. Exiting.")
        sys.exit(1)
    
    run_mode = args.run_mode if args.run_mode is not None else config.get("run_mode", "train")
    task = args.task if args.task is not None else config.get("task", "cola")
    tokenizer_name = args.tokenizer if args.tokenizer is not None else config.get("tokenizer", "ipa")
    model_name = args.model if args.model is not None else config.get("model", "bilstm")
    vocab_file = args.vocab_file if args.vocab_file is not None else config.get("vocab_file", "thu.csv")
    max_len = args.max_len if args.max_len is not None else config.get("max_len", 50)
    batch_size = args.batch_size if args.batch_size is not None else config.get("batch_size", 32)
    epochs = args.epochs if args.epochs is not None else config.get("epochs", 5)
    lr = args.lr if args.lr is not None else config.get("lr", 1e-3)
    patience = args.patience if args.patience is not None else config.get("patience", 3)
    dataset_path = args.dataset_path if args.dataset_path is not None else config.get("dataset_path", "dataset")
    checkpoint_path = args.checkpoint_path if args.checkpoint_path is not None else config.get("checkpoint_path", f"best_model_{task}_{model_name}.pth")
    
    resume = args.resume if hasattr(args, "resume") and args.resume is not None else config.get("resume", False)
    
    configuration = {
        "run_mode": run_mode,
        "task": task,
        "tokenizer": tokenizer_name,
        "model": model_name,
        "vocab_file": vocab_file,
        "max_len": max_len,
        "batch_size": batch_size,
        "epochs": epochs,
        "lr": lr,
        "patience": patience,
        "dataset_path": dataset_path,
        "checkpoint_path": checkpoint_path,
        "resume": resume
    }
    print("Configuration loaded:")
    print(configuration)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    num_classes = {"cola": 2, "sst2": 2, "mrpc": 2, "qqp": 2, "mnli": 3, "qnli": 2, "rte": 2, "stsb": 1}[task]
    
    if tokenizer_name == "ipa":
        tokenizer = IpaTokenizer(vocab_file, empty_token="<EMPTY>", unknown_token="<UNK>")
    elif tokenizer_name in ["bpe", "unigram", "wordpiece"]:
        if run_mode == "train":
            glue_train = load_dataset("glue", task, split="train")
            if "sentence" in glue_train.column_names:
                train_sentences = glue_train["sentence"]
            elif "sentence1" in glue_train.column_names and "sentence2" in glue_train.column_names:
                train_sentences = glue_train["sentence1"] + glue_train["sentence2"]
            elif "premise" in glue_train.column_names and "hypothesis" in glue_train.column_names:
                train_sentences = glue_train["premise"] + glue_train["hypothesis"]
            elif "question" in glue_train.column_names and "sentence" in glue_train.column_names:
                train_sentences = glue_train["question"] + glue_train["sentence"]
            else:
                raise ValueError(f"Cannot determine sentences for tokenization for task {task}. Available columns: {glue_train.column_names}")
            
            if tokenizer_name == "bpe":
                tokenizer = BPETokenizer(train_sentences, special_tokens=["<UNK>", "<PAD>"], vocab_size=30000)
            elif tokenizer_name == "unigram":
                tokenizer = UnigramTokenizer(train_sentences, special_tokens=["<UNK>", "<PAD>"], vocab_size=30000)
            elif tokenizer_name == "wordpiece":
                tokenizer = WordPieceTokenizer(train_sentences, special_tokens=["<UNK>", "<PAD>"], vocab_size=30000)
        else:
            tokenizer = IpaTokenizer(vocab_file, empty_token="<EMPTY>", unknown_token="<UNK>")
    elif tokenizer_name == "char":
        tokenizer = CharTokenizer()
    elif tokenizer_name == "byte":
        tokenizer = ByteTokenizer()
    else:
        sys.exit("Invalid tokenizer. Choose 'ipa', 'bpe', 'unigram', 'char', 'wordpiece', or 'byte'.")
    
    if run_mode == "train":
        # Nếu không muốn tiếp tục training (resume==False) mà checkpoint đã tồn tại thì dừng lại để tránh ghi đè
        if not resume and os.path.isfile(checkpoint_path):
            print(f"Checkpoint file {checkpoint_path} already exists. To avoid overwriting the existing checkpoint, either set '--resume' to continue training or change the checkpoint path in the config.")
            sys.exit(1)
    
        dataset_train = GLUEDataset(split="train", tokenizer=tokenizer, max_len=max_len, task=task, dataset_path=dataset_path)
        dataset_val   = GLUEDataset(split="validation", tokenizer=tokenizer, max_len=max_len, task=task, dataset_path=dataset_path)
        train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(dataset_val, batch_size=batch_size)
        
        if tokenizer_name == "ipa":
            vocab_size = len(tokenizer.token2idx)
        elif tokenizer_name in ["bpe", "unigram", "wordpiece"]:
            vocab_size = len(tokenizer.vocab)
        elif tokenizer_name == "char":
            vocab_size = len(tokenizer.char_list)
        elif tokenizer_name == "byte":
            vocab_size = 259
        else:
            sys.exit("Cannot determine vocab size for the selected tokenizer.")
        
        model = get_model(model_name, vocab_size, device, num_classes=num_classes)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scaler = GradScaler()
        criterion = nn.MSELoss() if task == "stsb" else nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5, verbose=True)
        
        # Nếu resume được bật, nạp trọng số từ checkpoint (nếu tồn tại)
        if resume:
            if os.path.isfile(checkpoint_path):
                print(f"Resuming training from checkpoint: {checkpoint_path}")
                model.load_state_dict(torch.load(checkpoint_path))
            else:
                print(f"Checkpoint {checkpoint_path} not found. Training from scratch.")
        
        best_metric = -float('inf')
        patience_counter = 0
        
        print("Starting training...")
        results_log = [
            "Training Configuration:",
            f"Task: {task}",
            f"Tokenizer: {tokenizer_name}",
            f"Model: {model_name}",
            f"Vocab file: {vocab_file}",
            f"Max sequence length: {max_len}",
            f"Batch size: {batch_size}",
            f"Epochs: {epochs}",
            f"Learning rate: {lr}",
            f"Dataset Path: {dataset_path}",
            "\nEpoch Results:"
        ]
        
        for epoch in range(epochs):
            train_loss = train_model(model, train_loader, optimizer, criterion, device, scaler, task)
            val_metric = evaluate_model(model, val_loader, device, task)
            logger.info(f"Epoch {epoch+1}/{epochs}: Loss = {train_loss:.4f}, Metric = {val_metric:.4f}")
            results_log.append(f"Epoch {epoch+1}/{epochs}: Loss = {train_loss:.4f}, Metric = {val_metric:.4f}")
            scheduler.step(val_metric)
            
            if val_metric > best_metric:
                best_metric = val_metric
                patience_counter = 0
                torch.save(model.state_dict(), checkpoint_path)
                logger.info(f"Saved best model with metric {best_metric:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info("Early stopping triggered")
                    break
        
        model.load_state_dict(torch.load(checkpoint_path))
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"training_results_{task}_{tokenizer_name}_{model_name}_{timestamp}.txt"
        with open(results_file, "w") as f:
            f.write("\n".join(results_log))
        print(f"Training complete. Results saved to {results_file}")
    
    else:
        split = run_mode  
        dataset_eval = GLUEDataset(split=split, tokenizer=tokenizer, max_len=max_len, task=task, dataset_path=dataset_path)
        eval_loader = DataLoader(dataset_eval, batch_size=batch_size)
        
        if tokenizer_name == "ipa":
            vocab_size = len(tokenizer.token2idx)
        elif tokenizer_name in ["bpe", "unigram", "wordpiece"]:
            vocab_size = len(tokenizer.vocab)
        elif tokenizer_name == "char":
            vocab_size = len(tokenizer.char_list)
        elif tokenizer_name == "byte":
            vocab_size = 259
        else:
            sys.exit("Cannot determine vocab size for the selected tokenizer.")
        
        model = get_model(model_name, vocab_size, device, num_classes=num_classes)
        if not os.path.isfile(checkpoint_path):
            print(f"Checkpoint file {checkpoint_path} not found. Cannot run {run_mode} without a trained model.")
            sys.exit(1)
        model.load_state_dict(torch.load(checkpoint_path))
        metric = evaluate_model(model, eval_loader, device, task)
        logger.info(f"{run_mode.capitalize()} Metric = {metric:.4f}")
        print(f"{run_mode.capitalize()} complete. Metric = {metric:.4f}")

if __name__ == "__main__":
    main()
