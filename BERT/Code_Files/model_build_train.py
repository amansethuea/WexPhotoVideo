# Dataset Libraries
import pandas as pd
import numpy as np

# Preprocessing Libraries
import re
import contractions
from collections import defaultdict
import demoji

# Visualization Libraries
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# ML and Evaluation Libraries
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, train_test_split, GridSearchCV

# Torch ML Libraries
import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from transformers import RobertaModel, RobertaTokenizer
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments, DistilBertConfig
from transformers import DistilBertModel, DistilBertTokenizer
from tqdm.auto import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import DataCollatorWithPadding
from sklearn.utils.class_weight import compute_class_weight
from sentiment_classifier import SentimentClassifier

# Other Libraries
import warnings
import time
import sys
import os

# Ignore all warnings
warnings.filterwarnings("ignore")


class ModelBuildTrainDataset(Dataset):
    def __init__(self, reviews, targets, tokenizer, max_len):
        self.reviews = reviews
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, item):
        review = str(self.reviews[item])
        target = self.targets[item]

        encoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long)
        }


class ModelBuildTrain:
    def __init__(self):
        if sys.platform.startswith("win"):
            self.path = os.getcwd() + "/../Model_Train_Data_Files/"
            self.df = pd.read_csv(self.path + 'cleaned_bert_final_3label.csv')
            self.saved_models_path = os.getcwd() + "/../Saved_Models/"
        elif sys.platform.startswith("darwin"):
            self.path = os.getcwd() + "/Model_Train_Data_Files/"
            self.df = pd.read_csv(self.path + 'cleaned_bert_final_3label.csv')
            self.saved_models_path = os.getcwd() + "/Saved_Models/" 
        elif sys.platform.startswith("linux"):
            self.path = os.getcwd() + "/../Model_Train_Data_Files/"
            self.df = pd.read_csv(self.path + 'cleaned_bert_final_3label.csv')
            self.saved_models_path = os.getcwd() + "/../Saved_Models/"

        # Setting the Batch Size, Max Length, Learning Rate, Dropout Rate
        self.NAME = "db"
        self.BATCH_SIZE = 64
        self.MAX_LEN = 512
        self.EPOCHS = 8
        self.LEARNING_RATE = 2e-5
        self.DROPOUT_RATE = 0.3

        self.MODEL_NAME = 'distilbert-base-uncased'
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.MODEL_NAME)

        self.sentiment_classifier = SentimentClassifier(n_classes=3)

    def train_val_test_split(self):
        # Create train, validation, and test splits (90/5/5 split)
        df_train, df_temp = train_test_split(self.df, test_size=0.1, random_state=42, stratify=self.df['Sentiment'])
        df_val, df_test = train_test_split(df_temp, test_size=0.5, random_state=42, stratify=df_temp['Sentiment'])
        print(len(df_train), len(df_val), len(df_test))
        return df_train, df_temp, df_val, df_test
    
    def create_data_loader(self, df, tokenizer, max_len, batch_size):
        ds = ModelBuildTrainDataset(
            reviews=df['Content'].to_numpy(),
            targets=df['Sentiment'].to_numpy(),
            tokenizer=tokenizer,
            max_len=max_len    
        )

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        return DataLoader(ds, batch_size=batch_size, num_workers=0, collate_fn=data_collator)
    
    def train_epoch(self, model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
        model = model.train()
        correct_predictions = 0
        total_loss = 0

        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs, targets)

            _, preds = torch.max(outputs, dim=1)
            correct_predictions += torch.sum(preds == targets)
            total_loss += loss.item()

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        return correct_predictions.double() / n_examples, total_loss / n_examples
    
    def eval_model(self, model, data_loader, loss_fn, device, n_examples):
        model = model.eval()
        correct_predictions = 0
        total_loss = 0

        with torch.no_grad():
            for d in data_loader:
                input_ids = d["input_ids"].to(device)
                attention_mask = d["attention_mask"].to(device)
                targets = d["targets"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs, targets)

                _, preds = torch.max(outputs, dim=1)
                correct_predictions += torch.sum(preds == targets)
                total_loss += loss.item()
        model.train()
        return correct_predictions.double() / n_examples, total_loss / n_examples

    def set_device(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return device

    def dataset_to_store_results(self):
        results_df = pd.DataFrame(columns=['Epochs', 'Batch Size', 'Learning Rate', 'Dropout Rate', 'Train Accuracy', 'Train Loss', 'Val Accuracy', 'Val Loss','Test Accuracy', 'Max Length'])
        return results_df
    
    def main(self):
        df_train, df_temp, df_val, df_test = self.train_val_test_split()
        
        train_data_loader = self.create_data_loader(df_train, self.tokenizer, self.MAX_LEN, self.BATCH_SIZE)
        val_data_loader = self.create_data_loader(df_val, self.tokenizer, self.MAX_LEN, self.BATCH_SIZE)
        test_data_loader = self.create_data_loader(df_test, self.tokenizer, self.MAX_LEN, self.BATCH_SIZE)

        # Initialize the model, optimizer, and scheduler
        model = self.sentiment_classifier

        # Load the model
        device = self.set_device()
        model = model.to(device)
        model = nn.DataParallel(model)

        optimizer = optim.AdamW(model.parameters(), lr=self.LEARNING_RATE, weight_decay=0.01)
        total_steps = len(train_data_loader) * self.EPOCHS

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=100,
            num_training_steps=total_steps
            )

        loss_fn = nn.CrossEntropyLoss().to(device)

        # Model name and output path
        PATH = self.saved_models_path + f"/model_{self.BATCH_SIZE}_{self.MAX_LEN}_{self.EPOCHS}_{self.DROPOUT_RATE}_{self.NAME}_3label.pth"

        # Early stopping parameters
        early_stopping_patience = 3
        best_val_acc = 0.0
        patience_counter = 0

        self.progress_bar = tqdm(range(total_steps))

        # Training loop with history tracking and early stopping
        history = defaultdict(list)
        learning_rates = []

        # Initialize a list to store each experiment's results as DataFrames
        results_list = []

        print(f"Batch Size: {self.BATCH_SIZE}")
        print()

        for epoch in range(self.EPOCHS):
            print(f"Epoch: {epoch + 1}/{self.EPOCHS}")
            print("-" * 10)

            start_time = time.time()

            train_acc, train_loss = self.train_epoch(
                model,
                train_data_loader,
                loss_fn,
                optimizer,
                device,
                scheduler,
                len(df_train)
            )

            val_acc, val_loss = self.eval_model(
                model,
                val_data_loader,
                loss_fn,
                device,
                len(df_val)
            )

            end_time = time.time()
            epoch_duration = end_time - start_time

            current_lr = optimizer.param_groups[0]['lr']
            learning_rates.append(current_lr)

            history['train_acc'].append(train_acc.item())
            history['train_loss'].append(train_loss)
            history['val_acc'].append(val_acc.item())
            history['val_loss'].append(val_loss)
            history['epoch_time'].append(epoch_duration)

            print(f"Train loss {train_loss} accuracy {train_acc}")
            print(f"Val   loss {val_loss} accuracy {val_acc}")
            print(f"Learning Rate: {current_lr}")
            print(f"Epoch duration: {epoch_duration:.2f} seconds")
            print()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save(model.module.state_dict(), PATH)
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print("Early stopping triggered")
                    break


if __name__ == "__main__":
    obj = ModelBuildTrain()
    obj.main()
