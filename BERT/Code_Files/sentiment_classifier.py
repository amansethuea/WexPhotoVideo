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

# Other Libraries
import warnings
import time
import sys
import os

# Ignore all warnings
warnings.filterwarnings("ignore")


class SentimentClassifier(nn.Module):
    def __init__(self, n_classes):
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
        self.NAME = "db" #b for BERT, #rb for RoBERTa, #db for DistilBERT, #dbseq for DistilBERTforSequenceClassification
        self.BATCH_SIZE = 64
        self.MAX_LEN = 512 #changed from 512 for batch size 64, otherwise it would give error
        self.EPOCHS = 8
        self.LEARNING_RATE = 2e-5
        self.DROPOUT_RATE = 0.3 #[0.3, 0.4, 0.5] #considered dropout of 0.3 already

        self.MODEL_NAME = 'distilbert-base-uncased'
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.MODEL_NAME)
        
        # Random seed for reproducibilty
        self.RANDOM_SEED = 42
        np.random.seed(self.RANDOM_SEED)
        torch.manual_seed(self.RANDOM_SEED)

        super(SentimentClassifier, self).__init__()
        self.distilbert = DistilBertModel.from_pretrained(self.MODEL_NAME)
        self.drop = nn.Dropout(p=self.DROPOUT_RATE)
        #self.additional_layer = nn.Linear(2, 64)  # Additional features layer
        self.out = nn.Linear(self.distilbert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs[0][:, 0, :]  # Use the [CLS] token embedding
        output = self.drop(pooled_output)
        return self.out(output)


if __name__ == "__main__":
    obj = SentimentClassifier()