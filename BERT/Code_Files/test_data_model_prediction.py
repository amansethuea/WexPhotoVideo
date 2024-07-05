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
from sklearn.metrics import precision_recall_fscore_support

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


class ModelPredictions(object):
    def __init__(self):
        if sys.platform.startswith("win"):
            self.path = os.getcwd() + "/../Model_Train_Data_Files/"
            self.df = pd.read_csv(self.path + 'cleaned_bert_final_3label.csv')
            self.df.Content = self.df.Content.astype(str)
            self.saved_models_path = os.getcwd() + "/../Saved_Models/"
            self.saved_model_file = self.saved_models_path + "model_64_512_8_0.3_db_3label.pth"
        elif sys.platform.startswith("darwin"):
            self.path = os.getcwd() + "/Model_Train_Data_Files/"
            self.df = pd.read_csv(self.path + 'cleaned_bert_final_3label.csv')
            self.df.Content = self.df.Content.astype(str)
            self.saved_models_path = os.getcwd() + "/Saved_Models/" 
            self.saved_model_file = self.saved_models_path + "model_64_512_8_0.3_db_3label.pth"
        elif sys.platform.startswith("linux"):
            self.path = os.getcwd() + "/../Model_Train_Data_Files/"
            self.df = pd.read_csv(self.path + 'cleaned_bert_final_3label.csv')
            self.df.Content = self.df.Content.astype(str)
            self.saved_models_path = os.getcwd() + "/../Saved_Models/"
            self.saved_model_file = self.saved_models_path + "model_64_512_8_0.3_db_3label.pth"

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
    
    def set_device(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device Found: {device}")
        return device
    
    def load_model(self):
        device = self.set_device()
        
        saved_model = self.sentiment_classifier
        # Map the model to the appropriate device (CPU in this case)
        saved_model.load_state_dict(torch.load(self.saved_model_file, map_location=device))
        saved_model = saved_model.to(device)
        saved_model = nn.DataParallel(saved_model) ## LIFESAVER!!
        print(saved_model.eval())
        print("Model loaded successfully")
        return saved_model.eval()
        
    
    def create_data_loader(self, df, tokenizer, max_len, batch_size):
        ds = ModelBuildTrainDataset(
            reviews=df['Content'].to_numpy(),
            targets=df['Sentiment'].to_numpy(),
            tokenizer=tokenizer,
            max_len=max_len    
        )

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        return DataLoader(ds, batch_size=batch_size, num_workers=0, collate_fn=data_collator)

    def get_predictions_and_Scores(self, dash=False):
        device = self.set_device()
        saved_model = self.load_model()
        self.MAX_LEN = 160
        test_data_loader = self.create_data_loader(self.df, self.tokenizer, self.MAX_LEN, self.BATCH_SIZE)
    
        all_preds = []
        all_targets = []
        print("Generating Predictions..")
        with torch.no_grad():
            for d in test_data_loader:
                input_ids = d["input_ids"].to(device)
                attention_mask = d["attention_mask"].to(device)
                targets = d["targets"].to(device)  # Assuming 'targets' column is 'labels'

                outputs = saved_model(input_ids=input_ids, attention_mask=attention_mask)
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs

                _, preds = torch.max(logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
                # Clear GPU cache
                torch.cuda.empty_cache()
        
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)

        print("Predictions generated successfully. Proceeding to calculate scores..")
        precision, recall, fscore, _ = precision_recall_fscore_support(all_targets, all_preds, average='weighted')
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1-score: {fscore:.2f}")
        print()
        print("Classification Report")
        print(classification_report(all_targets, all_preds))
        print()

        self.df['Predictions'] = all_preds
        
        label_mapping = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}

        # Map numerical labels to sentiment labels
        plot_names_true = [label_mapping[label] for label in self.df.Sentiment.value_counts().keys()]
        plot_values_true = self.df.Sentiment.value_counts().values

        plot_names_pred = [label_mapping[label] for label in self.df.Predictions.value_counts().keys()]
        plot_values_pred = self.df.Predictions.value_counts().values

        print(plot_names_true, plot_values_true)
        print()
        print(plot_names_pred, plot_values_pred)
        print()

        print("Actual vs Predicted Sentiment Histogram")
        fig = go.Figure(data=[
            go.Bar(name='True Labels', x=plot_names_true, y=plot_values_true),
            go.Bar(name='Predictions', x=plot_names_pred, y=plot_values_pred)
            ])
        # Change the bar mode
        fig.update_layout(barmode='group')
        if dash:
            return fig
        else:
            fig.show()

        # Replace values / labels in the 'Prediction' column with actual Sentiments labels
        self.df['Predictions'] = self.df['Predictions'].replace({0: 'Negative', 1: 'Neutral', 2: 'Positive'})
        
        print("Saving model to an external source..")
        self.df.to_csv(self.path + "final_bert_predictions.csv", index=False)
        print("Predictions saved successfully. Please check final_bert_predictions.csv")

if __name__ == "__main__":
    obj = ModelPredictions()
    obj.get_predictions_and_Scores()
    