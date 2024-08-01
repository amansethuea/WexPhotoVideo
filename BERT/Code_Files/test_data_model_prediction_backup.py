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
from sklearn.metrics import classification_report, precision_recall_fscore_support
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import DataCollatorWithPadding

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
        elif sys.platform.startswith("darwin"):
            self.path = os.getcwd() + "/Model_Train_Data_Files/"
            self.df = pd.read_csv(self.path + 'cleaned_bert_final_3label.csv')
            self.df.Content = self.df.Content.astype(str)
            self.saved_models_path = os.getcwd() + "/Saved_Models/"
        elif sys.platform.startswith("linux"):
            self.path = os.getcwd() + "/../Model_Train_Data_Files/"
            self.df = pd.read_csv(self.path + 'cleaned_bert_final_3label.csv')
            self.df.Content = self.df.Content.astype(str)
            self.saved_models_path = os.getcwd() + "/../Saved_Models/"

        self.BATCH_SIZE = 64
        self.MAX_LEN = 512
        self.MODEL_NAME = 'distilbert-base-uncased'
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.MODEL_NAME)

        # Paths to the saved models
        self.model_paths = {

            0: os.path.join(self.saved_models_path, "model_64_512_8_0.3_db_24_negative.pth"),
            1: os.path.join(self.saved_models_path, "model_64_512_8_0.3_db_42_neutral.pth"),
            2: os.path.join(self.saved_models_path, "model_64_512_4_0.3_db_42_positive.pth")

        }


    def set_device(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device Found: {device}")
        return device

    def initialize_model(self, model_path):
        device = self.set_device()
        model = DistilBertForSequenceClassification.from_pretrained(self.MODEL_NAME, num_labels=3)
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        model = model.to(device)
        model = nn.DataParallel(model)
        model.eval()
        return model

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
         # Initialize models
        self.models = {label: self.initialize_model(path) for label, path in self.model_paths.items()}
        time.sleep(5)

        device = self.set_device()
        self.MAX_LEN = 512
        test_data_loader = self.create_data_loader(self.df, self.tokenizer, self.MAX_LEN, self.BATCH_SIZE)
    
        all_preds = []
        all_targets = []
        print("Generating Predictions..")
        
        # Initialize predictions list with NaNs or placeholders
        all_preds_placeholder = np.full(len(self.df), -1)

        with torch.no_grad():
            for d in test_data_loader:
                input_ids = d["input_ids"].to(device)
                attention_mask = d["attention_mask"].to(device)
                targets = d["targets"].to(device)

                # Predict for each sample
                for i in range(len(input_ids)):
                    sentiment_label = self.df.iloc[i]['Sentiment']
                    model = self.models[sentiment_label]
                    
                    outputs = model(input_ids=input_ids[i:i+1], attention_mask=attention_mask[i:i+1])
                    logits = outputs.logits
                    
                    _, preds = torch.max(logits, dim=1)
                    
                    all_preds_placeholder[i] = preds.cpu().numpy()

                all_targets.extend(targets.cpu().numpy())

                # Clear GPU cache
                torch.cuda.empty_cache()

        all_preds = np.array(all_preds_placeholder)
        all_targets = np.array(all_targets)

        # Ensure predictions length matches DataFrame length
        if len(all_preds) != len(self.df):
            print(f"Warning: Number of predictions ({len(all_preds)}) does not match number of records in DataFrame ({len(self.df)})")
            if len(all_preds) > len(self.df):
                all_preds = all_preds[:len(self.df)]  # Adjust predictions to match DataFrame length
            elif len(all_preds) < len(self.df):
                all_targets = all_targets[:len(all_preds)]  # Adjust targets if fewer predictions are available

        print("Predictions generated successfully. Proceeding to calculate scores..")
        precision, recall, fscore, _ = precision_recall_fscore_support(all_targets, all_preds, average='weighted')
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1-score: {fscore:.2f}")
        print()
        print("Classification Report")
        print(classification_report(all_targets, all_preds))
        print()

        # Ensure DataFrame length matches length of predictions
        if len(all_preds) == len(self.df):
            self.df['Predictions'] = all_preds
        else:
            print("Error: The length of predictions does not match the length of DataFrame.")

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
            # Replace values / labels in the 'Prediction' column with actual Sentiments labels
            self.df['Predictions'] = self.df['Predictions'].replace({0: 'Negative', 1: 'Neutral', 2: 'Positive'})
        
            print("Saving model to an external source..")
            self.df.to_csv(self.path + "final_bert_predictions.csv", index=False)
            print("Predictions saved successfully. Please check final_bert_predictions.csv")
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