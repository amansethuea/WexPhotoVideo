# Dataset Libraries
import pandas as pd

# Preprocessing Libraries
import contractions
import spacy
import demoji

import warnings
warnings.filterwarnings('ignore')

# Other Libraries
import os
import re
import sys


class CleanAndPreprocess(object):
  def __init__(self):
    self.nlp = spacy.load("en_core_web_lg") # Invoking nlp large pre-trained model

    if sys.platform.startswith("win"):
        self.path = os.getcwd() + "/../Model_Train_Data_Files/"
        try:
          self.df = pd.read_csv(self.path + 'fetched_test_reviews.csv') # File that needs to be cleaned before testing ML model
        except FileNotFoundError:
           print("bert_final_3label.csv might not be formed yet. Waiting for file to be formed first.")
    elif sys.platform.startswith("darwin"):
       self.path = os.getcwd() + "/Model_Train_Data_Files/"
       try:
        self.df = pd.read_csv(self.path + 'fetched_test_reviews.csv') # File that needs to be cleaned before testing ML model
       except FileNotFoundError:
          print("bert_final_3label.csv might not be formed yet. Waiting for file to be formed first.")
    elif sys.platform.startswith("linux"):
        self.path = os.getcwd() + "/../Model_Train_Data_Files/"
        try:
          self.df = pd.read_csv(self.path + 'fetched_test_reviews.csv') # File that needs to be cleaned before testing ML model
        except FileNotFoundError:
           print("bert_final_3label.csv might not be formed yet. Waiting for file to be formed first.")

  def create_sentiment_labels(self, rating):
    rating = int(rating)

    # Convert to class
    if rating <= 2:
        return 0
    elif rating == 3:
        return 1
    else:
        return 2

  def apply_sentiment_to_test_data(self):
    # Apply to the dataset
    self.df['Sentiment'] = self.df['Rating'].apply(self.create_sentiment_labels)

  def check_mandatory_columns(self):
    if sys.platform.startswith("win"):
        self.path = os.getcwd() + "/../Model_Train_Data_Files/"
        self.df = pd.read_csv(self.path + 'fetched_test_reviews.csv') # File that needs to be cleaned before testing ML model
    elif sys.platform.startswith("darwin"):
      self.path = os.getcwd() + "/Model_Train_Data_Files/"
      self.df = pd.read_csv(self.path + 'fetched_test_reviews.csv') # File that needs to be cleaned before testing ML model
    elif sys.platform.startswith("linux"):
        self.path = os.getcwd() + "/../Model_Train_Data_Files/"
        self.df = pd.read_csv(self.path + 'fetched_test_reviews.csv') # File that needs to be cleaned before testing ML model

    columns_to_check = ['Rating', 'Content', 'Sentiment']
    # Check if the columns exist in the DataFrame
    existing_columns = [col for col in columns_to_check if col in self.df.columns]
    missing_columns = set(columns_to_check) - set(existing_columns)
    if missing_columns:
      if 'Sentiment' in missing_columns:
        print("Sentiment column not found. Creating sentiment labels")
        self.apply_sentiment_to_test_data()
        existing_columns = [col for col in columns_to_check if col in self.df.columns]
        print("Sentiment labels created successfully")
        return existing_columns
      else:
        print(f"Following mandatory columns are missing from the DataFrame: {missing_columns}")
        return False
    else:
      print("Found all mandatory columns. Proceeding..")
      return existing_columns

  def find_and_drop_missing_records(self):
        existing_columns = self.check_mandatory_columns()
        if existing_columns:
          # Check for missing values in the existing columns
          missing_values = self.df[existing_columns].isnull().sum()
          print("Missing values before dropping rows:")
          print(missing_values)
          df_cleaned = self.df.dropna(subset=existing_columns)
          # Verify that there are no more missing values in those columns
          missing_values_after = df_cleaned[existing_columns].isnull().sum()
          print("\nMissing values after dropping rows:")
          print(missing_values_after)
          print("\nShape of the DataFrame before dropping rows:", self.df.shape)
          print("Shape of the DataFrame after dropping rows:", df_cleaned.shape)
        else:
          print("Mandatory columns not found in test data file. Exiting..")
          exit(0)

  def reviews_lowercasing(self):
    self.df['Content'] = self.df['Content'].str.lower()
    print("All reviews lowercased successfully")

  def find_emoji(self, text):
    emoj = re.compile("["
    u"\U0001F600-\U0001F64F"  # emoticons
    u"\U0001F300-\U0001F5FF"  # symbols & pictographs
    u"\U0001F680-\U0001F6FF"  # transport & map symbols
    u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
    u"\U00002500-\U00002BEF"  # chinese char
    u"\U00002702-\U000027B0"
    u"\U000024C2-\U0001F251"
    u"\U0001f926-\U0001f937"
    u"\U00010000-\U0010ffff"
    u"\u2640-\u2642"
    u"\u2600-\u2B55"
    u"\u200d"
    u"\u23cf"
    u"\u23e9"
    u"\u231a"
    u"\ufe0f"  # dingbats
    u"\u3030"
                  "]+", re.UNICODE)

    if len(str(text))<2:
        emoji_dict = demoji.findall(text)
        for key, value in emoji_dict.items():
            return value
    else:
        demoji.findall(str(text))
        values_list  = []
        for _, values in demoji.findall(str(text)).items():
            values_list.append(values)

        if "star" in values_list:
            return re.sub(emoj, 'star', str(text))

        emoji_removed = re.sub(emoj, '', str(text))
        return emoji_removed

  def emoji_remove_replace(self):
    self.df["cleaned_emoji"] = self.df["Content"].apply(self.find_emoji)
    emojis_found = self.df["cleaned_emoji"].isnull().sum()
    print(f"Emoji Records to be removed: {emojis_found}")
    self.df = self.df.dropna()
    print("Emojis, Symbols, Chinese Chracters, Flags etc. removed successfully")

  def find_contractions(self, text):
    # Removing Contractions
    expanded_text = []
    for word in text.split():
        expanded_text.append(contractions.fix(word))

    expanded_text = ' '.join(expanded_text)
    return expanded_text

  def contractions_removal(self):
    self.df["contractions_removed"] = self.df["cleaned_emoji"].apply(self.find_contractions)
    print("Contractions removed successfully")

  def find_brackets_links_punctuations(self, text):
    text = text.lower() #lowercases the string
    text = re.sub(r"\.{2,}",".", text) #removes trailing ...
    text = re.sub(r"\!{2,}","!", text) #removes trailing !!!
    text = re.sub(r"\+{2,}","", text) #removes trailing +++
    text = re.sub(r"\?{2,}","?", text) #removes trailing ???
    text = re.sub(r"\_{2,}"," ", text) #removes trailing ___
    text = re.sub(r"[\[\]]", '', text) #removes [ or ]
    text = re.sub(r"\(.*?\)", '', text) #removes (text)
    text = re.sub(r'["“”]', '', text) #removes quotation maeks ""
    text = re.sub(r'(\w)\1+', r'\1\1', text) #removes repeating characters and repaces it with 2 occurances
    text = re.sub(r'\*', '', text) #removes *
    text = re.sub(r"\s*([.])\s*", '. ', text) #removes whitespaces for .
    text = re.sub(r"\s*([,])\s*", ', ', text) #removes whitespaces for ,
    text = re.sub(r'https?://\S+|www\.\S+', '', text) #removes links

    return text

  def brackets_links_punctuations_removal(self):
    self.df["regex"] = self.df["contractions_removed"].apply(self.find_brackets_links_punctuations)
    self.df = self.df.drop(self.df.loc[self.df["regex"].str.len() < 2].index)
    self.df = self.df.dropna()
    print("Brackets, Links, Numbers in text or Punctuations removed succesfully")

  def find_unwanted_spaces(self, text):
    return text.strip()

  def strip_unwanted_spaces(self):
    self.df["regex"] = self.df["regex"].apply(self.find_unwanted_spaces)
    self.df = self.df.drop(self.df.loc[self.df["regex"].str.len() < 2].index)
    self.df.loc[self.df["regex"].str.len() < 3].regex.value_counts()
    self.df.loc[(self.df["regex"] != "ok") & (self.df["regex"].str.len() < 3)].value_counts()
    self.df = self.df.drop(self.df.loc[(self.df["regex"] != "ok") & (self.df["regex"].str.len() < 3)].index)
    self.df[self.df["regex"].str.len() < 3]
    print("Unwanted trailing spaces removed successfully")

  # Remove more meaningless special characters
  def find_special_char(self, text):
    if len(text)<4:
      text = text.strip()
      text = re.sub(r'[^A-Za-z0-9\s]', '', text)
    return text

  def special_char_removal(self):
    self.df['regex'] = self.df['regex'].apply(self.find_special_char)
    self.df.loc[self.df["regex"].str.len() < 10].regex.value_counts()
    self.df['regex'] = self.df['regex'].astype(str)
    self.df.drop(columns = ['cleaned_emoji',	'contractions_removed'], inplace = True)
    print("Special Characters removed successfully")

  def find_drop_duplicates(self, cleaned=False):
    if cleaned:
      print("\nShape of the DataFrame before dropping duplicate rows:", self.df.shape)
      duplicate = self.df[self.df.duplicated('regex')]
      self.df = self.df.drop_duplicates(subset=['Content', 'regex'])
      print("Duplicate records removed successfully from pre-processed data")
      print("\nShape of the DataFrame after dropping duplicate rows:", self.df.shape)
    else:
      print("\nShape of the DataFrame before dropping duplicate rows:", self.df.shape)
      duplicate = self.df[self.df.duplicated('Content')]
      self.df = self.df.drop_duplicates(subset=['Content'])
      print("Duplicate records removed successfully from raw data")
      print("\nShape of the DataFrame after dropping duplicate rows:", self.df.shape)

  def drop_regex_column(self):
    # Replace Content with regex column
    self.df.drop(columns = ['Content'], inplace = True)
    self.df = self.df.rename(columns={'regex': 'Content'})
    print("Regex column dropped successfully")

  def save_preprocessed_test_data(self):
    if sys.platform.startswith("win"):
       cleaned_preprocessed_data = os.getcwd() + "/../Model_Train_Data_Files/cleaned_bert_final_3label.csv"
    elif sys.platform.startswith("darwin"):
       cleaned_preprocessed_data = os.getcwd() + "/Model_Train_Data_Files/cleaned_bert_final_3label.csv"
    elif sys.platform.startswith("linux"):
       cleaned_preprocessed_data = os.getcwd() + "/../Model_Train_Data_Files/cleaned_bert_final_3label.csv"
  

    self.df.to_csv(cleaned_preprocessed_data, index=False)
    print("Test Data Pre-processed and saved successfully. Please check cleaned_bert_final_3label.csv")

  def main(self, model_type="ML"):
    if model_type.upper() in ["ML", "MACHINE LEARNING", "SUPERVISED"]:
      print()
      print("START: Initiating Pre-processing for ML modelling")
      print("######################### STEP 1 #########################")
      print("Drop Missing records")
      self.find_and_drop_missing_records()
      print("######################### STEP 2 #########################")
      print("Lowercase the reviews text")
      self.reviews_lowercasing()
      print("######################### STEP 3 #########################")
      print("Drop Duplicate records from Raw Data ")
      self.find_drop_duplicates(cleaned=False)
      print("######################### STEP 4 #########################")
      print("Remove Emojis and other Symbols")
      self.emoji_remove_replace()
      print("######################### STEP 5 #########################")
      print("Remove Contractions")
      self.contractions_removal()
      print("######################### STEP 6 #########################")
      print("Remove Brackets, Links and Punctuations")
      self.brackets_links_punctuations_removal()
      print("######################### STEP 7 #########################")
      print("Remove unwanted trailing spaces")
      self.strip_unwanted_spaces()
      print("######################### STEP 8 #########################")
      print("Remove Special Characters")
      self.special_char_removal()
      print("######################### STEP 9 #########################")
      print("Drop Duplicate records from Pre-processed Data")
      self.find_drop_duplicates(cleaned=True)
      print("######################### STEP 10 #########################")
      print("Drop all other unwanted columns")
      self.drop_regex_column()
      print("Save Pre-processed data")
      print("######################### STEP 11 #########################")
      self.save_preprocessed_test_data()
      print("END: Data cleaned and pre-processed.")
      print()

    elif model_type in ["BERT"]:
      print()
      print("START: Initiating Pre-processing for BERT modelling")
      print("######################### STEP 1 #########################")
      print("Drop Missing records")
      self.find_and_drop_missing_records()
      print("######################### STEP 2 #########################")
      print("Lowercase the reviews text")
      self.reviews_lowercasing()
      print("######################### STEP 3 #########################")
      print("Drop Duplicate records from Raw Data ")
      self.find_drop_duplicates(cleaned=False)
      print("######################### STEP 4 #########################")
      print("Remove Emojis and other Symbols")
      self.emoji_remove_replace()
      print("######################### STEP 5 #########################")
      print("Remove Contractions")
      self.contractions_removal()
      print("######################### STEP 6 #########################")
      print("Remove Brackets, Links and Punctuations")
      self.brackets_links_punctuations_removal()
      print("######################### STEP 7 #########################")
      print("Remove unwanted trailing spaces")
      self.strip_unwanted_spaces()
      print("######################### STEP 8 #########################")
      print("Remove Special Characters")
      self.special_char_removal()
      print("######################### STEP 9 #########################")
      print("Drop Duplicate records from Pre-processed Data")
      self.find_drop_duplicates(cleaned=True)
      print("######################### STEP 10 #########################")
      print("Drop all other unwanted columns")
      self.drop_regex_column()
      print("Save Pre-processed data")
      print("######################### STEP 11 #########################")
      self.save_preprocessed_test_data()
      print("END: Data cleaned and pre-processed.")
      print()
    else:
      print(f"Invalid model type: {model_type}. Please provide correct model type.")


if __name__ == "__main__":
  obj = CleanAndPreprocess()
  obj.main(model_type="BERT")
