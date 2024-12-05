#!/usr/bin/env python
# coding: utf-8

# In[ ]:


rimary Columns for Emotion Detection: title, selftext (if available)
Secondary Columns: Text (created by concatenating title and selftext)
Identification Columns: id, Label (not used directly in emotion inference but useful for referencing or grouping data)


# In[1]:


import pandas as pd

# File path
file_path = r"C:\Users\Hazel\Downloads\Sentiment Analysis files\Sentimentanalysisdata\redditdatafromjson.csv"

# Load the dataset
df = pd.read_csv(file_path, encoding='utf-8', low_memory=False)

# Check if 'title' column exists in the dataset
if 'title' in df.columns:
    # Count the number of rows with a non-empty title
    rows_with_title = df[df['title'].notna()]
    print(f"Total rows with a title: {rows_with_title.shape[0]}")
else:
    print("The dataset does not contain a 'title' column.")


# In[2]:


import pandas as pd
import os

# File path
file_path = r"C:\Users\Hazel\Downloads\Sentiment Analysis files\Sentimentanalysisdata\redditdatafromjson.csv"

# Load the dataset
df = pd.read_csv(file_path, encoding='utf-8', low_memory=False)

# Display initial size in MB
initial_size_mb = os.path.getsize(file_path) / (1024 * 1024)
print(f"Initial dataset size: {initial_size_mb:.2f} MB")

# Check if 'title' column exists in the dataset
if 'title' in df.columns:
    # Drop all other columns except for 'title', 'selftext', and 'Text'
    df_cleaned = df[['title', 'selftext', 'Text']]

    # Count the number of rows with a non-empty 'title'
    rows_with_title = df_cleaned[df_cleaned['title'].notna()]
    print(f"Total rows with a title: {rows_with_title.shape[0]}")

    # Optionally, save the cleaned file
    output_cleaned_file_path = r"C:\Users\Hazel\Downloads\Sentiment Analysis files\cleaned_reddit_data.csv"
    df_cleaned.to_csv(output_cleaned_file_path, index=False, encoding='utf-8')
    print(f"Cleaned dataset saved to: {output_cleaned_file_path}")

else:
    print("The dataset does not contain a 'title' column.")


# Now chosing which column has a data, and if the row has more than 1 data per column, only pick one and output into the final column
# 

# In[6]:


import pandas as pd
import re

# File path for the cleaned data
file_path = r"C:\Users\Hazel\Downloads\Sentiment Analysis files\cleaned_reddit_data.csv"

# Load the dataset
df = pd.read_csv(file_path, encoding='utf-8', low_memory=False)

# Helper function to clean text while preserving emojis
def clean_text(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "]+", flags=re.UNICODE)
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r"[^a-zA-Z\s" + emoji_pattern.pattern + "]", '', text)  # Keep letters, spaces, and emojis
    text = re.sub(r"\s+", ' ', text).strip()  # Remove extra spaces
    return text

# Function to combine text columns, prioritizing columns with values
def combine_text(row):
    title_text = row['title'] if pd.notna(row['title']) else ""
    selftext_text = row['selftext'] if pd.notna(row['selftext']) else ""
    text_column_text = row['Text'] if pd.notna(row['Text']) else ""
    
    # If title and selftext both have text, check for emotions and choose one
    if title_text.strip() and selftext_text.strip():
        return title_text  # You can change to selftext_text if needed
    
    # If title has content, return it
    if title_text.strip():
        return title_text
    
    # If selftext has content, return it
    if selftext_text.strip():
        return selftext_text
    
    # If Text column has content, return it
    if text_column_text.strip():
        return text_column_text
    
    return ""  # If none have content, return empty string

# Apply the function to combine the text columns
df['Combined_Text'] = df.apply(combine_text, axis=1)

# Remove rows where 'Combined_Text' is empty (if needed)
df = df[df['Combined_Text'].str.strip() != '']

# Save the combined data to a new CSV
output_file_path = r"C:\Users\Hazel\Downloads\Sentiment Analysis files\combined_reddit_data.csv"
df[['Combined_Text']].to_csv(output_file_path, index=False, encoding='utf-8')

print(f"Combined dataset saved to: {output_file_path}")


# Now breaking into smaller chunks 

# In[7]:


import pandas as pd
import os

# File path for the combined data
input_file_path = r"C:\Users\Hazel\Downloads\Sentiment Analysis files\combined_reddit_data.csv"

# Output folder path
output_folder = r"C:\Users\Hazel\Downloads\Sentiment Analysis files\split_files"

# Make sure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Load the dataset
df = pd.read_csv(input_file_path, encoding='utf-8')

# Rename the column to 'Text'
df = df.rename(columns={'Combined_Text': 'Text'})

# Total size in KB and chunk size in KB (50,000 KB = 50 MB)
total_size_kb = 1150519  # Given total size
chunk_size_kb = 50000    # 50 MB

# Initialize variables
total_rows = len(df)
total_size_bytes = total_size_kb * 1024  # Convert KB to bytes
chunk_size_bytes = chunk_size_kb * 1024  # Convert KB to bytes
file_count = 0
current_size = 0
current_chunk = []

# Iterate through the dataframe and split into chunks
for index, row in df.iterrows():
    # Add the row to the current chunk
    current_chunk.append(row)
    current_size += row.memory_usage(deep=True)
    
    # If the chunk size exceeds the limit, write to a new file
    if current_size >= chunk_size_bytes:
        # Create a DataFrame from the current chunk
        chunk_df = pd.DataFrame(current_chunk)
        
        # Save the chunk to a CSV file
        output_file = os.path.join(output_folder, f"split_part_{file_count + 1}.csv")
        chunk_df.to_csv(output_file, index=False, encoding='utf-8')
        
        # Reset the chunk and size counter for the next chunk
        current_chunk = []
        current_size = 0
        file_count += 1

# Handle any remaining rows after the loop
if current_chunk:
    chunk_df = pd.DataFrame(current_chunk)
    output_file = os.path.join(output_folder, f"split_part_{file_count + 1}.csv")
    chunk_df.to_csv(output_file, index=False, encoding='utf-8')
    file_count += 1

print(f"File splitting completed. {file_count} files created.")


# In[8]:


import pandas as pd
import torch
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load the tokenizer and model for PyTorch
model_name = 'bhadresh-savani/distilbert-base-uncased-emotion'

# Load the tokenizer and model
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()  # Set model to evaluation mode
except Exception as e:
    print(f"Error loading the model '{model_name}'. Error: {e}")
    raise e

# Helper function to clean text while preserving emojis
def clean_text(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "]+", flags=re.UNICODE)
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r"[^a-zA-Z\s" + emoji_pattern.pattern + "]", '', text)  # Keep letters, spaces, and emojis
    text = re.sub(r"\s+", ' ', text).strip()  # Remove extra spaces
    return text

# Rule-based fallback method for emotion detection
def infer_emotion_using_keywords(text):
    emotion_keywords = {
        'anger': ['angry', 'hate', 'furious', 'annoyed', 'rage', 'irritated', 'mad'],
        'disgust': ['disgust', 'gross', 'nauseous', 'revolting', 'detest', 'repulsive'],
        'fear': ['fear', 'terror', 'scared', 'panic', 'afraid', 'anxious', 'worried'],
        'joy': ['happy', 'joy', 'delight', 'bless', 'grateful', 'love', 'excited'],
        'neutral': ['ok', 'fine', 'neutral', 'average', 'normal', 'standard'],
        'sadness': ['sad', 'cry', 'pain', 'hurt', 'loss', 'funeral', 'depressed', 'unhappy'],
        'shame': ['ashamed', 'shame', 'embarrassed', 'guilt', 'humiliated'],
        'surprise': ['surprise', 'unexpected', 'shock', 'amazed', 'wonder', 'astonished']
    }
    text = text.lower()
    for emotion, keywords in emotion_keywords.items():
        if any(keyword in text for keyword in keywords):
            return emotion
    return 'neutral'

# Batch inference for emotions with fallback
def infer_emotions_batch(texts, batch_size=32):
    emotions = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            inputs = tokenizer(batch, max_length=512, truncation=True, padding=True, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
            logits = outputs.logits
            predicted_labels = torch.argmax(logits, axis=1)
            batch_emotions = [model.config.id2label[label.item()] for label in predicted_labels]
        except Exception as e:
            print(f"Error inferring emotions for batch: {e}. Using fallback method.")
            batch_emotions = [infer_emotion_using_keywords(text) for text in batch]
        emotions.extend(batch_emotions)
    return emotions

# Process dataset
def process_dataset(file_path, text_column, batch_size=32):
    df = pd.read_csv(file_path, encoding='utf-8', low_memory=False)
    df['Text'] = df[text_column].fillna('').apply(clean_text)
    df['Emotion'] = infer_emotions_batch(df['Text'].tolist(), batch_size=batch_size)
    return df[['Emotion', 'Text']]

# Process the uploaded file
file_path = r"C:\Users\Hazel\Downloads\Sentiment Analysis files\split_files\split_part_1.csv"  # Update path
reddit_data = process_dataset(file_path, text_column='Text', batch_size=32)

# Save the processed data to the specified output folder
output_file_path = r"C:\Users\Hazel\Downloads\Sentiment Analysis files\split_files\output\formatted_emotion_data_with_emojis.csv"  # New output path
reddit_data.to_csv(output_file_path, index=False, encoding='utf-8')

print(f"Formatted dataset saved to: {output_file_path}")


# Since this file wont work in my local, I have run this code in Google Colab, and have used T4-GPU and on the code, it has combined the environment for CPU and GPU running

# First thing, stop words were removed to reduce the file, stop words from sklearn used, code reference below

# In[ ]:


import pandas as pd
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Load the dataset from your Google Drive
file_path = '/content/drive/MyDrive/Colab Notebooks/split_part_2.csv'  # Adjust to your actual file path
df = pd.read_csv(file_path, encoding='utf-8', low_memory=False)

# Helper function to clean text while preserving emojis and removing stopwords
def clean_text_with_stopwords(text):
    # Remove URLs and non-alphabetic characters except emojis
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "]+", flags=re.UNICODE)
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r"[^a-zA-Z\s" + emoji_pattern.pattern + "]", '', text)  # Keep letters, spaces, and emojis
    text = re.sub(r"\s+", ' ', text).strip()  # Remove extra spaces

    # Remove stopwords and track them
    original_words = text.split()
    filtered_words = [word for word in original_words if word.lower() not in ENGLISH_STOP_WORDS]
    removed_stopwords = set(original_words) - set(filtered_words)
    
    # Return cleaned text and removed stopwords
    return ' '.join(filtered_words), removed_stopwords

# Apply stopword removal and clean text
df['Cleaned_Text'], df['Removed_Stopwords'] = zip(*df['Text'].fillna('').apply(clean_text_with_stopwords))

# Print removed stopwords for the first few rows to verify
print("Removed stopwords for the first row:", df['Removed_Stopwords'].iloc[0])

# Display the first few rows of the dataframe with the 'Cleaned_Text' and 'Removed_Stopwords' columns
df[['Text', 'Cleaned_Text', 'Removed_Stopwords']].head()


# In[ ]:


Then have proceeded in doing emotion detection using distilbert


# In[ ]:


import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Specify the model name
model_name = 'bhadresh-savani/distilbert-base-uncased-emotion'

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Check if GPU is available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)  # Move the model to the appropriate device

# Function to perform emotion inference
def infer_emotions_batch(texts, batch_size=32):
    emotions = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            # Tokenize the batch and move inputs to the same device as the model
            inputs = tokenizer(batch, max_length=256, truncation=True, padding=True, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            logits = outputs.logits
            predicted_labels = torch.argmax(logits, axis=1)
            batch_emotions = [model.config.id2label[label.item()] for label in predicted_labels]
        except Exception as e:
            print(f"Error inferring emotions for batch: {e}.")
            batch_emotions = ['neutral'] * len(batch)  # Fallback if there's an error
        emotions.extend(batch_emotions)
    return emotions

# Example usage of the model on a dataset column
df['Emotion'] = infer_emotions_batch(df['Cleaned_Text'].tolist(), batch_size=32)

# Save the processed data
output_file_path = '/content/formatted_emotion_data_with_emojis.csv'
df.to_csv(output_file_path, index=False, encoding='utf-8')

print(f"Formatted dataset with emotions saved to: {output_file_path}")


# Then need to clean up the file, remove the LPT in the columns and only retain Emotions and Text columns

# In[12]:


import pandas as pd

# File path
file_path = r"C:\Users\Hazel\Downloads\Sentiment Analysis files\formatted_emotion_data_with_emojis.csv"

# Load the dataset
df = pd.read_csv(file_path)

# Drop the unwanted columns: 'Text' and 'Removed_Stopwords'
df = df.drop(columns=['Text', 'Removed_Stopwords'])

# Rename 'Cleaned_Text' to 'Text'
df = df.rename(columns={'Cleaned_Text': 'Text'})

# Remove 'LPT:' from the 'Text' column
df['Text'] = df['Text'].str.replace("LPT:", "", regex=False).str.strip()

# Save the cleaned dataset
output_file_path = r"C:\Users\Hazel\Downloads\Sentiment Analysis files\cleaned_emotion_data.csv"
df.to_csv(output_file_path, index=False, encoding='utf-8')

print(f"Cleaned dataset saved to: {output_file_path}")


# In[13]:


import pandas as pd

# Load the dataset
file_path = r"C:\Users\Hazel\Downloads\formatted_emotion_data_split_part_3.csv"
df = pd.read_csv(file_path)

# Drop the old 'Text' column (first column)
df = df.drop(df.columns[0], axis=1)

# Save the updated file
output_file_path = r"C:\Users\Hazel\Downloads\formatted_emotion_data_split_part_3_cleaned.csv"
df.to_csv(output_file_path, index=False, encoding='utf-8')

print(f"Updated dataset saved to: {output_file_path}")


# Now its time to do the same for Twitter data, we are dropping all other columns and will only retain Main_text column, then will do stopwords and will rename column to Text

# In[17]:


from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS  # Ensure this import is correct

# Load the dataset
file_path = r"C:\Users\Hazel\Downloads\Sentiment Analysis files\Sentimentanalysisdata\split_files\categorized_tweets_chunk_1.csv"
df = pd.read_csv(file_path, encoding='utf-8')

# Retain only the 'Main_Text' column
if 'Main_Text' not in df.columns:
    raise KeyError("'Main_Text' column is missing from the dataset. Ensure the dataset has the correct structure.")
df = df[['Main_Text']]

# Helper function to clean text while preserving emojis and removing stopwords
def clean_text_with_stopwords(text):
    # Remove URLs and non-alphabetic characters except emojis
    emoji_pattern = re.compile("[" 
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "]+", flags=re.UNICODE)
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r"[^a-zA-Z\s" + emoji_pattern.pattern + "]", '', text)  # Keep letters, spaces, and emojis
    text = re.sub(r"\s+", ' ', text).strip()  # Remove extra spaces

    # Remove stopwords
    original_words = text.split()
    filtered_words = [word for word in original_words if word.lower() not in ENGLISH_STOP_WORDS]
    
    # Return cleaned text
    return ' '.join(filtered_words)

# Apply the cleaning function to the 'Main_Text' column
df['Text'] = df['Main_Text'].fillna('').apply(clean_text_with_stopwords)

# Keep only the cleaned 'Text' column
df = df[['Text']]

# Save the updated file
output_file_path = r"C:\Users\Hazel\Downloads\Sentiment Analysis files\Sentimentanalysisdata\split_files\categorized_tweets_chunk_1_cleaned.csv"
df.to_csv(output_file_path, index=False, encoding='utf-8')

print(f"Cleaned dataset saved to: {output_file_path}")


# In[18]:


from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS  # Ensure this import is correct

# Load the dataset
file_path = r"C:\Users\Hazel\Downloads\Sentiment Analysis files\Sentimentanalysisdata\split_files\categorized_tweets_chunk_2.csv"
df = pd.read_csv(file_path, encoding='utf-8')

# Retain only the 'Main_Text' column
if 'Main_Text' not in df.columns:
    raise KeyError("'Main_Text' column is missing from the dataset. Ensure the dataset has the correct structure.")
df = df[['Main_Text']]

# Helper function to clean text while preserving emojis and removing stopwords
def clean_text_with_stopwords(text):
    # Remove URLs and non-alphabetic characters except emojis
    emoji_pattern = re.compile("[" 
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "]+", flags=re.UNICODE)
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r"[^a-zA-Z\s" + emoji_pattern.pattern + "]", '', text)  # Keep letters, spaces, and emojis
    text = re.sub(r"\s+", ' ', text).strip()  # Remove extra spaces

    # Remove stopwords
    original_words = text.split()
    filtered_words = [word for word in original_words if word.lower() not in ENGLISH_STOP_WORDS]
    
    # Return cleaned text
    return ' '.join(filtered_words)

# Apply the cleaning function to the 'Main_Text' column
df['Text'] = df['Main_Text'].fillna('').apply(clean_text_with_stopwords)

# Keep only the cleaned 'Text' column
df = df[['Text']]

# Save the updated file
output_file_path = r"C:\Users\Hazel\Downloads\Sentiment Analysis files\Sentimentanalysisdata\split_files\categorized_tweets_chunk_2_cleaned.csv"
df.to_csv(output_file_path, index=False, encoding='utf-8')

print(f"Cleaned dataset saved to: {output_file_path}")


# In[ ]:


Aligning Column Names


# In[19]:


# File path
file_path = r"C:\Users\Hazel\Downloads\Sentiment Analysis files\reddit3.csv"
output_file_path = r"C:\Users\Hazel\Downloads\Sentiment Analysis files\reddit3_updated.csv"

# Load the dataset
df = pd.read_csv(file_path, encoding='utf-8')

# Rename the column 'Text.1' to 'Text'
if 'Text.1' in df.columns:
    df = df.rename(columns={'Text.1': 'Text'})

# Save the updated dataset
df.to_csv(output_file_path, index=False, encoding='utf-8')

print(f"Updated dataset saved to: {output_file_path}")


# In[20]:


# File path
file_path = r"C:\Users\Hazel\Downloads\Sentiment Analysis files\reddit1.csv"
output_file_path = r"C:\Users\Hazel\Downloads\Sentiment Analysis files\reddit1_formatted.csv"

# Load the dataset
df = pd.read_csv(file_path, encoding='utf-8')

# Rename columns to 'Text' and 'Emotion'
if len(df.columns) >= 2:  # Ensure there are at least 2 columns
    df = df.rename(columns={df.columns[0]: 'Text', df.columns[1]: 'Emotion'})

# Keep only 'Text' and 'Emotion' columns
df = df[['Text', 'Emotion']]

# Save the updated dataset
df.to_csv(output_file_path, index=False, encoding='utf-8')

print(f"Formatted dataset saved to: {output_file_path}")


# Combining of Dataset that has Emoji column and dropping rows that doesn't have both values or has less than 2 in the two columns, emojis retained

# In[21]:


# File paths
file_paths = [
    r"C:\Users\Hazel\Downloads\Sentiment Analysis files\reddit1.csv",
    r"C:\Users\Hazel\Downloads\Sentiment Analysis files\reddit2.csv",
    r"C:\Users\Hazel\Downloads\Sentiment Analysis files\reddit3.csv",
    r"C:\Users\Hazel\Downloads\Sentiment Analysis files\twitter1.csv",
    r"C:\Users\Hazel\Downloads\Sentiment Analysis files\twitter2.csv"
]

# Output file path
output_file_path = r"C:\Users\Hazel\Downloads\Sentiment Analysis files\combined_sentiment_data.csv"

# Initialize an empty DataFrame to store combined data
combined_df = pd.DataFrame()

# Loop through each file and process
for file_path in file_paths:
    # Load the file
    df = pd.read_csv(file_path, encoding='utf-8')
    
    # Rename columns to ensure consistency
    if len(df.columns) >= 2:  # Ensure at least two columns exist
        df = df.rename(columns={df.columns[0]: 'Text', df.columns[1]: 'Emotion'})
        
    # Keep only 'Text' and 'Emotion' columns
    df = df[['Emotion', 'Text']]
    
    # Append the processed data to the combined DataFrame
    combined_df = pd.concat([combined_df, df], ignore_index=True)

# Drop rows where 'Emotion' or 'Text' is empty, null, or contains only one value
combined_df['Text'] = combined_df['Text'].fillna('').astype(str)
combined_df['Emotion'] = combined_df['Emotion'].fillna('').astype(str)
combined_df = combined_df[combined_df['Text'].str.strip().str.len() > 1]  # Remove rows where Text is empty or too short
combined_df = combined_df[combined_df['Emotion'].str.strip().str.len() > 1]  # Remove rows where Emotion is empty or too short

# Save the combined and cleaned data to the output file
combined_df.to_csv(output_file_path, index=False, encoding='utf-8')

print(f"Combined dataset saved to: {output_file_path}")


# In[ ]:


combined large dataset with Emotion and Text fixed and removed duplicate text with same emotion


# In[22]:


import pandas as pd

# File paths
file1_path = r"C:\Users\Hazel\Downloads\Sentiment Analysis files\combined_sentiment_data.csv"
file2_path = r"C:\Users\Hazel\Downloads\Sentiment Analysis files\formatted_emotion_data_with_emojis.csv"
output_file_path = r"C:\Users\Hazel\Downloads\Sentiment Analysis files\final_combined_emotion_data.csv"

# Load the datasets
df1 = pd.read_csv(file1_path, encoding='utf-8')
df2 = pd.read_csv(file2_path, encoding='utf-8')

# Ensure both datasets have the same column names
df1 = df1.rename(columns={df1.columns[0]: 'Emotion', df1.columns[1]: 'Text'})
df2 = df2.rename(columns={df2.columns[0]: 'Emotion', df2.columns[1]: 'Text'})

# Combine the datasets
combined_df = pd.concat([df1, df2], ignore_index=True)

# Drop duplicate rows where both 'Emotion' and 'Text' are the same
combined_df = combined_df.drop_duplicates(subset=['Emotion', 'Text'], keep='first')

# Save the final combined dataset
combined_df.to_csv(output_file_path, index=False, encoding='utf-8')

print(f"Final combined dataset saved to: {output_file_path}")


# Another error, the final output have Text in the Emotion column. So had to redo it 

# In[2]:


import pandas as pd

# File paths
file1_path = r"C:\Users\Hazel\Downloads\Sentiment Analysis files\actualfilesEmoji\combined_sentiment_data.csv"
file2_path = r"C:\Users\Hazel\Downloads\Sentiment Analysis files\actualfilesEmoji\formatted_emotion_data_with_emojis.csv"
output_file_path = r"C:\Users\Hazel\Downloads\Sentiment Analysis files\final_combined_emotion_data_cleaned.csv"

# Load the datasets
df1 = pd.read_csv(file1_path, encoding='utf-8')
df2 = pd.read_csv(file2_path, encoding='utf-8')

# Verify the number of columns and their names
if df1.shape[1] != 2 or df2.shape[1] != 2:
    raise ValueError("Both datasets must have exactly two columns: 'Emotion' and 'Text'.")

# Rename columns to ensure consistency
df1.columns = ['Emotion', 'Text']
df2.columns = ['Emotion', 'Text']

# Clean the data
# Remove leading/trailing spaces in 'Emotion' and 'Text'
df1['Emotion'] = df1['Emotion'].str.strip()
df1['Text'] = df1['Text'].str.strip()
df2['Emotion'] = df2['Emotion'].str.strip()
df2['Text'] = df2['Text'].str.strip()

# Combine the datasets
combined_df = pd.concat([df1, df2], ignore_index=True)

# Drop duplicate rows where both 'Emotion' and 'Text' are the same
combined_df = combined_df.drop_duplicates(subset=['Emotion', 'Text'], keep='first')

# Drop rows with missing or empty values in 'Emotion' or 'Text'
combined_df = combined_df.dropna(subset=['Emotion', 'Text'])
combined_df = combined_df[(combined_df['Emotion'] != '') & (combined_df['Text'] != '')]

# Reorder columns to ensure 'Emotion' comes first
combined_df = combined_df[['Emotion', 'Text']]

# Save the final combined dataset
combined_df.to_csv(output_file_path, index=False, encoding='utf-8')

print(f"Final combined dataset saved to: {output_file_path}")


# Some data in the Emotion has Text words, cleaning done to only retain the 8 valid emotions, and output the removed sentences into a new file so we could process for Emotion detection

# In[3]:


import pandas as pd

# Load the dataset
file_path = r"C:\Users\Hazel\Downloads\Sentiment Analysis files\actualfilesEmoji\finalemotiondatanoduplicate.csv"
output_clean_path = r"C:\Users\Hazel\Downloads\Sentiment Analysis files\cleaned_emotion_data.csv"
output_removed_path = r"C:\Users\Hazel\Downloads\Sentiment Analysis files\removed_emotion_data.csv"

# The valid emotions list
valid_emotions = ["joy", "sadness", "fear", "anger", "surprise", "neutral", "disgust", "shame"]

# Read the dataset
df = pd.read_csv(file_path, encoding='utf-8')

# Separate rows with valid emotions and those with invalid emotion text
valid_emotion_df = df[df['Emotion'].isin(valid_emotions)]
removed_emotion_df = df[~df['Emotion'].isin(valid_emotions)]

# Save the cleaned and removed datasets
valid_emotion_df.to_csv(output_clean_path, index=False, encoding='utf-8')
removed_emotion_df.to_csv(output_removed_path, index=False, encoding='utf-8')

valid_emotion_df.shape, removed_emotion_df.shape


# from the removed file, have done Emotion detection
# Validation of Emotions:
# 
# Checked if the Emotion column contains valid emotions (joy, sadness, etc.).
# Rows with invalid entries in the Emotion column were cleared (set to blank).
# Emotion Detection:
# 
# A basic keyword-based emotion detection method was implemented.
# Detected emotions based on keywords in the Text column for rows with blank Emotion.
# Single Output File:
# 
# All processing was performed in-place, and a single cleaned dataset was saved.

# In[ ]:


import pandas as pd

# File paths
file_path = r"C:\Users\Hazel\Downloads\Sentiment Analysis files\removed_emotion_data.csv"
output_clean_path = r"C:\Users\Hazel\Downloads\Sentiment Analysis files\cleaned_removed_emotion_data.csv"
output_text_path = r"C:\Users\Hazel\Downloads\Sentiment Analysis files\extracted_text_sentences.csv"

# Load the dataset
df = pd.read_csv(file_path, encoding='utf-8')

# Define a list of valid emotions
valid_emotions = ["joy", "sadness", "fear", "anger", "surprise", "neutral", "disgust", "shame"]

# Extract rows where the Emotion column contains sentences
invalid_emotion_rows = df[~df['Emotion'].isin(valid_emotions)]
valid_emotion_rows = df[df['Emotion'].isin(valid_emotions)]

# Save the invalid emotion rows as sentences in a separate file
invalid_emotion_rows[['Emotion']].to_csv(output_text_path, index=False, header=False, encoding='utf-8')

# Remove sentences from the Emotion column in the invalid rows
invalid_emotion_rows['Emotion'] = ''

# Combine valid and cleaned invalid rows
cleaned_df = pd.concat([valid_emotion_rows, invalid_emotion_rows], ignore_index=True)

# Perform simple emotion detection for blank Emotion rows (example using keyword detection)
def detect_emotion(text):
    keywords = {
        "happy": "joy",
        "sad": "sadness",
        "fearful": "fear",
        "angry": "anger",
        "surprised": "surprise",
        "neutral": "neutral",
        "disgusted": "disgust",
        "ashamed": "shame",
    }
    for keyword, emotion in keywords.items():
        if keyword in text.lower():
            return emotion
    return "unknown"  # Default if no match is found

# Apply emotion detection for blank Emotion rows
cleaned_df.loc[cleaned_df['Emotion'] == '', 'Emotion'] = cleaned_df.loc[cleaned_df['Emotion'] == '', 'Text'].apply(detect_emotion)

# Save the cleaned dataset
cleaned_df.to_csv(output_clean_path, index=False, encoding='utf-8')

print(f"Cleaned data saved to: {output_clean_path}")
print(f"Extracted text saved to: {output_text_path}")


# In[ ]:


Combining of two major files and removing duplicates 


# In[4]:


# File paths
file1_path = r"C:\Users\Hazel\Downloads\Sentiment Analysis files\formatted_emotion_data_with_emojis.csv"
file2_path = r"C:\Users\Hazel\Downloads\Sentiment Analysis files\cleaned_emotion_data.csv"
output_file_path = r"C:\Users\Hazel\Downloads\Sentiment Analysis files\merged_cleaned_emotion_data_with_emojis.csv"

# Load the datasets
df1 = pd.read_csv(file1_path, encoding='utf-8')
df2 = pd.read_csv(file2_path, encoding='utf-8')

# Ensure both datasets have the correct column names
df1.columns = ['Emotion', 'Text']
df2.columns = ['Emotion', 'Text']

# Remove leading/trailing spaces in both columns
df1['Emotion'] = df1['Emotion'].str.strip()
df1['Text'] = df1['Text'].str.strip()
df2['Emotion'] = df2['Emotion'].str.strip()
df2['Text'] = df2['Text'].str.strip()

# Merge the datasets
merged_df = pd.concat([df1, df2], ignore_index=True)

# Drop duplicates where both 'Emotion' and 'Text' are the same
merged_df = merged_df.drop_duplicates(subset=['Emotion', 'Text'], keep='first')

# Save the merged and cleaned dataset, ensuring emojis are preserved
merged_df.to_csv(output_file_path, index=False, encoding='utf-8')

print(f"Merged and cleaned data saved to: {output_file_path}")


# In[5]:


import pandas as pd

# File paths
source_file_path = r"C:\Users\Hazel\Downloads\Sentiment Analysis files\actualfilesEmoji\combined_sentiment_data.csv"
destination_file_path = r"C:\Users\Hazel\Downloads\Sentiment Analysis files\actualfilesEmoji\formatted_emotion_data_with_emojis.csv"
output_file_path = r"C:\Users\Hazel\Downloads\Sentiment Analysis files\actualfilesEmoji\updated_formatted_emotion_data_with_emojis.csv"

# Load the datasets
source_df = pd.read_csv(source_file_path, encoding='utf-8')
destination_df = pd.read_csv(destination_file_path, encoding='utf-8')

# Filter rows with specific emotions: neutral, disgust, shame
filtered_df = source_df[source_df['Emotion'].isin(['neutral', 'disgust', 'shame'])]

# Append filtered data to the destination DataFrame
updated_df = pd.concat([destination_df, filtered_df], ignore_index=True)

# Save the updated DataFrame
updated_df.to_csv(output_file_path, index=False, encoding='utf-8')

print(f"Updated file saved to: {output_file_path}")



# In[16]:


# Load the two datasets
file1_path = r"C:\Users\Hazel\Downloads\Sentiment Analysis files\combined_sentiment_data.csv"
file2_path = r"C:\Users\Hazel\Downloads\Sentiment Analysis files\nodup_emotion_dataset_cleaned.csv"

# Read the datasets
df1 = pd.read_csv(file1_path, encoding="utf-8")
df2 = pd.read_csv(file2_path, encoding="utf-8")

# Combine the datasets
combined_df = pd.concat([df1, df2], ignore_index=True)

# Save the combined dataset
output_combined_path = r"C:\Users\Hazel\Downloads\Sentiment Analysis files\combinednewest.csv"
combined_df.to_csv(output_combined_path, index=False, encoding="utf-8")

output_combined_path


# In[1]:


import pandas as pd

# Define file paths
file1 = r"C:\Users\Hazel\Downloads\emotion_dataset_raw.csv"
file2 = r"C:\Users\Hazel\Downloads\Sentiment Analysis files\emotion_dataset.csv"

# Read the files into DataFrames
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

# Combine the datasets (append rows)
combined_df = pd.concat([df1, df2], ignore_index=True)

# Save the combined DataFrame to a new CSV file
output_file = r"C:\Users\Hazel\Downloads\combined_emotion_dataset.csv"
combined_df.to_csv(output_file, index=False)

print(f"Combined dataset saved to: {output_file}")


# In[ ]:




