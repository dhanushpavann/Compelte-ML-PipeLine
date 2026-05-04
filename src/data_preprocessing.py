import os
import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string
import nltk
nltk.download('stopwords')
nltk.download('punkt')

# --- SETUP ---
ps = PorterStemmer()
STOP_WORDS = set(stopwords.words('english'))
PUNCTUATION = set(string.punctuation)

# Ensure that the "logs" directory exsits
log_dir='logs'
os.makedirs(log_dir,exist_ok=True)

# Logger
logger=logging.getLogger('data_preprocessing')
logger.setLevel('DEBUG')

# Console Handler
console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')

# File Handler
log_file_path=os.path.join(log_dir,'data_preprocessing.log')
file_handler=logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

# Formatter
formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add Handler to Logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)


# def transform_text(text):
#     """
#     Transforms the input text by converting it into lowercase,tokenizing,removing stopwords and punctuation and stemming.
#     """
#     ps = PorterStemmer()
#     # convert to LowerCase
#     text=text.lower()
#     # Tokenize the text
#     text=nltk.word_tokenize(text)
#     # Remove non-alphanumeric tokens
#     text=[word for word in text if word.isalnum()]
#     # Remove stop words and punctuation
#     text=[word for word in text if word not in stopwords.words('english') and  word not in string.punctuation]
#     # Stem the words
#     text=[ps.stem(word) for word in text]
#     # Join the tokens back into a single string
#     return " ".join(text)

def transform_text(text):
    # 1. Lowercase
    text = text.lower()
    
    # 2. Tokenize
    tokens = nltk.word_tokenize(text)
    
    # 3. Filter alphanumeric
    tokens = [word for word in tokens if word.isalnum()]
    
    # 4. Remove Stopwords and Punctuation
    # (Note: Using global STOP_WORDS and PUNCTUATION for speed)
    tokens = [word for word in tokens if word not in STOP_WORDS and word not in PUNCTUATION]
    
    # 5. Stemming
    tokens = [ps.stem(word) for word in tokens]
    
    # 6. Join
    return " ".join(tokens)

def preprocess_df(df,text_column='text',target_column='target'):
    """
    Preprocesses the DataFrame by encoding the target column,removing duplicates and transforming the text column
    """
    try:
        logger.debug("Starting Preprocessing for DataFrame")
        # Encoder the Target column
        encoder=LabelEncoder()
        df[target_column]=encoder.fit_transform(df[target_column])
        logger.debug("Target Column Encoded")

        # Remove the duplicates
        df=df.drop_duplicates(keep='first').copy()
        logger.debug("Duplicates removed")

        # Apply text transformation to the text column
        df.loc[:,text_column]=df[text_column].apply(transform_text)
        logger.debug("Text column Transformed")
        return df
    
    except KeyError as e:
        logger.debug("Column not found: %s",e)
        raise
    except Exception as e:
        logger.debug("Error during preprocessing of DataFrame : %s",e)
        raise
def main(text_column='text',target_column='target'):
    try:
        # Fetch the data from ./data/raw
        train_data=pd.read_csv('./data/raw/train.csv')
        test_data=pd.read_csv('./data/raw/test.csv')
        logger.debug("Data Fetched Succesfully")

        # Transform the Data
        logger.debug("----------------- Processing Train Data -----------------")
        train_preprocessed_data=preprocess_df(train_data,text_column,target_column)
        logger.debug("----------------- Processing Test Data -----------------")
        test_preprocessed_data=preprocess_df(test_data,text_column,target_column)

        # Store the data inside data/processed
        data_path=os.path.join("./data","interim")
        os.makedirs(data_path,exist_ok=True)

        train_preprocessed_data.to_csv(os.path.join(data_path,'train_processed.csv'),index=False)
        test_preprocessed_data.to_csv(os.path.join(data_path,'test_processed.csv'),index=False)

        logger.debug("Processed data to: %s",data_path)
    except FileNotFoundError as e:
        logger.error("File not Found: %s",e)
        raise
    except pd.errors.EmptyDataError as e:
        logger.error("No Data: %s",e)
        raise
    except Exception as e:
        logger.error("Failed to complete data preprocessing step: %s",e)
        print(f"Error: {e}")    
        raise


if __name__ == "__main__":
    main()

        