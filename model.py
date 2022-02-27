
import joblib
import pandas as pd
import numpy as np
import joblib 
from sklearn.feature_extraction.text import TfidfVectorizer
import string
import re

df = pd.read_csv("C://Users//Amit//Desktop//Deployment//model_files//sample30.csv")



# Converting floats to int
def remove_float(text):
    # values are split at decimal point
    lst = []
    for each in text:
        lst.append(str(each).split('.')[0])
    
    # all values converting to integer data type
    final_list = [i for i in lst if type(i) != float]
    final_list = ''.join(final_list)
    return final_list

# Remove numbers from string
def remove_digit(text):
    res = ''.join([i for i in text if not i.isdigit()])
    return res

# Defiing function for Removing punctuntion
def remove_punct(text):
    reviews = "".join([l for l in text if l not in string.punctuation])
    return reviews

# Defiing function for Removing extra white spaces
def remove_extra_white_space(text):
    return re.sub(' +',' ',text)

# Defiing function for Removing leading and lagging spaces
def remove_white_strip(text):
    return text.strip()

import nltk
from nltk.stem import WordNetLemmatizer
# nltk.download('averaged_perceptron_tagger')
from nltk.corpus import wordnet
 
lemmatizer = WordNetLemmatizer()
 
# Define function to lemmatize each word with its POS tag
 
# POS_TAGGER_FUNCTION : TYPE 1
def pos_tagger(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:         
        return None



def lemamtizer(*txt):
    
    txt = str(txt)
    # tokenize the review and find the POS tag for each token
    pos_tagged = nltk.pos_tag(nltk.word_tokenize(txt)) 
    wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))
    lemmatized_sentence = []
    
    for word, tag in wordnet_tagged:
        if tag is None:
        # if there is no available tag, append the token as is
            lemmatized_sentence.append(word)
        else:       
        # else use the tag to lemmatize the token
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
    lemmatized_sentence = " ".join(lemmatized_sentence)
    lemmatized_sentence = lemmatized_sentence[3:-6]
    lemmatized_sentence = lemmatized_sentence.lstrip()
 
    return lemmatized_sentence    



# Loading pickly files for predictions
load_ml_model = joblib.load("xgboost_savedd")
load_tfidf = joblib.load("tfidf_savedd")
load_user_rec = joblib.load("user_pred_recc")

def predict_product(user_name):
    

    # Filtering out a list of the top 20 recommended product ids
    top_20 = load_user_rec.loc[user_name].sort_values(ascending=False)[0:20].index
    # Creating a dataframe with only the recommended products
    top_20_df = df[df['id'].isin(top_20)][['id','reviews_title', 'reviews_text']]
    # Droping na items from dataframe
    top_20_df = top_20_df.dropna()
    # Combining title and reviews and droping columns after combining
    top_20_df['reviews'] = top_20_df['reviews_title'] + ' ' + top_20_df['reviews_text']
    top_20_df = top_20_df.drop(['reviews_title', 'reviews_text'], 1)
    
    # Removing floating point digits
    top_20_df['reviews'] = top_20_df['reviews'].apply(lambda x: remove_float(x))
    # Removing ints
    top_20_df['reviews'] = top_20_df['reviews'].apply(lambda x: remove_digit(x))
    # Removing punctuation
    top_20_df['reviews'] = top_20_df['reviews'].apply(lambda x: remove_punct(x))
    # Removing extra whitespaces
    top_20_df['reviews'] = top_20_df['reviews'].apply(lambda x: remove_extra_white_space(x))
    # Removing leading and lagging whitespace
    top_20_df['reviews'] = top_20_df['reviews'].apply(lambda x: remove_white_strip(x))    
    
    # Lematizing text reviews of recommended products
    top_20_lema = pd.DataFrame()
    top_20_lema['reviews'] = top_20_df['reviews'].apply(lambda x: lemamtizer(x))
    
    # Vectorizing the text reviews for recommended products 
    top_20_tfidf = load_tfidf.transform(top_20_lema['reviews'])
    # Sentiment based prediction for recommended products
    top_20_lema['pred'] = load_ml_model.predict(top_20_tfidf)
    # Combining ID Column to dataframe
    top_20_lema['id'] = top_20_df['id']
    
    # Creating THe final dataframe with the top 20 products
    fin = pd.DataFrame()
    fin['items'] = top_20
    # Mapping the product name from the product ID
    fin['prod_name'] = df[df['id'].isin(top_20)]['name'].unique()
    # Computing the Percentage of positive reviews
    fin['positive_percent'] = fin['items'].apply(lambda x: top_20_lema[top_20_lema['id'] == x]['pred'].value_counts(normalize = True)[0])
    # Filtering the top 5 based on percentage of positive reviews
    final = fin.sort_values(by='positive_percent', ascending=False)[:5]

    # Printing the top 5 products in a string format so flask can read it
    final_string=""
    for num,prod in enumerate(final['prod_name'].values, 1):
        final_string=final_string+' '+f"[{str(num)}] "+prod
    
    return final_string