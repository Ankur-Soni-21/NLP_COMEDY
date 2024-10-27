from utils.logging_config import configure_logging
logger = configure_logging();

import pandas as pd
def load_data():
    # Load data
    logger.info('Loading data')
    df = pd.read_csv('output/data/02_Sentiment_Data.csv')
    # print(df.info())
    return df

# Data columns (total 20 columns):
#  #   Column             Non-Null Count  Dtype  
# ---  ------             --------------  -----  
#  0   Unnamed: 0         499 non-null    int64  
#  1   url                499 non-null    object 
#  2   title              499 non-null    object 
#  3   date_posted        499 non-null    object 
#  4   transcript         499 non-null    object 
#  5   comedian           499 non-null    object 
#  6   language           499 non-null    object 
#  7   runtime            468 non-null    float64
#  8   rating             458 non-null    float64
#  9   rating_type        499 non-null    int64
#  10  words              499 non-null    object
#  11  word_count         499 non-null    int64
#  12  f_words            499 non-null    int64
#  13  s_words            499 non-null    int64
#  14  diversity          499 non-null    int64
#  15  diversity_ratio    499 non-null    float64
#  16  polarity           499 non-null    float64
#  17  subjectivity       499 non-null    float64
#  18  temporal_polarity  499 non-null    object
#  19  split_transcripts  499 non-null    object
# dtypes: float64(5), int64(6), object(9)
# memory usage: 78.1+ KB
# None

def accept_english_only(df):
    # Filter out non-English tweets
    logger.info('Filtering out non-English scripts')
    df = df[df.language == 'en']
    # print(df.language.value_counts())
    return df

# language
# en    491
# Name: count, dtype: int64

from nltk.corpus import stopwords
from gensim.utils import simple_preprocess
stop_words = stopwords.words('english')
additional_stop_words = [
'get', 'know', 'say', 'go', 'thing', 'come', 'right', 'really', 'think', 
'man', 'make', 'look', 'love', 'want', 'like', "'", 'people', 'well', 'one', 
'even', 'use', 'take', 'need', 'also', 'see', 'much', 'back', 'many',
'shit', 'shitter', 'shitting', 'shite', 'bullshit', 'shitty',
'fuck', 'fucking', 'fuckin', 'fucker', 'muthafucka', 
'motherfuckers', 'motherfucke', 'motha', 'motherfucker'
]
stop_words.extend(additional_stop_words)

def clean_tokens(df):
    logger.info('Cleaning tokens')
    

    
    df['words'] = df.transcript.apply(
        lambda x: [word for word in simple_preprocess(x, deacc=True) 
                  if word not in stop_words]
    )
    
    if isinstance(df['words'][0], str):
        df['words'] = df['words'].apply(lambda x: x.split() if isinstance(x, str) else x)
    
    
    
    # print(df['words'][0])  # Display a sample of cleaned words
    
    # Check if df.words[0] contains any stop word
    stop_words_set = set(stop_words)
    contains_stop_word = any(word in stop_words_set for word in df['words'][0])
    logger.info(f"Contains stop word: {contains_stop_word}")
    return df



from gensim.models import Phrases
from gensim.models.phrases import Phraser
def get_bigrams_and_trigrams(df):
    # Get bigrams and trigrams
    logger.info('Getting bigrams and trigrams')
    bigram_phrases = Phrases(df['words'], min_count=10, threshold=5)
    trigram_phrases = Phrases(bigram_phrases[df['words']], min_count=5, threshold=3)
    
    
    bigram_model = Phraser(bigram_phrases)
    trigram_model = Phraser(trigram_phrases) 
    trigrams = [trigram_model[bigram_model[word]] for word in df['words']]   
    # print(trigrams[:5])
    
    
    stop_words_set = set(stop_words)
    contains_stop_word = any(word in stop_words_set for trigram in trigrams for word in trigram)
    logger.info(f"Trigrams contain stop word: {contains_stop_word}")
    
    return trigrams


import spacy
from tqdm import tqdm
import pickle

def lemmatize_text(trigrams):
    # Lematize text
    logger.info('Lematizing text')
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    allowed_postags=['NOUN','ADJ','VERB','ADV']
    lemmatized_words = []
    
    
    for sentence in tqdm(trigrams, desc="Lemmatizing"):
        document = nlp(" ".join(sentence))
        
        lemmatized_sentence = [
            token.lemma_ for token in document 
            if token.pos_ in allowed_postags and token.lemma_ not in stop_words
        ]
        lemmatized_words.append(lemmatized_sentence)
        
    # print(lemmatized_words[:5]);
    with open('output/data/03_Lemmatized_Words.pkl', 'wb') as f:
        pickle.dump(lemmatized_words, f)
    
    stop_words_set = set(stop_words)
    contains_stop_word = any(word in stop_words_set for lemma in lemmatized_words for word in lemma)
    logger.info(f"Lemmatized words contain stop word: {contains_stop_word}")
    
    return lemmatized_words
    

from gensim.corpora import Dictionary
def create_corpus():
    # Create corpus
    logger.info('Creating corpus')
    lemmatized_words = []
    with open('output/data/03_Lemmatized_Words.pkl', 'rb') as f:
        lemmatized_words = pickle.load(f)
   
    
    id2word = Dictionary(lemmatized_words)
    corpus = [id2word.doc2bow(doc) for doc in lemmatized_words]
    
    # print("Sample Dictionary:", list(id2word.items())[:10])  # Display a sample of the dictionary
    # print("Sample Corpus:", corpus[:5])
    return corpus, id2word




import numpy as np
from gensim.models import LdaMulticore

def train_lda_model(corpus, id2word):
    # Train LDA model
    num_of_topics = 7
    lda_model = LdaMulticore(corpus=corpus,
                             id2word=id2word,
                             num_topics=num_of_topics,
                             random_state=1,
                             chunksize=30,
                             passes=69,
                             per_word_topics=True,
                             minimum_probability=0.0,
                             alpha=0.91,
                             eta=0.31,
                             workers=4)
    
    lda_model.print_topics(7,num_words=15);
    return lda_model
    
    

# main function
if __name__ == '__main__':
    df = load_data()
    df = accept_english_only(df)
    df_new = clean_tokens(df)
    trigrams = get_bigrams_and_trigrams(df_new)
    lemmatized_words = lemmatize_text(trigrams)
    corpus, id2word = create_corpus()
    # lda_model = train_lda_model(corpus, id2word)
    