import pyLDAvis.gensim
import pyLDAvis
from matplotlib import font_manager
from nltk.corpus import stopwords
from gensim.utils import simple_preprocess
from gensim.models import Phrases
from gensim.models.phrases import Phraser
from gensim.corpora import Dictionary
import numpy as np
from gensim.models import LdaMulticore
import spacy
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
from gensim.models import CoherenceModel
import re
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


stop_words = stopwords.words('english')
additional_stop_words = [
    'get', 'know', 'say', 'go', 'thing', 'come', 'right', 'really', 'think', 
    'man', 'make', 'look', 'love', 'want', 'like', "'", 'people', 'well', 'one', 
    'even', 'use', 'take', 'need', 'also', 'see', 'much', 'back', 'many',
    'shit', 'shitter', 'shitting', 'shite', 'bullshit', 'shitty',
    'fuck', 'fucking', 'fuckin', 'fucker', 'muthafucka', 
    'motherfuckers', 'motherfucke', 'motha', 'motherfucker',
    
    # Laughter and audience-related words
    'laughter', 'laughe', 'laugh', 'laughs', 'laughing', 'laughting', 
    'laughers', 'laugher', 'laughed', 'laugheters', 'laughety', 
    'audience', 'audence', 'audien', 'audiences', 'audeince',
    'audience_laugh', 'audience_laughs', 'audience_laughe', 
    'audience_laughing', 'audience_laughter', 'audience_laughed',
    'audience_cheer', 'audience_cheers', 'audience_cheering', 
    'audience_cheere', 'audience_cheer_applause', 
    'audience_chuckle', 'audience_chuckled', 'audience_chuckling',
    'crowd', 'crowd_laugh', 'crowd_laughs', 'crowd_laughe', 
    'crowd_laughing', 'crowd_cheer', 'crowd_cheers', 'crowd_cheering',
    'crowd_applaud', 'crowd_applauding', 'crowd_applause', 
    'crowd_claps', 'crowd_clap', 'crowd_chuckle', 'crowd_chuckled',
    
    # Reaction and filler words
    'cheer', 'cheers', 'cheering', 'cheered', 
    'applaud', 'applauds', 'applauding', 'applauded', 'applause',
    'clap', 'claps', 'clapping', 'clapped',
    'chuckle', 'chuckles', 'chuckling', 'chuckled',
    'boo', 'boos', 'booed', 'booing',
    'hiss', 'hisses', 'hissing', 'hissed',
    'reaction', 'react', 'reacting', 'reacted', 'responses', 'response',
    'roar', 'roars', 'roaring', 'roared',
    'sigh', 'sighs', 'sighing', 'sighed',
    'gasp', 'gasps', 'gasping', 'gasped',
    'snicker', 'snickers', 'snickering', 'snickered',
    'giggle', 'giggles', 'giggling', 'giggled',
    'murmur', 'murmurs', 'murmuring', 'murmured',
    'titter', 'titters', 'tittering', 'tittered'
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
    
    df.to_csv('output/data/03_Clean_Data.csv', index=False)
    df.to_pickle('output/data/03_Clean_Data.pkl')
    
    stop_words_set = set(stop_words)
    
    
    contains_stop_word = any(word in stop_words_set for word in df['words'][0])
    logger.info(f"Contains stop word: {contains_stop_word}")
    return df

# {1,2,3,4,5,6,7,8,9,10} -> {{1,2} , {3,4} , {5,6} , {7,8} , {9,10}}  => bigrams
# {{1,2} , {3,4} , {5,6} , {7,8} , {9,10}} -> {{1,2,3} , {5,6,7} , {8,9,10}} => trigrams

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

def lemmatize_text(trigrams):
    # Lematize text
    logger.info('Lematizing text')
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    allowed_postags=['NOUN','ADJ']
    lemmatized_words = []
    
    
    for sentence in tqdm(trigrams, desc="Lemmatizing"):
        document = nlp(" ".join(sentence))
        
        lemmatized_sentence = [
            token.lemma_ for token in document 
            if token.pos_ in allowed_postags and token.lemma_ not in stop_words
        ]
        lemmatized_words.append(lemmatized_sentence)
        
    # print(lemmatized_words[:5]);
    with open('output/data/03_Lemmatized_Words_New.pkl', 'wb') as f:
        pickle.dump(lemmatized_words, f)
    
    stop_words_set = set(stop_words)
    contains_stop_word = any(word in stop_words_set for lemma in lemmatized_words for word in lemma)
    logger.info(f"Lemmatized words contain stop word: {contains_stop_word}")
    
    return lemmatized_words
    
def create_corpus():
    # Create corpus
    logger.info('Creating corpus')
    lemmatized_words = []
    with open('output/data/03_Lemmatized_Words_New.pkl', 'rb') as f:
        lemmatized_words = pickle.load(f)
   
    
    id2word = Dictionary(lemmatized_words)
    corpus = [id2word.doc2bow(doc) for doc in lemmatized_words]
    
    print("Sample Dictionary:", list(id2word.items())[:10])  # Display a sample of the dictionary
    print("Sample Corpus:", corpus[:5])
    return corpus, id2word

def train_lda_model(corpus, id2word):
    # Train LDA model
    alpha = float(input("Enter the value for alpha: "))
    eta = float(input("Enter the value for eta: "))
    passes = int(input("Enter the value for passes: "))
    num_of_topics = int(input("Enter the number of topics: "))
    logger.info('Training LDA model')
    
    lda_model = LdaMulticore(corpus=corpus,
                             id2word=id2word,
                             num_topics=num_of_topics,
                             random_state=1,
                             chunksize=30,
                             passes=passes,
                             eta=eta,
                             alpha=alpha,
                             eval_every=1,
                             per_word_topics=True,
                             workers=1)
    
    lda_model.print_topics(num_topics=num_of_topics,num_words=15);
    return lda_model

def save_lda_model(lda_model):
    # Save the trained LDA model
    lda_model.save('output/data/LDA/03_LDA_Model')
    logger.info('LDA model saved successfully')
    return

def compute_coherence_score(id2word):
    # Compute coherence score
    lemmatized_words = []
    with open('output/data/03_Lemmatized_Words_New.pkl', 'rb') as f:
        lemmatized_words = pickle.load(f)
    
    lda_model = LdaMulticore.load('output/data/LDA/03_LDA_Model')    
    
    coherence_model_lda = CoherenceModel(model=lda_model, texts=lemmatized_words, dictionary=id2word, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    logger.info(f'Coherence Score: {coherence_lda}')
    return coherence_lda

def save_topics_and_coherence_score(coherence_lda):
    coherence_score = coherence_lda
    lda_topics = LdaMulticore.load('output/data/LDA/03_LDA_Model').print_topics(7,num_words=15)
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    topic_info = []
    topic_names = assign_topic_names()
    
    for idx, topic in enumerate(lda_topics):
        topic_num = idx
        keyowrds = re.findall(r'\"(.*?)\"', topic[1])
        topic_name = topic_names.get(topic_num, f"Topic {topic_num}")
        topic_info.append(f"{topic_name}: [{', '.join(keyowrds)}]")
    
    for i, info in enumerate(topic_info):
        topic_name, keywords = info.split(": ")
        ax.text(0.1, 0.9 - i*0.07, f"{topic_name}", fontsize=10, ha='left', wrap=True, fontweight='bold')
        ax.text(0.25, 0.9 - i*0.07, f"--> {keywords}", fontsize=10, ha='left', wrap=True,)
    
    # Add coherence score at the bottom center
    ax.text(0.5, 0.1, f"Coherence Score: {coherence_score:.2f}", fontsize=12, ha='center', va='center', transform=ax.transAxes)    
    ax.axis('off') 
    plt.title("LDA Topics Visualization", fontsize=16)
    plt.tight_layout()
    plt.savefig("output/03/03_Topics_With_Coherence_New.png", dpi=300)
    plt.close(fig)
    
    logger.info('Coherence score saved successfully')
    return
    
def visualize_topics( corpus, id2word):
    # Visualize the topics using pyLDAvis
    lda_model = LdaMulticore.load('output/data/LDA/03_LDA_Model')
    LDAvis_prepared = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
    pyLDAvis.save_html(LDAvis_prepared, 'output/03/03_LDA_Visualization_New.html')
    return LDAvis_prepared

def calculate_topic_probabilities(corpus):

    lda_model = LdaMulticore.load('output/data/LDA/03_LDA_Model')
    topic_vecs = []
    num_topics = lda_model.num_topics
    for i in range(len(corpus)):
        top_topics = lda_model.get_document_topics(corpus[i], minimum_probability=0.0)
        topic_vec = [top_topics[i][1] for i in range(num_topics)]
        topic_vecs.append(topic_vec)
    return topic_vecs

def add_topic_probabilities_to_df(topic_vecs):
    df = pd.read_csv('output/data/03_Clean_Data.csv')
    topic_names = assign_topic_names()
    # Add topic probabilities into the main DataFrame with appropriate column names
    topic_columns = list(topic_names.values())
    LDA_probs = pd.DataFrame(data=topic_vecs, columns=topic_columns, index=df.index)
    df = pd.concat([df, LDA_probs], axis=1)
    return df

def assign_topic_names():
# Mapping topics to names based on interpretation of their keywords
    topic_names = {
        0: "Racial Stereotypes",
        1: "Spanish or Mexican Culture",
        2: "British_Slang",
        3: "Family Relationships",
        4: "Romantic Relationships",
        5: "Melennials/Pandemic",
        6: "General"
    }
    return topic_names



# main function
if __name__ == '__main__':
    
    df = load_data()
    df = accept_english_only(df)
    # save and returns new cleaned data
    df_new = clean_tokens(df)
    trigrams = get_bigrams_and_trigrams(df_new)
    lemmatized_words = lemmatize_text(trigrams)
    
    # #* fech saved lemmatized words and create corpus and lda model
    corpus, id2word = create_corpus()
    lda_model = train_lda_model(corpus, id2word)
    save_lda_model(lda_model);
    
    #* fetch lda_mdel and lemmtized words and find coherence score
    coherence_lda = compute_coherence_score(id2word)
    save_topics_and_coherence_score(coherence_lda)
    visualize_topics(corpus,id2word)
    
    topic_vecs = calculate_topic_probabilities(corpus)
    df = add_topic_probabilities_to_df(topic_vecs)
    df.to_csv("output/data/03_Data_LDA.csv",index="false")
    df.to_pickle("output/data/03_Data_LDA.pkl");
    print(df.head());
    
    