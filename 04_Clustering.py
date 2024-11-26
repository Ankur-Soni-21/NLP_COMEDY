import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.cm as cm
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from utils.logging_config import configure_logging
logger = configure_logging();

INPUT_DATA_CSV = 'output/data/03_Data_LDA.csv'
INPUT_DATA_PKL = 'output/data/03_Data_LDA.pkl'

topics= [
        "Racial Stereotypes",
        "Spanish or Mexican Culture",
        "British_Slang",
        "Family Relationships",
        "Romantic Relationships",
        "Melennials/Pandemic",
        "General"
    ]

def load_data():
    df = pd.read_pickle(INPUT_DATA_PKL)
    logger.info(f"Data loaded from {INPUT_DATA_PKL}")
    df.info()
    return df



plt.rcParams['figure.dpi'] = 150 
plt.rcParams['savefig.dpi'] = 300

def topic_visualization(df):

    color_palette = sns.color_palette("muted", len(topics))
    plt.figure(figsize=(10, 6))
    
    ax = sns.barplot(x=df[topics].mean().index, y=df[topics].loc[df.rating_type == 1].mean(), palette=color_palette)
    ax.set_xticks(range(len(topics)))
    ax.set_xticklabels(topics, rotation=40, ha='right')
    ax.set_title('Mean Topic Probabilities Across The Entire Dataset')
    ax.set(xlabel='Topics', ylabel='Mean Percentage per Transcript', ylim=(0, 1))
    
    plt.tight_layout()
    
    
    os.makedirs('output/04', exist_ok=True)
    plt.savefig('output/04/04_Topic_Visualization.png')


def data_prep_for_clustering(df):
    X = df[topics]
    X = StandardScaler().fit_transform(X)
    print(X.shape)
    return X
    

def clustering_with_diff_n(df,X):
    temp_dict = {}
    inertias = []
    
    for n_clusters in range(2, 15):
        clusterer = KMeans(n_clusters=n_clusters, random_state=1)
        cluster_labels = clusterer.fit_predict(X)
        
        silhouette_avg = silhouette_score(X, cluster_labels)
        print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)
        temp_dict[n_clusters] = [silhouette_avg] 
        
        inertia = clusterer.inertia_
        print("\tThe inertia is :", inertia)
        inertias.append(inertia)
    
    return temp_dict, inertias


def plot_silhouette_and_inertia(temp_dict,inertias):
    sns.set(font_scale=1.2)
    sns.set_style('ticks')
    s_scores = pd.DataFrame(temp_dict).T
    ax = sns.lineplot(x=s_scores.index, y=s_scores[0], color='teal')
    # ax.set_yticks([])  # Clear the yaxis labels / ticks
    ax.set_xticks(range(2,15))
    ax.set_ylabel('Silhouette score')
    ax.set_xlabel('Clusters')
    ax.figure.tight_layout()
    ax.figure.savefig('output/04/04_Silhouette_Plot.png')
    plt.close(ax.figure)

    ax = sns.lineplot(x=range(2,15), y=inertias, color='teal')
    ax.set_ylabel('SSE (inertia)')
    ax.set_xlabel('Clusters')
    ax.figure.tight_layout()
    ax.figure.savefig('output/04/04_Inertia_Plot.png')
    plt.close(ax.figure)
    
    
def clustering(df,X,n_clusters):
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    df['cluster_LDA'] = clusterer.fit_predict(X)
    
    for cluster in range(n_clusters):
        fig, (ax1) = plt.subplots(1, 1)
        fig.set_size_inches(4, 4)
        
        ax = sns.barplot(x=df[topics].mean().index, y=df[topics].loc[df.cluster_LDA == cluster].mean())
        ax.set_xticklabels(topics, rotation=40, ha='right')   
        ax.set_title(f'cluster: {cluster}')
        ax.figure.tight_layout()
        ax.figure.savefig(f'output/04/NON_TFIDF/04_Cluster_{cluster}.png')
    
    print(df.cluster_LDA.value_counts())
    return df


def tfidf_vectorization():
    def indetity_tokenizer(text):
        return text
    
    tfidf = TfidfVectorizer(tokenizer=indetity_tokenizer,
                            lowercase=False,
                            min_df=10,
                            max_df=0.4)

    lemmatized_words = []
    with open('output/data/03_Lemmatized_Words_New.pkl', 'rb') as f:
        lemmatized_words = pickle.load(f)
        
    X_tfidf = tfidf.fit_transform(lemmatized_words)
    print(X_tfidf.shape)
    return X_tfidf;


def clustering_using_tfidf_with_diff_n(X):
    temp_dict = {}
    inertias = []
    
    for n_clusters in range(2, 40):
        clusterer = KMeans(n_clusters=n_clusters, random_state=1)
        cluster_labels = clusterer.fit_predict(X)
        
        silhouette_avg = silhouette_score(X, cluster_labels)
        print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)
        temp_dict[n_clusters] = [silhouette_avg] 
        
        inertia = clusterer.inertia_
        print("\tThe inertia is :", inertia)
        inertias.append(inertia)
    
    return temp_dict, inertias


def plot_silhouette_and_inertia_tfidf(temp_dict,inertias):
    sns.set(font_scale=1.2)
    sns.set_style('ticks')
    s_scores = pd.DataFrame(temp_dict).T
    ax = sns.lineplot(x=s_scores.index, y=s_scores[0], color='teal')
    ax.set_xticks(range(2,40,4))
    ax.set_ylabel('Silhouette score')
    ax.set_xlabel('Clusters')
    ax.figure.tight_layout()
    ax.figure.savefig('output/04/04_Silhouette_Plot_TFIDF.png')
    plt.close(ax.figure)

    ax = sns.lineplot(x=range(2,40), y=inertias, color='teal')
    ax.set_ylabel('SSE (inertia)')
    ax.set_xlabel('Clusters')
    ax.figure.tight_layout()
    ax.figure.savefig('output/04/04_Inertia_Plot_TFIDF.png')
    plt.close(ax.figure)
    
    
def clustering_with_tfidf(df,X_tfidf,n_clusters):
    clusterer = KMeans(n_clusters=7, random_state=10)
    df['cluster_tfidf'] = clusterer.fit_predict(X_tfidf)
    
    for cluster in range(n_clusters):
        fig, (ax1) = plt.subplots(1, 1)
        fig.set_size_inches(4, 4)
        
        ax = sns.barplot(x=df[topics].mean().index, y=df[topics].loc[df.cluster_tfidf == cluster].mean())
        ax.set_xticklabels(topics, rotation=40, ha='right')   
        ax.set_title(f'cluster: {cluster}')
        ax.figure.tight_layout()
        ax.figure.savefig(f'output/04/TFIDF/04_Cluster_TFIDF_{cluster}.png')  
        
    print(df.cluster_tfidf.value_counts())
    return df

if __name__ == '__main__':
    
    # #* load data and plot topics
    df = load_data()
    topic_visualization(df)
    
    # #* try clustering with different number of clusters
    X = data_prep_for_clustering(df)
    temp_dict, inertias = clustering_with_diff_n(df,X)
    plot_silhouette_and_inertia(temp_dict,inertias)
    
    # # #* Clustering with X clusters
    # # #* X is based on our observation from the Silhouette and Inertia plots
    df = clustering(df,X,7)
    
    
    # # #* TF-IDF Vectorization and clustering
    X_tfidf = tfidf_vectorization()
    temp_dict_tfidf, inertia_tfidf = clustering_using_tfidf_with_diff_n(X_tfidf)
    plot_silhouette_and_inertia_tfidf(temp_dict_tfidf,inertia_tfidf)

    # #* Clustering with TF-IDF
    df = clustering_with_tfidf(df,X_tfidf,7)
    
    # #* Save the data
    df = df.iloc[:, 1:].reset_index(drop=True)
    
    df.to_pickle('output/data/04_Data_TFIDF.pkl')
    df.to_csv('output/data/04_Data_TFIDF.csv')
    
    