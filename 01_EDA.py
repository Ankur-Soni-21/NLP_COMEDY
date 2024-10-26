# os for file operations
import os
# logging for logging
import logging
# warnings to suppress warnings
import warnings

# pandas for datframes
import pandas as pd
# seaborn for visualizations
import seaborn as sns

# tqdm for progress bars
from tqdm import tqdm

# typing for data types
from typing import List
from typing import Tuple

# imdb for fetching movie information
from imdb import Cinemagoer

# langdetect for language detection
from langdetect import detect
from langdetect import DetectorFactory

# nltk for text processing
from nltk import word_tokenize
from nltk.corpus import stopwords

# matplotlib for visualizations
import matplotlib.pyplot as plt\
    
# wordcloud for word clouds
from wordcloud import WordCloud

# gensim for text processing
from gensim.utils import simple_preprocess

# configure logging settings
from utils.logging_config import configure_logging
logger = configure_logging()

# Suppress specific warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Set seed for consistent language detection
DetectorFactory.seed = 0


def combine_csv_files(file_paths: List[str], type_of_run: str) -> pd.DataFrame:
    try:
        # Create an empty list to store individual DataFrames
        dfs = []
        
        # Read each CSV file and append to the list
        for file_path in file_paths:
            df = pd.read_csv(file_path)
            dfs.append(df)
            logging.info(f"Successfully read file: {file_path}")
        
        # Combine all DataFrames
        combined_df = pd.concat(dfs, ignore_index=True)
        logging.info(f"Successfully combined {len(file_paths)} files with total {len(combined_df)} records")
        
        if type_of_run == 'test':
            return combined_df.head(10)
        else:
            return combined_df
        
    except Exception as e:
        logging.error(f"Error combining files: {e}")
        raise

def read_and_clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the combined transcript data.
    
    Args:
        df (pd.DataFrame): Combined DataFrame from multiple CSV files
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    try:
        # Rename columns to match the expected format
        df = df.rename(columns={
            'web-scraper-start-url': 'url',
            'title': 'title',
            'date_posted': 'date_posted',
            'transcript': 'transcript'
        })
        
        # Drop the web-scraper-order column
        df = df.drop('web-scraper-order', axis=1, errors='ignore')
        
        # Clean data
        df = df.dropna(subset=['transcript'])  # Remove rows with no transcript
        df = df.drop_duplicates(subset=['title', 'transcript'])  # Remove duplicates
        
        logging.info(f"Successfully cleaned data, remaining records: {len(df)}")
        return df
        
    except Exception as e:
        logging.error(f"Error cleaning data: {e}")
        raise

def detect_language(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect language of transcripts and add as a new column.
    
    Args:
        df (pd.DataFrame): Input DataFrame with transcripts
    
    Returns:
        pd.DataFrame: DataFrame with language column added
    """
    try:
        def safe_detect(text: str) -> str:
            try:
                # Use only first 500 characters for faster processing
                return detect(str(text)[:500])
            except:
                return 'unknown'
        
        df['language'] = df['transcript'].apply(safe_detect)
        
        # Log language distribution
        lang_counts = df['language'].value_counts()
        logging.info(f"Language distribution:\n{lang_counts}")
        
        return df
        
    except Exception as e:
        logging.error(f"Error in language detection: {e}")
        raise
    
def get_imdb_info(df: pd.DataFrame, type_of_run : str, cache_file: str = 'output/data/00_Clean_Data.csv') -> pd.DataFrame:
    """
    Fetch IMDB information (runtime and rating) for each title.
    
    Args:
        df (pd.DataFrame): Input DataFrame with titles
    
    Returns:
        pd.DataFrame: DataFrame with runtime and rating columns added
    """
    if os.path.exists(cache_file):
        logging.info(f"Loading cached data from: {cache_file}")
        return pd.read_csv(cache_file)
    
    
    ia = Cinemagoer()  # Updated to use Cinemagoer instead of IMDb
    errors = 0
    
    def fetch_movie_info(title: str) -> Tuple[str, str]:
        try:
            # Search with a cleaned and truncated title
            clean_title = str(title).strip()[:30]
            results = ia.search_movie(clean_title)
            if results:
                movie = ia.get_movie(results[0].movieID)
                runtime = movie.get('runtimes', [''])[0]
                rating = movie.get('rating', '')
                return runtime, rating
            
            return '', ''
        except Exception as e:
            nonlocal errors
            errors += 1
            logging.warning(f"Error fetching info for '{title}': {e}")
            return '', ''
    
    # Process all titlesī
    logging.info("Fetching IMDB information...")
    results = [fetch_movie_info(title) for title in tqdm(df['title'] , desc = 'Fetching IMDB info')]
    
    # Unpack results
    runtimes, ratings = zip(*results)
    
    # Add new columns
    df['runtime'] = runtimes
    df['rating'] = ratings
    
    # Convert to numeric where possible
    df['runtime'] = pd.to_numeric(df['runtime'], errors='coerce')
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    
    logging.info(f"IMDB info fetching complete. Failed to fetch {errors} titles")
    
    return df

def create_rating_features(df: pd.DataFrame) -> pd.DataFrame:
    
        """Create rating type feature and visualization"""
        
        df['rating_type'] = df.rating.apply(lambda x: 1 if x >= df.rating.mean() else 0)
        
        plt.figure(figsize=(10, 6))
        ax = sns.countplot(x='rating_type', data=df)
    
        # Set ticks before setting tick labels
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Low rating (< mean)', 'High rating (> mean)'])
        
        ax.set(title='Counts of specials with higher or lower than average ratings')
        plt.savefig('output//01/01_Rating_Distribution.png')
        plt.close()
        
        return df

def analyze_runtime_and_ratings(df: pd.DataFrame):
    
    """
    Create runtime and rating visualizations
    kde stands for Kernel Density Estimate, a method used to visualize the distribution of data by estimating its probability density function. 
    In the script, it's part of the seaborn library (sns.kdeplot) for creating smooth, continuous plots to show data distributions
    """
    
    # Runtime analysis
    valid_runtime = df[df.runtime > 0].runtime.astype(int)
    plt.figure(figsize=(10, 6))
    ax = sns.kdeplot(valid_runtime, fill=True, color="r")
    ax.set_title('Runtime KDE')
    ax.set(xlabel='minutes')
    plt.savefig('output/01/02_Runtime_Distribution.png')
    plt.close()
    
    logging.info(f'Runtime Mean: {valid_runtime.mean():.2f}')
    logging.info(f'Runtime SD: {valid_runtime.std():.2f}')
    
    # Rating analysis
    valid_rating = df[df.rating > 0].rating
    plt.figure(figsize=(10, 6))
    ax = sns.kdeplot(valid_rating, fill=True, color="g")
    ax.set_title('IMDb Rating KDE')
    plt.savefig('output/01/03_Rating_Distribution.png')
    plt.close()
    
    logging.info(f'Rating Mean: {valid_rating.mean():.2f}')
    logging.info(f'Rating SD: {valid_rating.std():.2f}')

def process_text(df: pd.DataFrame) -> pd.DataFrame:
    """Process text data and create word-related features"""
    
    stop_words = stopwords.words('english')
    stop_words.extend(['audience', 'laughter', 'laughing', 'announcer', 'narrator', 'cos'])
    
    # Tokenize and clean words
    df['words'] = df.transcript.apply(
        lambda x: [word for word in simple_preprocess(x, deacc=True) 
                  if word not in stop_words]
    )
    
    # Word count
    df['word_count'] = df.words.apply(len)
    
    # Process swear words
    f_words = ['fuck', 'fucking', 'fuckin', 'fucker', 'muthafucka', 
               'motherfuckers', 'motherfucke', 'motha', 'motherfucker']
    s_words = ['shit', 'shitter', 'shitting', 'shite', 'bullshit', 'shitty']
    
    df['f_words'] = df.words.apply(lambda x: sum(word.lower() in f_words for word in x))
    df['s_words'] = df.words.apply(lambda x: sum(word.lower() in s_words for word in x))
    
    # Remove swear words from words list
    swears = f_words + s_words + ['cunt', 'asshole', 'damn', 'goddamn', 'cocksucker']
    df['words'] = df.words.apply(lambda x: [word for word in x if word not in swears])
    
    # Create diversity features
    df['diversity'] = df.words.apply(lambda x: len(set(x)))
    df['diversity_ratio'] = df.diversity / df.word_count
    
    return df

def create_word_visualizations(df: pd.DataFrame):
    """Create word-related visualizations"""
    # F-words distribution
    plt.figure(figsize=(10, 6))
    ax = sns.kdeplot(df.f_words, fill=True, color="r")
    ax.set_title('F-Words Count KDE')
    plt.savefig('output/01/04_F_Words_Distribution.png')
    plt.close()
    
    # S-words distribution
    plt.figure(figsize=(10, 6))
    ax = sns.kdeplot(df.s_words, fill=True, color="r")
    ax.set_title('S-Words Count KDE')
    plt.savefig('output/01/05_S_Words_Distribution.png')
    plt.close()
    
    # Word diversity
    plt.figure(figsize=(10, 6))
    ax = sns.kdeplot(df.diversity, fill=True, color="purple")
    ax.set_title('Word Diversity KDE')
    plt.savefig('output/01/06_Word_Diversity.png')
    plt.close()
    
    # Diversity ratio
    plt.figure(figsize=(10, 6))
    ax = sns.kdeplot(df.diversity_ratio, fill=True, color="g")
    ax.set_title('Diversity / Total words KDE')
    plt.savefig('output/01/07_Diversity_Ratio.png')
    plt.close()
    
def create_wordclouds(df: pd.DataFrame):
    logging.info("Creating word clouds...")
    
    """Create and save word clouds"""
    
    wordcloud = WordCloud(
        background_color="white", 
        max_words=5000, 
        contour_width=3, 
        contour_color='midnightblue'
    )
    
     # Ensure the output directory exists
    os.makedirs('output', exist_ok=True)
    
    df_words = 0
    for _, row in df.iterrows():
        df_words += len(row['words'])
    logging.info(f"Total words in DataFrame: {df_words}")
    
    # Concatenate all words in the df.words column
    all_words = ' '.join([' '.join(words) for words in df.words])
    
    # Generate word cloud from the combined text
    wordcloud.generate(all_words)
    wordcloud.to_file('output/01/08_Wordcloud_All.png')
    logging.info("Word cloud saved for all words")
    
def process_transcript_data(file_paths: List[str] , type_of_run : str) -> pd.DataFrame:
    """
    Main function to process transcript data from multiple files.
    
    Args:
        file_paths (List[str]): List of paths to CSV files
    
    Returns:
        pd.DataFrame: Processed DataFrame with all additional information
    """
    try:
        # Combine and process data
        combined_df = combine_csv_files(file_paths,type_of_run)
        df = read_and_clean_data(combined_df)
        df = detect_language(df)
        df = get_imdb_info(df,type_of_run=type_of_run)
        
        # Create features and visualizations
        df = create_rating_features(df)
        analyze_runtime_and_ratings(df)
        df = process_text(df)
        create_word_visualizations(df)
        create_wordclouds(df)
        
        # Save processed data
        os.makedirs('output/data', exist_ok=True)
        df.to_pickle('output/data/00_Clean_Data.pkl')
        logging.info("Analysis completed successfully")
        return df
        
    except Exception as e:
        logging.error(f"Error in main process: {e}")
        raise
    
if __name__ == "__main__":
    file_paths = [
        'csv/0-100.csv',
        'csv/101-200.csv',
        'csv/201-300.csv',
        'csv/301-400.csv',
        'csv/401-500.csv'
    ]
    type_of_run = 'FINAL'
    logging.warning("TYPE_OF_RUN : " + type_of_run);
    processed_df = process_transcript_data(file_paths, type_of_run)
    logging.info("\nFirst few rows of processed data:")
    logging.info(processed_df.head())
    
    # Save processed data
    processed_df.to_csv('output/data/00_Clean_Data.csv', index=False)