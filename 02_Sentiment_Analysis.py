# Import required libraries
import pandas as pd
import numpy as np
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
import math
from pathlib import Path
from utils.logging_config import configure_logging
logger = configure_logging()

# Define input and output paths
INPUT_PATH_PKL = Path('output/data/01_Clean_Data.pkl')
INPUT_PATH_CSV = Path('output/data/01_Clean_Data.csv')
OUTPUT_PATH_DATA = Path('output/data')
OUTPUT_PATH_VISUAL = Path('output/02')

# Output files
SENTIMENT_PLOT = OUTPUT_PATH_VISUAL / '02_Sentiment_Analysis.png'
TEMPORAL_PLOT = OUTPUT_PATH_VISUAL / '03_Temporal_Sentiment_Analysis.png'
PROCESSED_DATA_PKL = OUTPUT_PATH_DATA / '02_Sentiment_Data.pkl'
PROCESSED_DATA_CSV = OUTPUT_PATH_DATA / '02_Sentiment_Data.csv'

def load_data(file_path):
    if str(file_path).endswith('.csv'):
        logger.info(f"Data loaded from {file_path}")
        return pd.read_csv(file_path)
    elif str(file_path).endswith('.pkl'):
        logger.info(f"Data loaded from {file_path}")
        return pd.read_pickle(file_path)
    else:
        logger.error("Unsupported file format. Please use CSV or PKL file.")
        raise ValueError("Unsupported file format. Please use CSV or PKL file.")

def calculate_sentiment_metrics(df):
    def get_sentiment(text):
        blob = TextBlob(str(text))
        return pd.Series({
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity
        })
    
    logger.info("Calculating sentiment metrics...")
    # Calculate sentiment metrics
    sentiment_df = df['transcript'].apply(get_sentiment)
    
    # Add new columns to original dataframe
    df['polarity'] = sentiment_df['polarity']
    df['subjectivity'] = sentiment_df['subjectivity']
    
    logger.info("Sentiment metrics calculated successfully!")
    return df

def plot_sentiment_analysis(df, output_path):
    logger.info("Plotting sentiment analysis...")
    plt.figure(figsize=(12, 10))
    
    # Create scatter plot
    plt.scatter(df['polarity'], df['subjectivity'], color='blue', alpha=0.5)
    
    # Customize plot
    plt.title('Sentiment Analysis', fontsize=20)
    plt.xlabel('<-- Negative -------- Positive -->', fontsize=15)
    plt.ylabel('<-- Facts -------- Opinions -->', fontsize=15)
    
    # Calculate x-axis limits to center around 0
    max_abs_polarity = max(abs(df['polarity'].max()), abs(df['polarity'].min()))
    plt.xlim(-max_abs_polarity, max_abs_polarity)
    
    # Add vertical line at x=0
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    
    # Add grid for better readability
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Sentiment analysis plot saved to: {output_path}")
    plt.close()

def split_text(text, n=10):
    # Calculate length of text, chunk size, and starting points
    length = len(str(text))
    size = math.floor(length / n)
    start = np.arange(0, length, size)
    
    # Create list of split text pieces
    split_list = []
    for piece in range(min(n, len(start))):
        split_list.append(text[start[piece]:start[piece]+size])
    return split_list

def create_transcript_pieces(df):
    logger.info("Splitting transcripts into pieces for temporal analysis...")
    list_pieces = []
    for transcript in df.transcript:
        split = split_text(transcript)
        list_pieces.append(split)
    logger.info("Transcripts split successfully!")
    return list_pieces

def calculate_polarity_over_time(list_pieces):
    logger.info("Calculating polarity over time...")
    polarity_transcript = []
    for lp in list_pieces:
        polarity_piece = []
        for p in lp:
            polarity_piece.append(TextBlob(str(p)).sentiment.polarity)
        polarity_transcript.append(polarity_piece)
    logger.info("Polarity calculated successfully!")
    return polarity_transcript

def plot_temporal_sentiment(df, polarity_transcript, plots_per_file=100):

    total_records = len(polarity_transcript)
    logger.info(f"Plotting temporal sentiment analysis for {total_records} records...")
    
    # THIS FORMULA ENSURES THAT ANY REMAINDER IS ACCOUNTED FOR BY ROUNDING UP THE DIVISION
    # 250 / 100 = 2.5 FILES
    # (250 + 100 - 1) // 100 = 3 FILES 
    #  "//" GIVES FLOOR DIVISION 
    num_files = (total_records + plots_per_file - 1) // plots_per_file
    
    for file_num in range(num_files):
        # Calculate start and end indices for current file
        start_idx = file_num * plots_per_file
        end_idx = min((file_num + 1) * plots_per_file, total_records)
        
        # Calculate grid dimensions
        num_rows = 25 # 25 rows
        num_cols = 4  # 4 columns to make 100 plots
        
        # Create figure
        fig = plt.figure(figsize=(20, 100))  # Large figure to accommodate all subplots
        
        # Create subplots for current file
        for idx in range(start_idx, end_idx):
            # Calculate subplot position
            plot_pos = idx - start_idx + 1
            ax = plt.subplot(num_rows, num_cols, plot_pos)
            
            # Plot data
            ax.plot(polarity_transcript[idx], 'b-', linewidth=1)
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            
            # Set title and customize appearance
            ax.set_title(df.iloc[idx]['comedian'], fontsize=8)
            ax.set_ylim(-0.2, 0.3)
            ax.grid(True, alpha=0.3)
            
            # Remove x-axis labels to save space
            ax.set_xticks([])
            
            # Only show y-axis labels for leftmost plots
            if plot_pos % num_cols != 1:
                ax.set_yticks([])
        
        # Adjust layout
        plt.tight_layout(pad=3.0)
        
        # Save file
        output_path = OUTPUT_PATH_VISUAL / f'02_Temporal_Sentiment_Batch_{file_num + 1}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved batch {file_num + 1} to: {output_path}")

def save_processed_data(df, list_pieces, polarity_transcript):

    # Add temporal polarity to dataframe
    df['temporal_polarity'] = polarity_transcript
    df['split_transcripts'] = list_pieces
    
    # Save complete processed dataframe
    df.to_pickle(PROCESSED_DATA_PKL)
    df.to_csv(PROCESSED_DATA_CSV)

def main():
    """
    Main function to orchestrate the entire sentiment analysis process
    """
    try:
        # Create output directory if it doesn't exist
        OUTPUT_PATH_VISUAL.mkdir(parents=True, exist_ok=True)
        
        # Try to load PKL file first, if not available, load CSV
        try:
            df = load_data(INPUT_PATH_PKL)
        except FileNotFoundError:
            df = load_data(INPUT_PATH_CSV)
        
        # Calculate overall sentiment metrics
        df = calculate_sentiment_metrics(df)
        
        # Plot and save overall sentiment analysis
        plot_sentiment_analysis(df, SENTIMENT_PLOT)
        
        # Create transcript pieces for temporal analysis
        list_pieces = create_transcript_pieces(df)
        print(f"Split {len(list_pieces)} transcripts into {len(list_pieces[0])} pieces each")
        
        # Calculate polarity over time
        polarity_transcript = calculate_polarity_over_time(list_pieces)
        
                
        # Save all processed data
        save_processed_data(df, list_pieces, polarity_transcript)
        
        # Plot and save temporal sentiment analysis
        plot_temporal_sentiment(df, polarity_transcript)

        
        print("Complete sentiment analysis completed successfully!")
        print(f"Processed data saved to: {PROCESSED_DATA_PKL}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()