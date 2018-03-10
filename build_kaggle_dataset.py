"""Read, split and save the kaggle dataset for our model"""

import csv
import os
import sys
import re
import numpy as np
import random
import nltk
from nltk.corpus import stopwords

def load_dataset(path_csv):
    """Loads dataset into memory from csv file"""
    # Open the csv file, need to specify the encoding for python3
    use_python3 = sys.version_info[0] >= 3
#    with (open(path_csv, encoding="windows-1252") if use_python3 else open(path_csv)) as f:
    with (open(path_csv, encoding="utf-8") if use_python3 else open(path_csv)) as f:
        csv_file = csv.reader(f, delimiter=',')
        dataset = []
#        words, tags = [], []
        
        # Define columns corresponding to dataset that we want to pull
        lsCols = [2, 3, 4, 13, 7, 6, 9, 12]

        # Each line of the csv corresponds to one word
        # Each line of the csv corresponds to a review
        for idx, row in enumerate(csv_file):
            if idx == 0: continue
#            sentence, word, pos, tag = row
            artist, title, label, tracklist, style, year, rating, review = (
                    [row[col] for col in lsCols])
            
            # Convert to string, append to dataset
            try:
                artist, title, label, tracklist, style, year, rating, review = (
                        str(artist), str(title), str(label), str(tracklist), 
                        str(style), str(year), str(rating), str(review))
                
                dataset.append([artist, title, label, tracklist, style, year, 
                                                                rating, review])
            except ValueError:
                print('Error appending review row to dataset on idx ',str(idx))
            
#            # If the first column is non empty it means we reached a new sentence
#            if len(sentence) != 0:
#                if len(words) > 0:
#                    assert len(words) == len(tags)
#                    dataset.append((words, tags))
#                    words, tags = [], []
#            try:
#                word, tag = str(word), str(tag)
#                words.append(word)
#                tags.append(tag)
#            except UnicodeDecodeError as e:
#                print("An exception was raised, skipping a word: {}".format(e))
#                pass

    return dataset

def process_dataset(dataset):
    """ Process the dataset further (remove markup, non-alpha, punctuation etc)
    
    Args:
        dataset: []

    
    """

    print('Processing dataset..')

    # Initialize new processed dataset (list) to return
    dataset_processed = []

    # Define regex helpers to clean review text
    regexBrackets = re.compile('<[^>]+>') # remove text between brackets
    regexAlpha = re.compile('[^a-z ]')
    
    # Define stopwords set to remove stopwords later
    stop_words = set(stopwords.words('english'))
       
    # Make changes to each entry
    intProgress = -1 # Used for outputting processing progress
    for idx, entry in enumerate(dataset):
        
        if np.mod(idx, len(dataset)//10) == 0:
            intProgress += 1
            print('  dataprocessing ', str(intProgress*10), '% complete')
        
        # Break apart current entry
        artist, title, label, tracklist, style, year, rating, review = entry
        
        # Weird edgecase - some reviews are in Japanese and cause problems
        # Japanese entries include strange formatting in tracklist
        # Use this formatting to skip try altogether (isalpha doesn't work)
        if '\n<a' not in tracklist: 
        
            tracks = tracklist.split() # Split tracklist into individual strings
    
            # Format review
            review = regexBrackets.sub(' ',review)
            review = review.replace("'",'')
            reviewWords = nltk.word_tokenize(review)
            reviewWords = [word for word in reviewWords if word.isalpha()]
    #        review = entry[-1]
    #        review = regexAlpha.sub(' ', review) 
    #        review = regexBrackets.sub(' ',review)
    #        review = review.replace("'",'') # Remove apostrophe (lazy...)
    #        review = review.replace('.',' ') # Remove period
    #        review = review.replace(',',' ') # Remove comma
            
            # Replace words (artist, tracklist, etc) from review
            prohibitedWords = entry[0:-1] #All elements in entry excet review
            tracklist = prohibitedWords[3] # Isolate tracklist to split apart
            tracklist = tracklist.replace(',', ' ')
            tracks = tracklist.split()
            del prohibitedWords[3] # Replace tracklist with individual tracks
            prohibitedWords.extend(tracks)
            reviewWords = [word for word in reviewWords if word not in prohibitedWords]
            
            # Remove stop words
            reviewWords = [word for word in reviewWords if word not in stop_words]
            
    #        regexWords = re.compile('|'.join(map(re.escape, prohibitedWords)))
    #        review = regexWords.sub(' ',review) # Remove words
            
            # Condense whitespace, add back to entry
            review = ' '.join(reviewWords)
            review = review.lower()
            
            # Check if review contains any non alpha characters (ie: japanese)
            # If it contains only alpha, proceed.. o/w do not
            review = regexAlpha.sub('!!!',review) # '!!!' to mark non alpha 
            if '!!!' not in review:
                
                entry[-1] = review             
        
                # Add labels for sentiment (score) and era (year of release)
                # Mean/median score is 3.7
                # Define three eras st each have approx same number of reviews
                # eras: 2001-2008, 2009-2013, 2014-2017
                
                # Assign positive/negative sentiment based off of score
                if float(rating) <= 3.7:
                    sentiment = 'negative'
                else:
                    sentiment = 'positive'
                
                # Assign era based off year of release
                if int(year) <= 2008:
                    era = '2001-2009'
                elif int(year) <= 2013:
                    era = '2009-2013'
                else:
                    era = '2014-2017'
                
                # Cast current entry as list, append labels, add to dataset
                entry = list(entry)
                entry.append(sentiment)
                entry.append(era)
                dataset_processed.append(entry)
    #            dataset[idx] = entry # Update current entry

    print('.. processing complete')
    return dataset_processed


def save_dataset(dataset, save_dir):
    """Writes sentences.txt and labels.txt files in save_dir from dataset

    Args:
        dataset: ([(["a", "cat"], ["O", "O"]), ...])
        save_dir: (string)
    """
    
    # Subset dataset to review, sentiment, era
    dataSub = [dataset[i][7:10] for i in range(0,len(dataset))]
    
    # Create directory if it doesn't exist
    print("Saving in {}...".format(save_dir))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Export the dataset
    with open(os.path.join(save_dir, 'reviews.txt'), 'w') as file_reviews:
        with open(os.path.join(save_dir, 'sentiments.txt'), 'w') as file_sentiments:
            with open(os.path.join(save_dir, 'eras.txt'),'w') as file_eras:
                for review, sentiment, era in dataSub:
                    #file_sentences.write("{}\n".format(" ".join(words)))
                    #file_labels.write("{}\n".format(" ".join(tags)))
                    file_reviews.write("{}\n".format("".join(review)))
                    file_sentiments.write("{}\n".format("".join(sentiment)))
                    file_eras.write("{}\n".format("".join(era)))
    print("- done.")


if __name__ == "__main__":
    # Check that the dataset exists (you need to make sure you haven't downloaded the `ner.csv`)
#    path_dataset = 'data/kaggle/ner_dataset.csv'
    path_dataset = 'data/RA_cleaned.csv'
    msg = "{} file not found. Make sure you have downloaded the right dataset".format(path_dataset)
    assert os.path.isfile(path_dataset), msg

    # Load the dataset into memory
    print("Loading Kaggle dataset into memory...")
    dataset = load_dataset(path_dataset)
    print("- done.")

    # Process dataset
    dataset_processed = process_dataset(dataset)

    # Split the dataset into train, dev and split
    # Shuffle with seed for reproducibility
    # To do: implement cross validation?
    m = len(dataset_processed) # number of samples after processing
    dataset_processed = random.Random(5).sample(dataset_processed, m)
    train_dataset = dataset_processed[:int(0.7*m)]
    dev_dataset = dataset_processed[int(0.7*m):int(0.85*m)]
    test_dataset = dataset_processed[int(0.85*m):]

    # Save datasets to file
    isToy = False # TODO: change this as necessary
    if isToy:
        # Toy datasets to test locally
        save_dataset(train_dataset[0:100], 'data/kaggle/train')
        save_dataset(dev_dataset[0:10], 'data/kaggle/dev')
        save_dataset(test_dataset[0:10], 'data/kaggle/test')
    else:
        # Full dataset
        save_dataset(train_dataset, 'data/kaggle/train')
        save_dataset(dev_dataset, 'data/kaggle/dev')
        save_dataset(test_dataset, 'data/kaggle/test')
        