"""Build vocabularies of words and tags from datasets"""

import argparse
from collections import Counter
import json
import os
import sys


parser = argparse.ArgumentParser()
parser.add_argument('--min_count_word', default=1, help="Minimum count for words in the dataset",
                    type=int)
parser.add_argument('--min_count_tag', default=1, help="Minimum count for tags in the dataset",
                    type=int)
parser.add_argument('--data_dir', default='data/kaggle', help="Directory containing the dataset")
parser.add_argument('--objective', default = 'sentiment', 
                    help="Define classification objective of model as either 'sentiment' or 'era'")
parser.add_argument('--is_split', default='True', help="Use all data or not? True/False")
parser.add_argument('--is_toy', default='False', help="True/False to use smaller dataset")

# Hyper parameters for the vocab
NUM_OOV_BUCKETS = 1 # number of buckets (= number of ids) for unknown words
PAD_WORD = '<pad>'
PAD_TAG = 'O'
#PAD_SENTIMENT = 'O'
#PAD_ERA = 'O'


def save_vocab_to_txt_file(vocab, txt_path):
    """Writes one token per line, 0-based line id corresponds to the id of the token.

    Args:
        vocab: (iterable object) yields token
        txt_path: (stirng) path to vocab file
    """
    with open(txt_path, "w") as f:
        f.write("\n".join(token for token in vocab))
        
    # Ensure that there are no empty lines written to file...
    f = open(txt_path,'r+')
    d = f.readlines()
    f.seek(0)
    for i in d:
        if i != '\n':
            f.write(i)
    f.truncate()
    f.close()


def save_dict_to_json(d, json_path):
    """Saves dict to json file

    Args:
        d: (dict)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        d = {k: v for k, v in d.items()}
        json.dump(d, f, indent=4)


def update_vocab(txt_path, vocab):
    """Update word and tag vocabulary from dataset

    Args:
        txt_path: (string) path to file, one sentence per line
        vocab: (dict or Counter) with update method

    Returns:
        dataset_size: (int) number of elements in the dataset
    """
    with open(txt_path) as f:
        for i, line in enumerate(f):
            vocab.update(line.strip().split(' '))


    return i + 1


if __name__ == '__main__':
    args = parser.parse_args()
    if args.is_split == "True":
        is_split = True
    elif args.is_split == "False":
        is_split = False
    else:
        raise ValueError("Please define is_split as either 'True' or 'False'")

    if args.is_toy == 'False':
        toy = ''
    elif args.is_toy == 'True':
        toy = '_small'
    else:
        raise ValueError("Please define is_toy as either 'True' or 'False'")

    # Build word vocab with train and test datasets (reviews)
    print("Building word vocabulary...")
    words = Counter()

    if is_split:
        print(' - split reviews')
        size_train_sentences = update_vocab(os.path.join(args.data_dir, 'train/reviews.txt'), words)
        size_dev_sentences = update_vocab(os.path.join(args.data_dir, 'dev/reviews.txt'), words)
        size_test_sentences = update_vocab(os.path.join(args.data_dir, 'test/reviews.txt'), words)
    else:
        print(' - all reviews')
        size_reviews = update_vocab(os.path.join(args.data_dir, 'reviews{}.txt'.format(toy)), words)
    print("- done.")

    # Build tag vocab with train and test datasets with inputted objective
    if is_split:
        if args.objective == 'sentiment':
            print("Building sentiments vocabulary...")
            tags = Counter()
            size_train_tags = update_vocab(os.path.join(args.data_dir, 'train/sentiments.txt'), tags)
            size_dev_tags = update_vocab(os.path.join(args.data_dir, 'dev/sentiments.txt'), tags)
            size_test_tags = update_vocab(os.path.join(args.data_dir, 'test/sentiments.txt'), tags)
            print("- done.")
        elif args.objective == 'era':
            print("Building eras vocabulary...")
            tags = Counter()
            size_train_tags = update_vocab(os.path.join(args.data_dir, 'train/eras.txt'), tags)
            size_dev_tags = update_vocab(os.path.join(args.data_dir, 'dev/eras.txt'), tags)
            size_test_tags = update_vocab(os.path.join(args.data_dir, 'test/eras.txt'), tags)
            print("- done.")
        else: raise ValueError("Invalid objective! Set as either 'sentiment' or 'era'")
    
            # Assert same number of examples in datasets
        assert size_train_sentences == size_train_tags
        assert size_dev_sentences == size_dev_tags
        assert size_test_sentences == size_test_tags

    else:
        print("Building vocabulary all folder...")
        sentiment_tags = Counter()
        size_sentiment_tags = update_vocab(os.path.join(args.data_dir, 'sentiments{}.txt'.format(toy)), sentiment_tags)
        era_tags = Counter()
        size_era_tags = update_vocab(os.path.join(args.data_dir, 'eras{}.txt'.format(toy)), era_tags)
        assert size_reviews == size_sentiment_tags
        assert size_reviews == size_era_tags
        print('-done')
#    # Build tag vocab with train and test datasets (sentiment)
#    print("Building sentiments vocabulary...")
#    sentiments = Counter()
#    size_train_sentiments = update_vocab(os.path.join(args.data_dir, 'train/sentiments.txt'), sentiments)
#    size_dev_sentiments = update_vocab(os.path.join(args.data_dir, 'dev/sentiments.txt'), sentiments)
#    size_test_sentiments = update_vocab(os.path.join(args.data_dir, 'test/sentiments.txt'), sentiments)
#    print("- done.")
#
#    # Build tag vocab with train and test datasets (ears)
#    print("Building eras vocabulary...")
#    eras = Counter()
#    size_train_eras = update_vocab(os.path.join(args.data_dir, 'train/eras.txt'), eras)
#    size_dev_eras = update_vocab(os.path.join(args.data_dir, 'dev/eras.txt'), eras)
#    size_test_eras = update_vocab(os.path.join(args.data_dir, 'test/eras.txt'), eras)
#    print("- done.")

    # Only keep most frequent tokens
    words = [tok for tok, count in words.items() if count >= args.min_count_word]
    if is_split:
        tags = [tok for tok, count in tags.items() if count >= args.min_count_tag]
#    sentiments = [tok for tok, count in sentiments.items() if count >= args.min_count_tag]
#    eras = [tok for tok, count in eras.items() if count >= args.min_count_tag]
    else:
        sentiment_tags = [tok for tok, count in sentiment_tags.items() if count >= args.min_count_tag]
        era_tags = [tok for tok, count in era_tags.items() if count >= args.min_count_tag]


    # Add pad tokens
    if PAD_WORD not in words: words.append(PAD_WORD)
    if is_split:
        if PAD_TAG not in tags: tags.append(PAD_TAG)
    else:
        if PAD_TAG not in sentiment_tags: 
            sentiment_tags.append(PAD_TAG)
        if PAD_TAG not in era_tags:
            era_tags.append(PAD_TAG)
#    if PAD_SENTIMENT not in sentiments: sentiments.append(PAD_SENTIMENT)
#    if PAD_ERA not in eras: eras.append(PAD_ERA)

    # Save vocabularies to file
    print("Saving vocabularies to file...")
    save_vocab_to_txt_file(words, os.path.join(args.data_dir, 'words.txt'))
    if is_split:
        save_vocab_to_txt_file(tags, os.path.join(args.data_dir, 'tags.txt'))
    else:
        save_vocab_to_txt_file(sentiment_tags, os.path.join(args.data_dir, 'sentiment_tags{}.txt'.format(toy)))
        save_vocab_to_txt_file(era_tags, os.path.join(args.data_dir, 'era_tags{}.txt'.format(toy)))
#    save_vocab_to_txt_file(sentiments, os.path.join(args.data_dir, 'sentiments.txt'))
#    save_vocab_to_txt_file(eras, os.path.join(args.data_dir, 'eras.txt'))
    print("- done.")

    # Save datasets properties in json file
    if is_split:
        sizes = {
            'train_size': size_train_sentences,
            'dev_size': size_dev_sentences,
            'test_size': size_test_sentences,
            'vocab_size': len(words) + NUM_OOV_BUCKETS,
            'number_of_tags': len(tags),
    #        'number_of_sentiments': len(sentiments),
    #        'number of eras':len(eras),
            'pad_word': PAD_WORD,
            'pad_tag': PAD_TAG,
    #        'pad_sentiment': PAD_SENTIMENT,
    #        'pad_era':PAD_ERA,
            'num_oov_buckets': NUM_OOV_BUCKETS
        }
    else:
        sizes = {
            'dataset_size': size_reviews,
            'vocab_size': len(words) + NUM_OOV_BUCKETS,
            'number_of_sentiments': len(sentiment_tags),
            'number_of_eras': len(era_tags),
    #        'number_of_sentiments': len(sentiments),
    #        'number of eras':len(eras),
            'pad_word': PAD_WORD,
            'pad_tag': PAD_TAG,
    #        'pad_sentiment': PAD_SENTIMENT,
    #        'pad_era':PAD_ERA,
            'num_oov_buckets': NUM_OOV_BUCKETS
        }

    save_dict_to_json(sizes, os.path.join(args.data_dir, 'dataset_params{}.json'.format(toy)))

    # Logging sizes
    to_print = "\n".join("- {}: {}".format(k, v) for k, v in sizes.items())
    print("Characteristics of the dataset:\n{}".format(to_print))