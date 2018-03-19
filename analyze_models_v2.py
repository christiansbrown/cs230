"""" Goal of script is to assess activations/etc of the model """

# Flow of script:

# Load best models
# - Sentiment prediction model
# - Era prediction model
#
# Load all reviews, tags into a single dataset
#
# For each model:
# 	Feed all data into model and output all activations as arrays
# 	For activations of a given sample:
#	- Compute cosine similarity between last activation and all prior activations
# 	- *** if similarity of activations no longer changing significantly ***
#		- Store word/add count to word
#
# Sort identified keywords by count
# 

# Import relevent modules
import os
import argparse
import pickle

import numpy as np
import tensorflow as tf

from model.utils import Params
from model.model_fn import model_fn
from model.input_fn import input_fn
from model.input_fn import load_dataset_from_text
from collections import Counter

# Define keywords function (change as necessary)
# Finds keywords given a sequence of similarities in sequence
def find_keywords(similarities):
    """
    Returns keyPoints: [[word_position, similarity], [,] ..]
    Positions and similarities of words that are possible keywords
    """
    
    # Initialize lists to hold some useful values
    first_derivs = []
    second_derivs = []
    d_first_derivs = [] # difference in first derivatives
    
    for i in range(0,len(similarities)-2):
        
        df1 = similarities[i+1] - similarities[i]
        df2 = similarities[i+2] - 2*similarities[i+1] + similarities[i]
        
        first_derivs.append(df1)
        second_derivs.append(df2)
        
        if i > 0:
            
            d_df1 = abs(first_derivs[i] - first_derivs[i-1])
            d_first_derivs.append(d_df1)            
    
    # Try keywords ideas here
    keyPoints = []    
    intThresh = .01*max(similarities)
    count = 0

    for i, first_deriv in reversed(list(enumerate(first_derivs))):
        
        if abs(first_deriv) > intThresh and count <= 10:
            
            keyPoints.append([i, similarities[i]])
            count += 1
        
    return keyPoints

# For inputting arguments from console
parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Directory containing params.json")
parser.add_argument('--data_dir', default='data/kaggle/all', help="Directory containing the dataset")
parser.add_argument('--restore_from', default='best_weights',
                    help="Subdirectory of model dir or file containing the weights")
parser.add_argument('--is_toy', default='False')

args = parser.parse_args()

if args.is_toy == 'False':
	toy = ''
elif args.is_toy == 'True':
	toy = '_small'
else:
	raise ValueError("Please specify is_toy as either 'True' or 'False'")

# Load parameters of model
sentiment_model_path = os.path.join(args.model_dir, 'params.json')
params_sentiment = Params(sentiment_model_path)


# Load parameters from the dataset (sizes, etc) into params
# data_params_path = os.path.join(args.data_dir, 'dataset_params_small.json')
data_params_path = os.path.join(args.data_dir, 'dataset_params{}.json'.format(toy))
params_sentiment.update(data_params_path)
num_oov_buckets = params_sentiment.num_oov_buckets

# Update model params to include number of tags attribute
params_sentiment.number_of_tags = params_sentiment.number_of_sentiments

# Get paths for vocabularies and dataset
# path_words = os.path.join(args.data_dir, 'words_small.txt')
path_words = os.path.join(args.data_dir, 'words{}.txt'.format(toy))
path_sentiment_tags = os.path.join(args.data_dir, 'sentiment_tags.txt')
# path_era_tags = os.path.join(args.data_dir, 'era_tags.txt')
# path_reviews = os.path.join(args.data_dir, 'reviews_small.txt')
path_reviews = os.path.join(args.data_dir, 'reviews{}.txt'.format(toy))
path_sentiments = os.path.join(args.data_dir, 'sentiments{}.txt'.format(toy))
# path_sentiments = os.path.join(args.data_dir, 'sentiments.txt')
# path_eras = os.path.join(args.data_dir, 'eras_small.txt')

# Load vocabularies
words = tf.contrib.lookup.index_table_from_file(path_words, num_oov_buckets=num_oov_buckets)
sentiments = tf.contrib.lookup.index_table_from_file(path_sentiment_tags)
# eras = tf.contrib.lookup.index_table_from_file(path_era_tags)

# Create the input data pipeline
reviews = load_dataset_from_text(path_reviews,words)
review_sentiments = load_dataset_from_text(path_sentiments,sentiments, isLabels=True)
# review_eras = load_dataset_from_text(path_eras,eras)

# Specify other parameters for the dataset and the model
params_sentiment.id_pad_word = words.lookup(tf.constant(params_sentiment.pad_word))
params_sentiment.id_pad_tag = words.lookup(tf.constant(params_sentiment.pad_tag))

# Create the iterator over the test set
inputs_sentiment = input_fn('eval', reviews, review_sentiments, params_sentiment)
# inputs_eras = input_fn('eval', reviews, review_eras, params_era)

# Define the model
print('Creating sentiment and era models...')
model_spec_sentiment = model_fn('eval', inputs_sentiment, params_sentiment, reuse=False)
# model_spec_era = model_fn('eval', inputs_era, params_era, reuse=False)
print('Done')

# Evaluate the model... 
# evaluate(model-spec, args.model_dir, params, args.restore_from)

# initialize saver to restore model
saver = tf.train.Saver()

with tf.Session() as sess:
	# Initialize lookup tables for both models
	sess.run(model_spec_sentiment['variable_init_op'])

	# Reload weights from the weights subdirectory
	save_path = os.path.join(args.model_dir, args.restore_from)
	if os.path.isdir(save_path):
		save_path = tf.train.latest_checkpoint(save_path)
	saver.restore(sess, save_path)

	# Evaluate
	num_steps = (params_sentiment.dataset_size + params_sentiment.batch_size - 1) // params_sentiment.batch_size
	# Figure out what is going on in this next step...
	# Evaluate sess call happens here
	# metrics = evaluate_sess(sess, model_spec, num_steps) 

	# Update the model, obtain outputs
	update_metrics = model_spec_sentiment['update_metrics']
	eval_metrics= model_spec_sentiment['metrics']
	outputs = model_spec_sentiment['outputs']
	predictions = model_spec_sentiment['predictions']
	labels = model_spec_sentiment['labels']
	sentences = model_spec_sentiment['sentence']

	global_step = tf.train.get_global_step()

	# Load the dataset into the pipeline and initialize the metrics init op
	sess.run(model_spec_sentiment['iterator_init_op'])
	sess.run(model_spec_sentiment['metrics_init_op'])

	# Compute metrics over the dataset
	output_vals = []
	prediction_vals = []
	labels_vals = []
	sentence_vals = []
	# Maybe the shape is different so I am returning something different...?

	for i in range(num_steps):
		print('step number:',i)
		sess.run(update_metrics)

		step_output, step_pred, step_labels, step_sentences = sess.run([outputs, predictions, labels, sentences])

		# output_vals.append(sess.run(outputs))
		# prediction_vals.append(sess.run(predictions))
		# labels_vals.append(sess.run(labels))
		# sentence_vals.append(sess.run(sentences))

		output_vals.append(step_output)
		prediction_vals.append(step_pred)
		labels_vals.append(step_labels)
		sentence_vals.append(step_sentences)

	# Extract values for metrics
	metrics_values = {k: v[0] for k, v in eval_metrics.items()}
	metrics_val = sess.run(metrics_values)
	metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_val.items())

	# Extract values for outputs
	# outputs = sess.run(outputs)

# Convert to arrays?
# output_vals = np.array(output_vals)
# prediction_vals = np.array(prediction_vals)
# labels_vals = np.array(labels_vals)
# sentence_vals = np.array(sentence_vals)

pkl_output = output_vals#[2]
pkl_preds = prediction_vals#[2]
pkl_labels = labels_vals#[2]
pkl_sentences = sentence_vals#[2]

# print(np.shape(output_vals))
# print(np.shape(prediction_vals))
# print(np.shape(labels_vals))
# print(np.shape(sentence_vals))

# print(np.shape(pkl_output))
# print(np.shape(pkl_preds))
# print(np.shape(pkl_labels))
# print(np.shape(pkl_sentences))

# exit(0)


print(' - Done!')
# exit(0)

# write to pickle for tentative analysis
# pickle.dump(output_vals, open( "output_vals_small.pkl", "wb" ) 
pickle.dump(pkl_output, open( "output_vals.pkl", "wb" ) )
pickle.dump(pkl_preds, open( "prediction_vals.pkl", "wb" ) )
pickle.dump(pkl_labels, open( "labels_vals.pkl", "wb" ) )
pickle.dump(pkl_sentences, open( "sentence_vals.pkl", "wb" ) )


#%% Putting it all together - analysis pipeline
      
# Read in files from pickle (shouldn't be necessary in AWS)  
# Nesting: [epochs, batch_size, values]
#       - Values: seq length/activations, labels, etc.
output_vals = pickle.load(open("output_vals.pkl","rb"))
pred_vals = pickle.load(open("prediction_vals.pkl","rb"))
label_vals = pickle.load(open("labels_vals.pkl","rb"))
sentence_vals = pickle.load(open("sentence_vals.pkl","rb"))

# Obtain mapping from word_id to words
word_map = {}
vocab_path = 'data/kaggle/all/words.txt' # Change as necessary
with open(vocab_path) as f:
    word_map = {i:word.strip() for i, word in enumerate(f,1)}

# Iterate through data and build up counters
good_keywords = Counter() 
bad_keywords = Counter()   
    
# Outputs for current epoch [batch_size, max_seq_length, activations]   
for epoch_id, epoch_outputs in enumerate(output_vals):

    print('Epoch {} of {}'.format(epoch_id+1, len(output_vals)))
    
    # Obtain current epochs predictions, labels, tokenized reviews
    epoch_labels = label_vals[epoch_id]
    epoch_preds = pred_vals[epoch_id]
    epoch_reviews = sentence_vals[epoch_id]
    
    correct_count = 0
    
    # Sequence of activations for each example in batch [max_seq_length, acts]
    for review_id, activations in enumerate(epoch_outputs):     
    
        # Obtain current reviews preds, labels, tokenized review
        label = epoch_labels[review_id]
        pred = epoch_preds[review_id]
        review = epoch_reviews[review_id] # Sequence of integer IDs
        
        # Obtain last activation
        similarities = []
        last_activation = activations[-1]
        
        # Index of current activation, h-dimensional activation for current wrd
        for idx, curr_activation in enumerate(activations):
    
            # Compute similarity
            similarity = cosine(curr_activation, last_activation)
            similarities.append(similarity)
            
        # Investigate keywords if model predicted correct
        
        if label == pred:
            correct_count += 1
        
        if True:#label == pred:
            
            
            # Identify keywords
            keyPoints = find_keywords(similarities)
            word_positions, _ = zip(*keyPoints)
            
            # Find associated word_ids and words
            word_ids = [review[i] for i in word_positions]
            words = [word_map[k+1] for k in word_ids]            
            
            
            # Update good or bad keywords counter depending on result
            if label == 0:
                bad_keywords.update(words)
            else:
                good_keywords.update(words)

    print(' Predicted {} of {} sentiments correctly'.format(
                                        correct_count+1, len(epoch_outputs)))

    # Pickle the counter for analysis?
    pickle.dump(bad_keywords, open( "bad_keywords.pkl", "wb"))
    pickle.dump(good_keywords, open( "good_keywords.pkl", "wb"))    
    



# TODO: Dump a few other useful things as well


# read from cPickle
# output_vals = pickle.load( open( "output_vals_small.pkl", "rb" ) )


# Load vocabularies

# # Load data
# reviews = load_dataset_from_text(path_all_reviews, words)
# sentiment_labels = load_dataset_from_text(path_all_sentiments, tags)

# # Load best models
# sentiment_model_specs = model_fn('eval', sentiment_inputs, params, reuse=False)