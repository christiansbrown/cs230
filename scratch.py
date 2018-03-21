"""

Scratchwork to understand what I am doing..

"""
#%%
import pickle
import os

import numpy as np
import random

from scipy.spatial.distance import cosine
from matplotlib import pyplot as plt
from collections import Counter


#%% Read saved activations from analysis.py
"""
Shape of output vals: [num_epochs, batch_size, max_seq_length, activations]
Current example:
    num_epochs: 4
    batch_size: 32
    max_seq_length: 400 something?
    activations: 25
    
Notes: 
    There are two extra entries in here... probably coming dataset.repeat to 
    get the code to compile correctly.
    Lets figure out which ones these are....    

"""

#%% Read saved inputs (sentence, labels) and outputs (outputs, preds)
# read from pickle

obj = '_era'

output_vals = pickle.load(open("output_vals{}.pkl".format(obj),"rb"))
pred_vals = pickle.load(open("pred_vals{}.pkl".format(obj),"rb"))
label_vals = pickle.load(open("label_vals{}.pkl".format(obj),"rb"))
sentence_vals = pickle.load(open("sentence_vals{}.pkl".format(obj),"rb"))

#%% Read in vocabulary and rebuild mapping? Hopefully it works!

word_map = {}
vocab_path = 'data/kaggle/all/words.txt'
with open(vocab_path) as f:
    word_map = {i:word.strip() for i, word in enumerate(f,1)}
    
#%% See if we can rebuild a sentence...

epoch_id = 1
review_id = 13
sentence = sentence_vals[epoch_id][review_id] # ids of sentences

sentence_text = ''
for word_id in sentence:
    if word_id+1 == len(word_map): break
    sentence_text += word_map[word_id+1] + ' '
    
print(sentence_text)
print('pred:',pred_vals[epoch_id][review_id])
print('label:',label_vals[epoch_id][review_id][0])


#%% Print out stuff to visualize/fiddle with
plt.close('all')

# We read in a single epochs worth of information
similarity_vals = []

for epoch_id, epoch_vals in enumerate(output_vals):

    epoch_similarities = []
    
    for review_id, review in enumerate(epoch_vals):
        
        
        # Obtain last activation
        similarities = []
        last_activation = review[-1]
        
        for ii in range(0,len(review)-1):
            
            # Obtain current activation
            curr_activation = review[ii]
            similarity = cosine(curr_activation, last_activation)
            similarities.append(similarity)
            
            # Compute and store cosine similarities
        
        # Display results only for reviews that made the correct prediction
        curr_pred = pred_vals[epoch_id][review_id]
        curr_label = label_vals[epoch_id][review_id]
        
    #    if curr_pred == curr_label:
            
        if review_id in range(0,10) and epoch_id == 1:
            
            with plt.style.context('ggplot'):
                plt.figure(figsize = (12,8)) 
                plt.plot(similarities)
                strTitle = str(curr_pred) + ': Cosine similarities for review_id ' + str(review_id)
                plt.title(strTitle)
    
    
        epoch_similarities.append(similarities)
        
    similarity_vals.append(epoch_similarities)
        
    
#%% Tentative idea - Compute first/second derivatives and visualize
plt.close('all')    
    
epoch_id = 0

for review_id in range(0,10):
    
    similarities = similarity_vals[epoch_id][review_id]
    sentence = sentence_vals[epoch_id][review_id]
    label = label_vals[epoch_id][review_id]
    pred = pred_vals[epoch_id][review_id]
    
    first_derivs = []
    second_derivs = []
    
    d_first_derivs = []
    
    for i in range(0,len(similarities)-2):
        
        df1 = similarities[i+1] - similarities[i]
        df2 = similarities[i+2] - 2*similarities[i+1] + similarities[i]
        
        first_derivs.append(df1)
        second_derivs.append(df2)
        
        if i > 0:
            
            d_df1 = abs(first_derivs[i] - first_derivs[i-1])
            d_first_derivs.append(d_df1)
        
    
    # Define criteria to select key words?  Ideas...
    # - abs(d_df1) goes from above thresh to below thresh    
        
    # Try keyword ideas here
    keyPoints = []    
    intThresh = .01*max(similarities)
    count = 0
    
    pad_id = np.max(sentence)
    
    for i, first_deriv in reversed(list(enumerate(first_derivs))):
        
        if abs(first_deriv) > intThresh and count <= 10 and sentence[i] != pad_id:
            
            keyPoints.append([i, similarities[i]])
            count += 1
    
#    for i in range(1,len(d_first_derivs)):
#        
#                
#        # Adding points resulting in a large change in similarity
#        if (abs(d_first_derivs[i]) > intThresh and i > 10 
#            and abs(first_derivs[i]) > abs(first_derivs[i-1])):
#            keyPoints.append([i,similarities[i]])
        
        
#        if abs(d_first_derivs[i]) > 3*abs(d_first_derivs[i-1]) and abs(d_first_derivs[i]) > intThresh:
#            if i > 5: keyPoints.append([i,similarities[i]])
            
            # Also include inflection points?
            
        
#        if d_first_derivs[i] > intThresh and i > 5:
#            keyPoints.append([i,similarities[i]])    
#        
#        # Adding points in the knee region
#        # Counter to measure flatness of similarities
#        if similarities[i] < intThresh:
#            count += 1
#        else:
#            count = 0
#            potential_points = []
#            
#        if count in range(0,8):
#            potential_points.append([i-1, similarities[i-1]])
#        
#        if count == 10:
#            for x,y in potential_points:
#                keyPoints.append([x,y])
                
        # Go through list backwards, once derivative starts to climb
        # - so long as derivative is still climbing, record a few of the points
        
        
    #    if abs(d_first_derivs[i]) < intThresh and d_first_derivs[i-1] > intThresh:
    #        keyPoints.append([[i-1],similarities[i-1]])
        
    #    if abs(d_first_derivs[i]) < intThresh and abs(d_first_derivs[i-1]) > intThresh:
    #        keyPoints.append([i-1, similarities[i-1]])
        
    #    diff = abs(abs(d_first_derivs[i]) - abs(d_first_derivs[i-1]))    
    #    if diff > intThresh:        
    #        keyPoints.append([i, similarities[i]])
            
#    # Plot results
#    with plt.style.context('ggplot'):
#        plt.figure(figsize = (9,6))    
#    #    plt.plot(first_derivs, alpha = .4)
#        plt.plot(d_first_derivs, alpha = .4)    
#        plt.plot(similarities, alpha = .9)`
    #    x, y = zip(*keyPoints)
    #    plt.scatter(x,y)
#        plt.legend(['df1','d_df1','Similarities'])    
    
    with plt.style.context('ggplot'):
        plt.figure(figsize=(9,6))
        plt.plot(similarities, alpha = .8)
        
        x,y = zip(*keyPoints)
        plt.scatter(x,y, c = 'b', marker = 'x')
        
        first_pad = np.argmax(sentence>=max(sentence))
        plt.plot([first_pad,first_pad],[0,max(similarities)], c='k', alpha=.3)
        
        plt.legend(['similarities','EOS', 'key word'])

        strTitle = 'Pred: {}, Label: {}, Pad_pos: {}'.format(pred,label,first_pad)
        plt.title(strTitle)
        
    
# Also - return the keywords and add it to a dictionary?
    
    
#%% Putting it all together - analysis pipeline
      
# Define keywords function (change as necessary)
# Finds keywords given a sequence of similarities in sequence
def find_keywords(similarities, review):
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

    # ID for pad word, so we don't add <pad> to keywords
    pad_id = np.max(review)

    for i, first_deriv in reversed(list(enumerate(first_derivs))):
        
        if abs(first_deriv) > intThresh and count <= 10 and review[i] != pad_id:
            
            keyPoints.append([i, similarities[i]])
            count += 1
        
    return keyPoints
        
# Remapping prediction IDs because it is inconsistent with labels
def remap_preds(preds):
    
    for i, pred in enumerate(preds):
        if pred == 0:
            preds[i] = 1
        elif pred == 1:
            preds[i] = 2
        else:
            preds[i] = 0
    
    return preds

 
# Read in files from pickle (shouldn't be necessary in AWS)  
# Nesting: [epochs, batch_size, values]
#       - Values: seq length/activations, labels, etc.
output_vals = pickle.load(open("output_vals{}.pkl".format(obj),"rb"))
pred_vals = pickle.load(open("pred_vals{}.pkl".format(obj),"rb"))
label_vals = pickle.load(open("label_vals{}.pkl".format(obj),"rb"))
sentence_vals = pickle.load(open("sentence_vals{}.pkl".format(obj),"rb"))

# Obtain mapping from word_id to words
word_map = {}
vocab_path = 'data/kaggle/all/words.txt' # Change as necessary
with open(vocab_path) as f:
    word_map = {i:word.strip() for i, word in enumerate(f,1)}

# Iterate through data and build up counters
if obj == '_sent':
    good_keywords = Counter() 
    bad_keywords = Counter()   
else: 
    era1_keywords = Counter() # 2001-2008
    era2_keywords = Counter() # 2009-2013
    era3_keywords = Counter() # 2013-2017

# Outputs for current epoch [batch_size, max_seq_length, activations]   
for epoch_id, epoch_outputs in enumerate(output_vals):

    print('Epoch {} of {}'.format(epoch_id+1, len(output_vals)))
    
    # Obtain current epochs predictions, labels, tokenized reviews
    epoch_labels = label_vals[epoch_id]
    epoch_preds = remap_preds(pred_vals[epoch_id])
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
        
#        if True:#label == pred:
        if label == pred:
            
            # Identify keywords
            keyPoints = find_keywords(similarities)
            word_positions, _ = zip(*keyPoints)
            
            # Find associated word_ids and words
            word_ids = [review[i] for i in word_positions]
            words = [word_map[k+1] for k in word_ids]
            
            # Update good or bad keywords counter depending on result
            if obj == '_era':
                if label == 0:
                    era1_keywords.update(words)
                elif label == 1:
                    era2_keywords.update(words)
                else:
                    era3_keywords.update(words)
            else:
                if label == 1:
                    bad_keywords.update(words)
                else:
                    good_keywords.update(words)
            

    print(' Predicted {} of {} classes correctly'.format(
                                        correct_count, len(epoch_outputs)))

    # Pickle the counter for analysis?
#    pickle.dump(bad_keywords, open( "bad_keywords.pkl", "wb"))
#    pickle.dump(good_keywords, open( "good_keywords.pkl", "wb"))    
    
#%% Open pkl filesf rom AWS
    
#output_vals = pickle.load(open("output_vals_sent.pkl","rb"))


good_keywords = pickle.load(open("good_keywords{}.pkl".format(obj),"rb"))
bad_keywords = pickle.load(open("bad_keywords{}.pkl".format(obj),"rb"))

#%% Quickly analyze sets (sentiments)

# TODO: Double check that good/bad mapping is done correctly?

n = 200 # number of most common words to return

top_good = good_keywords.most_common(n)
top_bad = bad_keywords.most_common(n)

good_words, _ = zip(*top_good)
good_words = set(good_words)
bad_words, _ = zip(*top_bad)
bad_words = set(bad_words)

good_unique = good_words - bad_words
bad_unique = bad_words - good_words

print(good_unique)
print(bad_unique)

#%% Quickly analze sets (eras)

n = 50

top_era1 = era1_keywords.most_common(n)
top_era2 = era2_keywords.most_common(n)
top_era3 = era3_keywords.most_common(n)

era1_words, _ = zip(*top_era1)
era1_words = set(era1_words)
era2_words, _ = zip(*top_era2)
era2_words = set(era2_words)
era3_words, _ = zip(*top_era3)
era3_words = set(era3_words)

era1_unique = era1_words - era2_words - era3_words
era2_unique = era2_words - era1_words - era3_words
era3_unique = era3_words - era1_words - era2_words

print(era1_unique)
print(era2_unique)
print(era3_unique)

#%% Combine sets together to think of some cool stuff...

era1_good = set.intersection(good_unique, era1_unique)
print('era1 good')
print(era1_good)
print('')

era2_good = set.intersection(good_unique, era2_unique)
print('era2 good')
print(era2_good)
print('')

era3_good = set.intersection(good_unique, era3_unique)
print('era3 good')
print(era3_good)
print('')

era1_bad = set.intersection(bad_unique, era1_unique)
print('era1 bad')
print(era1_bad)
print('')

era2_bad = set.intersection(bad_unique, era2_unique)
print('era2 bad')
print(era2_bad)
print('')

era3_bad = set.intersection(bad_unique, era3_unique)
print('era3 bad')
print(era3_bad)
print('')

era1_na = era1_unique - era1_good - era1_bad
print('era1 na')
print(era1_na)
print('')

era2_na = era2_unique - era2_good -era2_bad
print('era2 na')
print(era2_na)
print('')

era3_na = era3_unique - era3_good - era3_bad
print('era3 na')
print(era3_na)
print('')




#%% Open up all similarities?

similarity_vals_sent = pickle.load(open("similarity_vals_sent.pkl","rb"))
similarity_vals_era = pickle.load(open("similarity_vals_era.pkl","rb"))


#%% Make a few plots of the data and stuff!
plt.close('all')
"""
For a given epoch_id and review_id
- Review
- Era
    - Prediction
    - Label
    - Keywords (Position in sequence, word)
    - Similarity plot
- Sentiment
    - Prediction
    - Label
    - Keywords (Position in sequence, word)
    - Similarity Plot
-
"""

epoch_id = 5 #6
review_id = 20 #25

sentence_ids = sentence_vals[0][0]
sentence = ''
for word_id in sentence_ids:
    if word_id+1 == len(word_map): break
    sentence += word_map[word_id+1] + ' '
     
# Read in files from pickle (shouldn't be necessary in AWS)  
# Nesting: [epochs, batch_size, values]
#       - Values: seq length/activations, labels, etc.
    
for i in range(0,2):
    
    if i == 1: # Era
        obj = '_era'
    else:
        obj = '_sent'
    
    # Import
    output_vals = pickle.load(open("output_vals{}.pkl".format(obj),"rb"))
    pred_vals = pickle.load(open("pred_vals{}.pkl".format(obj),"rb"))
    label_vals = pickle.load(open("label_vals{}.pkl".format(obj),"rb"))
    sentence_vals = pickle.load(open("sentence_vals{}.pkl".format(obj),"rb"))
    similarity_vals = pickle.load(open("similarity_vals{}.pkl".format(obj),"rb"))
    
    # Get current epoch_id/review_id examples
    activations = output_vals[epoch_id][review_id]
    pred = pred_vals[epoch_id][review_id]
    label = label_vals[epoch_id][review_id]
    
    # Remap pred if era
    if i == 1:
        if pred == 0:
            pred = 1
        elif pred == 1:
            pred = 2
        else:
            pred = 0
    
    # Build and display sentence
    sentence_ids = sentence_vals[epoch_id][review_id]
    review = ''
    for word_id in sentence_ids:
        if word_id+1 == len(word_map): break
        review += word_map[word_id+1] + ' '

    # Obtain current set of similarities?
    similarities = []
    last_activation = activations[-1]
    for idx, curr_activation in enumerate(activations):
        similarity = cosine(curr_activation, last_activation)
        similarities.append(similarity)
        
    # Identify keyPoints
    keyPoints = find_keywords(similarities, sentence_ids)
    word_positions, _ = zip(*keyPoints)
    
    # Find associated word_ids and words
    word_ids = [sentence_ids[i] for i in word_positions]
    words = [word_map[k+1] for k in word_ids]  
        
    words_zipped = [[pos, word] for pos, word in zip(word_positions, words)]
    
    # Plot the results
    with plt.style.context('ggplot'):
        
        plt.figure(figsize=(16,6))
        
        plt.plot(similarities, alpha = .8)
        
        x,y = zip(*keyPoints)
        plt.scatter(x, y, c = 'b', marker = 'x')
        
        first_pad = np.argmax(sentence_ids>=max(sentence_ids))
        plt.xlim([0,first_pad])
#        plt.plot([first_pad,first_pad],[0,max(similarities)], c='k', alpha=.3)
        
        plt.legend(['Similarities','Keywords'])
    
        
#        strTitle = '{} - Pred: {}, Label: {}, Pad_pos: {}'.format(obj, pred,label,first_pad)
        
        if i == 1:
            strTitle = 'Era Classification: Activation Similarities vs. Word Position'
        else:
            strTitle = 'Sentiment Classification: Activation Similarities vs. Word Position'
        
        plt.title(strTitle)
        plt.xlabel('Word position')
        plt.ylabel('Similarity to final activation')
        
    # Output Various Results
    print('Epoch ID: {}, Review ID: {}'.format(epoch_id, review_id))
    print('Review text:')
    print(review)
    print('')
    if i == 1: print('Era')
    else: print('Sentiment')
    print('Label: {}, Prediction: {}'.format(label, pred))
    print('')
    print('Keywords:')
    print(words_zipped)
    
        
        

