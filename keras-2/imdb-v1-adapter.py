
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import TextVectorization
import numpy as np
import os
import string
import random
from keras.datasets import imdb


"""
## Implement the miniature GPT model
"""
vocab_size = 20000      # Only consider the top 20k words
maxlen = 80             # Max sequence size
embed_dim = 256         # Embedding size for each token
num_heads = 2           # Number of attention heads
feed_forward_dim = 256  # Hidden layer size in feed forward network inside transformer

batch_size = 128


"""shell
curl -O https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
tar -xf aclImdb_v1.tar.gz
"""


# The dataset contains each review in a separate text file
# The text files are present in four different folders
# Create a list all files
filenames = []
directories = [
    "E:/aclImdb/train/pos",
    "E:/aclImdb/train/neg",
    "E:/aclImdb/test/pos",
    "E:/aclImdb/test/neg",
]
for dir in directories:
    for f in os.listdir(dir):
        filenames.append(os.path.join(dir, f))

print(f"{len(filenames)} files")

# Create a dataset from text files
random.shuffle(filenames)
text_ds = tf.data.TextLineDataset(filenames)
text_ds = text_ds.shuffle(buffer_size=256)
text_ds = text_ds.batch(batch_size)


def custom_standardization(input_string):
    """Remove html line-break tags and handle punctuation"""
    lowercased = tf.strings.lower(input_string)
    stripped_html = tf.strings.regex_replace(lowercased, "<br />", " ")
    return tf.strings.regex_replace(stripped_html, f"([{string.punctuation}])", r" \1")


# Create a vectorization layer and adapt it to the text
vectorize_layer = TextVectorization(
    standardize=custom_standardization,
    max_tokens=vocab_size - 1,
    output_mode="int",
    output_sequence_length=maxlen + 1,
)
vectorize_layer.adapt(text_ds)
vocab = vectorize_layer.get_vocabulary()  # To get words back from token indices


def prepare_lm_inputs_labels(text):
    """
    Shift word sequences by 1 position so that the target for position (i) is
    word at position (i+1). The model will use all words up till position (i)
    to predict the next word.
    """
    text = tf.expand_dims(text, -1)
    tokenized_sentences = vectorize_layer(text)
    x = tokenized_sentences[:, :-1]
    y = tokenized_sentences[:, 1:]
    return x, y


text_ds = text_ds.map(prepare_lm_inputs_labels, num_parallel_calls=tf.data.AUTOTUNE)
text_ds = text_ds.prefetch(tf.data.AUTOTUNE)

###############################

# Создание обратного маппинга (индекс -> слово)
index_lookup = dict(enumerate(vocab))

def indices_to_text(indices):
    words = [index_lookup.get(index, "") for index in indices]
    return ' '.join(words)

def print_out():
    # inputs, targets: shape(batch_size, maxlen)
    for inputs, targets in text_ds:
        for i in range(inputs.shape[0]):
            t0 = inputs[i,:].numpy()
            t1 = targets[i,:].numpy()

            print(">>", indices_to_text(t0))
            print(">>", indices_to_text(t1))
            print("="*65)


print_out()
