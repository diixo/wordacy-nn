'''
code: text_generation_with_miniature_gpt.py
'''
import re
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import TextVectorization
import numpy as np
import os
import string
import random
from keras.datasets import imdb
from latin_symbols_demo import transliterate_lower


"""
## Implement the miniature GPT model
"""
vocab_size = 20000      # Only consider the top 20k words
maxlen = 80             # Max sequence size
embed_dim = 256         # Embedding size for each token
num_heads = 2           # Number of attention heads
feed_forward_dim = 256  # Hidden layer size in feed forward network inside transformer

batch_size = 128


(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)

################################################################################
word_to_index = imdb.get_word_index()
index_to_word = dict([(value, key) for (key, value) in word_to_index.items()])
################################################################################
def detokenize(index_array):
    return " ".join([transliterate_lower(index_to_word.get(i, " ")) for i in index_array])
################################################################################

txt_lines  = []
txt_lines.extend([detokenize(review) for review in x_train])
txt_lines.extend([detokenize(review) for review in x_test])
print("Train sequences: ", len(txt_lines))
################################################################################

#txt_lines = txt_lines[591:599]

# Create a dataset from text files
random.shuffle(txt_lines)
text_ds = tf.data.Dataset.from_tensor_slices(txt_lines)
text_ds = text_ds.shuffle(buffer_size=256)
text_ds = text_ds.batch(batch_size)



def custom_standardization(input_string):
    lowercased = tf.strings.lower(input_string)
    return lowercased
    # stripped_html = tf.strings.regex_replace(lowercased, "<br />", " ")
    # result = tf.strings.regex_replace(stripped_html, f"([{string.punctuation}])", r" \1")
    # return tf.strings.unicode_decode(result, input_encoding='utf-8', errors='ignore')
################################################################################


# Create a vectorization layer and adapt it to the text
vectorize_layer = TextVectorization(
    standardize=custom_standardization,
    max_tokens=vocab_size - 1,
    output_mode="int",
    output_sequence_length=maxlen + 1,
)
vectorize_layer.adapt(text_ds)
vocab = vectorize_layer.get_vocabulary()
print("vocabulary", len(vocab))


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
        break

if __name__ == "__main__":
    print_out()
