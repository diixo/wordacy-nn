'''
Implement the miniature GPT model
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
def transliterate_lower(txt: str):
    translation = {
    # Main[48]:
    '\u00e0': 'a', '\u00e1': 'a',  '\u00e2': 'a',  '\u00e3': 'a',  '\u00e4': 'ae', '\u00e5': 'a', '\u0103': 'a',
    '\u00e7': 'c', '\u0107': 'c',  '\u010d': 'c',  '\u010f': 'd',  '\u00f0': 'dh', '\u00e9': 'e', '\u00e8': 'e',
    '\u00ea': 'e', '\u00eb': 'e',  '\u011f': 'g',  '\u00ec': 'i',  '\u00ed': 'i',  '\u00ee': 'i', '\u00ef': 'i',
    '\u0142': 'l', '\u0148': 'n',  '\u00f1': 'n',  '\u00f2': 'o',  '\u00f3': 'o',  '\u00f4': 'o', '\u00f5': 'o',
    '\u0159': 'r', '\u015b': 's',  '\u0219': 's',  '\u0165': 't',  '\u021b': 't',  '\u00fa': 'u', '\u00f9': 'u',
    '\u00fb': 'u', '\u016f': 'u',  '\u00fc': 'ue', '\u00fd': 'y',  '\u00ff': 'y',  '\u017a': 'z', '\u017c': 'z',
    '\u017e': 'z', '\u00fe': 'th', '\u00e6': 'ae', '\u00f6': 'oe', '\u00f8': 'oe', '\u00df': 'ss',

    # Extended[22]: 
    '\u0101': 'a', '\u0105': 'a', '\u0111': 'd', '\u0113': 'e', '\u0117': 'e', '\u0119': 'e',
    '\u012b': 'i', '\u012f': 'i', '\u0137': 'k', '\u013a': 'l', '\u013c': 'l', '\u013e': 'l',
    '\u0144': 'n', '\u0146': 'n', '\u014d': 'o', '\u0151': 'o', '\u0155': 'r', '\u0161': 's',
    '\u016b': 'u', '\u0171': 'u', '\u0173': 'u', '\u01b6': 'z',
    }
    return txt.translate(str.maketrans(translation))
################################################################################
def detokenize(index_array):
    return " ".join([transliterate_lower(index_to_word.get(i, " ")) for i in index_array])
################################################################################

txt_lines  = []
txt_lines.extend([detokenize(review) for review in x_train])
txt_lines.extend([detokenize(review) for review in x_test])
print("Train sequences: ", len(txt_lines))
################################################################################

random.shuffle(txt_lines)
text_ds = tf.data.Dataset.from_tensor_slices(txt_lines)
text_ds = text_ds.shuffle(buffer_size=256)
text_ds = text_ds.batch(batch_size)


def custom_standardization(input_string):
    lowercased = tf.strings.lower(input_string)
    stripped_html = tf.strings.regex_replace(lowercased, "<br />", " ")
    punctuation = " %$!?:,;\" @~&()*_<=>{|}[/]^\\"
    return tf.strings.regex_replace(stripped_html, f"([{punctuation}])", r" \1")
################################################################################


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
for inputs, targets in text_ds:
    # inputs, targets: shape(batch_size, maxlen)
    print(inputs.shape, targets.shape)
    break

################################################################################

"""
## Implement a Transformer block as a layer
"""
def causal_attention_mask(batch_size, n_dest, n_src, dtype):
    """
    Mask the upper half of the dot product matrix in self attention.
    This prevents flow of information from future tokens to current token.
    1's in the lower triangle, counting from the lower right corner.
    """
    i = tf.range(n_dest)[:, None]
    j = tf.range(n_src)
    m = i >= j - n_src + n_dest
    mask = tf.cast(m, dtype)
    mask = tf.reshape(mask, [1, n_dest, n_src])
    mult = tf.concat(
        [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)], 0
    )
    return tf.tile(mask, mult)


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads, embed_dim)
        self.ffn = keras.Sequential(
            [
                layers.Dense(ff_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size = input_shape[0]
        seq_len = input_shape[1]
        causal_mask = causal_attention_mask(batch_size, seq_len, seq_len, tf.bool)
        attention_output = self.att(inputs, inputs, attention_mask=causal_mask)
        attention_output = self.dropout1(attention_output)
        out1 = self.layernorm1(inputs + attention_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)


"""
## Implement an embedding layer

Create two separate embedding layers: one for tokens and one for token index
(positions).
"""
class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions


def create_model():
    inputs = layers.Input(shape=(maxlen,), dtype=tf.int32)
    embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(embed_dim, num_heads, feed_forward_dim)
    x = transformer_block(x)
    outputs = layers.Dense(vocab_size)(x)
    model = keras.Model(inputs=inputs, outputs=[outputs, x])
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(
        "adam",
        loss=[loss_fn, None],
    )  # No loss and optimization based on word embeddings from transformer block
    return model


"""
## Implement a Keras callback for generating text
"""
class TextGenerator(keras.callbacks.Callback):
    """A callback to generate text from a trained model.
    1. Feed some starting prompt to the model
    2. Predict probabilities for the next token
    3. Sample the next token and add it to the next input

    Arguments:
        max_tokens: Integer, the number of tokens to be generated after prompt.
        start_tokens: List of integers, the token indices for the starting prompt.
        index_to_word: List of strings, obtained from the TextVectorization layer.
        top_k: Integer, sample from the `top_k` token predictions.
        print_every: Integer, print after this many epochs.
    """

    def __init__(self, max_tokens, start_tokens, index_to_word, top_k=10, print_every=1):
        self.max_tokens = max_tokens
        self.start_tokens = start_tokens
        self.index_to_word = index_to_word
        self.print_every = print_every
        self.k = top_k

    def sample_from(self, logits):
        logits, indices = tf.math.top_k(logits, k=self.k, sorted=True)
        indices = np.asarray(indices).astype("int32")
        preds = keras.activations.softmax(tf.expand_dims(logits, 0))[0]
        preds = np.asarray(preds).astype("float32")
        return np.random.choice(indices, p=preds)

    def detokenize(self, number):
        return self.index_to_word[number]

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.print_every != 0:
            return

        return self.generate(self.start_tokens)

    def generate(self, input_tokens):

        start_tokens = [_ for _ in input_tokens]
        num_tokens_generated = 0
        tokens_generated = []

        while num_tokens_generated <= self.max_tokens:
            pad_len = maxlen - len(start_tokens)
            sample_index = len(start_tokens) - 1
            if pad_len < 0:
                x = start_tokens[:maxlen]
                sample_index = maxlen - 1
            elif pad_len > 0:
                x = start_tokens + [0] * pad_len
            else:
                x = start_tokens
            x = np.array([x])
            y, _ = self.model.predict(x)
            sample_token = self.sample_from(y[0][sample_index])
            tokens_generated.append(sample_token)
            start_tokens.append(sample_token)
            num_tokens_generated = len(tokens_generated)
        
            txt = " ".join(
                [self.detokenize(_) for _ in input_tokens + tokens_generated]
        )
        print(f"generated text:\n{txt}\n")

################################################################################
start_prompt = "this movie is"
num_tokens_generated = 40


str_tokens = start_prompt.split()

id_to_word = vectorize_layer.get_vocabulary()

# Tokenize starting prompt
word_to_id = {}
for index, word in enumerate(id_to_word):
    word_to_id[word] = index


start_tokens = [word_to_id.get(_, 1) for _ in str_tokens]
text_gen_callback = TextGenerator(num_tokens_generated, start_tokens, id_to_word)

############################################################
# convert string sentence to tokenized sentence

tokenized_sentences = vectorize_layer(start_prompt)
tokenized_sentences = tokenized_sentences[:len(str_tokens)].numpy()

#print(start_tokens, tokenized_sentences)
############################################################

model = create_model()

model.fit(text_ds, epochs=25, callbacks=[text_gen_callback])
