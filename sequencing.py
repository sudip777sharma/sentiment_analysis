# sequencing --> turning sentence in to data
from msilib import sequence
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences 

sentences = ['i love my dog', 'i love my cat', 'do you love my !dog', 'do you think my dog is amazing']

tokenizer = Tokenizer(num_words=100)
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(sentences)

pre_padded = pad_sequences(sequences)
post_padded = pad_sequences(sequences, padding = 'post', truncating = 'pre', maxlen = 5)

print(word_index)
print(sequences)
print(pre_padded)
print(post_padded)