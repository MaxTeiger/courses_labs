# -*- coding: utf-8 -*-
# <nbformat>4</nbformat>

# <codecell>

import numpy as np
import pandas as pd
import operator

# <markdowncell>

# This is an introduction to basic sequence-to-sequence learning using a Long short term memory (LSTM) module.
# 
# Given a string of characters representing a math problem "3141+42" we would like to generate a string of characters representing the correct solution: "3183". Our network will learn how to do basic mathematical operations.
# 
# The important part is that we will not first use our human intelligence to break the string up into integers and a mathematical operator. We want the computer to figure all that out by itself.
# 
# Each math problem is an input sequence: a list of {0,...,9} integers and math operation symbols
# The result of the operation ("$3141+42$" $\rightarrow$ "$3183$"</span>) is the sequence to decode.

# <markdowncell>

# **math_operators** is the set of $5$ operations we are going to use to build are input sequences.<br/>
# The math_expressions_generation function uses them to generate a large set of examples

# <codecell>

def math_expressions_generation(n_samples=1000, n_digits=3, invert=True):
    X, Y = [], []
    math_operators = {
        '+': operator.add, 
        '-': operator.sub,
        '*': operator.mul,
        '/': operator.truediv,
        '%': operator.mod
    }
    for i in range(n_samples):
        a, b = np.random.randint(1, 10**n_digits, size=2)
        op = np.random.choice(list(math_operators.keys()))
        res = math_operators[op](a, b)
        x = "".join([str(elem) for elem in (a, op, b)])
        if invert is True:
            x = x[::-1]
        y = "{:.5f}".format(res) if isinstance(res, float) else str(res)
        X.append(x)
        Y.append(y)
    return X, Y

# <codecell>

X, y = math_expressions_generation(n_samples=int(1e5), n_digits=3, invert=True)
for X_i, y_i in list(zip(X, y))[:20]:
    print(X_i[::-1], '=', y_i)

# <markdowncell>

# # I - Standard sequence to sequence model

# <markdowncell>

# ## Seq2Seq architecture

# <markdowncell>

# # The Seq2Seq architecture
# 
# ### Two LSTM: an encoder and a decoder
# 
# <img src="../images/teacher_forcing_train.png" style="width: 600px;" />
# 
# - ## Training with teacher forcing
#     - Build the Seq2Seq model for training
#     - Example:
#         - Input sequence "94+8" must be given the the encoding LSTM
#         - True answers "1", "0", "2" are given to the LSTM encoder after the prediction is made to help it predict well the next token during training
#     - We advice to use Keras functional API here
# 
# - ### Encoder:
#     - Define a layer encoder_inputs of shape (None, self.encoder_vocabulary_size) 
#     - Instantiate the encoder LSTM layer before connecting it. Call it $\text{encoder_lstm}$
#         - Use the return_state param so it returns its last  state_h and state_c
#         - We need to pass those to the decoder LSTM afterwards to connect the $2$ LSTMs
#     - Connect $\text{encoder_lstm}$ to  encoder_inputs and get $\text{encoder_lstm}$'s last  state_h and state_c in a variable $\text{encoder_states = [state_h,  state_c]}$
#       
# - ### Decoder:
#     - Define a layer decoder_inputs of shape (None, self.decoder_vocabulary_size) 
#     - Instantiate the decoder LSTM layer before connecting it. Call it $\text{decoder_lstm}$.
#         - Pass encoder's last[state_h, state_c] to decoder initial_state argument to connect the two LSTM
#         - Use the return_sequences param so the decoder returns all the $h_{t}^{dec}$
#             - We need them to compute the predictions using the $h_{t}^{dec}$
#     - Connect $\text{decoder_lstm}$ to decoder_inputs and get the $h_{t}^{dec}$ hidden layers in a $\text{decoder_outputs}$ node
# 
# - ### Output:
#      - At this point we have all our $h_{t}^{dec}$ in a $\text{decoder_outputs}$ node, ready to be used to perform a token prediction for each timestep
#      - Define a Dense layer. Call it $\text{decoder_dense}$, with a softmax activation and self.decoder_vocabulary_size dimensionality
#      - Connect $\text{decoder_dense}$ to $\text{decoder_outputs}$ and make the result the $outputs$ node
#      - At this point each $h_{t}^{dec}$ has been mapped to a $( \text{num_decoder_tokens},1)$ vector of probability distribution over the next token. 
#      - Tensor at this point is shape $(batch, \text{max_decoder_seq_length}, \text{num_decoder_tokens})$

# <markdowncell>

# - ## Inference (testing time, no teacher forcing)
#    - We are going to see how to perform inference, that is decoding an input sequence without using teacher forcing
#    - We won't provide the answer from previous timesteps like during training.
#    - Thus we cannot provide with the <...EOS> part of the sequences like in training
#    - Predictions have to be performed one step at a time, first using the last state of encoder and GO token to produce the $1{st}$ decoder hidden layer  $h_{0}^{dec}$. Then $h_{0}^{dec}$ will be used to predict the $1{st}$ token. The $1{st}$ token predicted and $h_{0}^{dec}$ will be used to produce $h_{1}^{dec}$ and so on
# 
# - ## Requirements
#     - To perform inference we are going to need an $\text{encoder_model}$ that takes in the input sequence and returns the last $h$ and $c$ state to pass to the decoder
# 
#     - To perform inference we also are going to need a $\text{decoder model}$ that takes in the previous hidden state, a token like GO or previous prediction and returns next hidden state and prediction. We must iterate over those successive steps of taking in prev hidden state and token and output next hidden state and token to produce the whole decoded sentence without teacher forcing
# 
# To do this we are going to reuse layers and nodes used before to take advantage of the **trained weights**:
#    - encoder_inputs and encoder_states nodes that are already connected through a trained lstm to produce an encoder's last state
#    - decoder_lstm and decoder_inputs 
# 
# - ## encoder_model
#    - Use the class Model from keras.models
#    - Define the node $\text{encoder_inputs}$ as input to the model
#    - Define the node $\text{encoder_states}$ as output to the model
# 
# -  ## decoder_model:
#    - Define $2$ $Input$ keras.layers of dimensionality $\text{latent_dim}$: $\text{decoder_state_input_h}$ and $\text{decoder_state_input_c}$
#       - They are the parameters refering to the decoder_model's last state and to be received upon each call to function predict at each iteration
#           - Stack them in a  decoder_states_inputs variable: $\text{decoder_states_inputs} = [decoder\_state\_input\_h, decoder\_state\_input\_c]$
#           - connect decoder_lstm to decoder_inputs using the argument $\text{initial_state}$ = $\text{decoder_states_inputs}$ and get $\text{decoder_outputs}$, $\text{decoder_state_h}$, $\text{decoder_state_c}$ from that connection:
#           - this way we specify the fact that at each call to predict decoder_lstm will use the previous state received in argument to produce the next state
#           - decoder_outputs is all the $h_{t}^{dec}$ produced. There we give 1 token at a time, 1 prediction at a time thus decoder_output shape is $\text{(1,latent_dim)}$
#           - $\text{decoder_state_h}$  is the last  $h_{t}^{dec}$ and $\text{decoder_state_c}$, is the second part of decoder_lstm's last state
#           - Stack $\text{decoder_state_h}$ and $\text{decoder_state_c}$ in a $\text{decoder_states}$ variable:  $\text{decoder_states = [state_h, state_c]}$
#               - Those have to be returned after prediction
#           - Connect $\text{decoder_dense}$ layer from previous section to $\text{decoder_outputs}$ to have the distribution probability over the next token and call $output$ the node from that connection
#           - At this point we have our next prediction ($ouput$ node) and newest state ( [state_h, state_c]). We are ready to define the decoder_model:
#              - Make $\text{[decoder_inputs] + decoder_states_inputs}$ the inputs to the model (input args upon call to predict)
#              - Make $\text{[output] + decoder_states}$ the output to the model (output args upon call to predict)
#           
#           
#       

# <markdowncell>

# **GO** is the character ("=") that marks the beginning of decoding for the decoder LSTM<br/>
# **EOS** is the character ("\n") that marks the end of sequence to decode for the decoder LSTM

# <codecell>

from keras.models import Model
from keras.layers import Input, LSTM, Dense
from sklearn.model_selection import train_test_split

class Seq2seq():
    def __init__(self, X, y):
        # Special tokens
        self.GO = '='
        self.EOS = '\n'
        # Dataset properties
        self.X = None
        self.y = None
        self.X_tr = None
        self.X_val = None
        self.y_tr = None
        self.y_val = None
        self.n = None
        self.encoder_char_index = None
        self.encoder_char_index_inversed = None
        self.decoder_char_index = None
        self.decoder_char_index_inversed = None
        self.encoder_vocabulary_size = None
        self.decoder_vocabulary_size = None
        self.max_encoder_sequence_length = None
        self.max_decoder_sequence_length = None
        # Preprocessed data
        self.encoder_input_data_tr = None
        self.encoder_input_data_val = None
        self.decoder_input_data_tr = None
        self.decoder_input_data_val = None
        self.decoder_target_data_tr = None
        self.decoder_target_data_val = None
        # Model properties
        self.training_model = None
        self.inference_encoder_model = None
        self.inference_decoder_model = None
        self.batch_size = None
        self.epochs = None
        self.latent_dim = None
        # Model layers and states that we want to keep in memory between training and inference
        ## Encoder
        self.encoder_inputs = None
        self.encoder_states = None
        ## Decoder
        self.decoder_inputs = None
        self.decoder_lstm = None
        self.decoder_all_hdec = None
        self.decoder_dense = None
        # Dataset construction call
        self.load_and_preprocess_data(X, y)
        self.construct_dataset()
        
    def load_and_preprocess_data(self, X, y):
        self.X = list(X)
        self.y = list(map(lambda token: self.GO + token + self.EOS, y))
        self.n = len(self.X)
        encoder_characters = sorted(list(set("".join(self.X))))
        decoder_characters = sorted(list(set("".join(self.y))))
        self.encoder_char_index = dict((c, i) for i, c in enumerate(encoder_characters))
        self.encoder_char_index_inversed = dict((i, c) for i, c in enumerate(encoder_characters))
        self.decoder_char_index = dict((c, i) for i, c in enumerate(decoder_characters))
        self.decoder_char_index_inversed = dict((i, c) for i, c in enumerate(decoder_characters))
        self.encoder_vocabulary_size = len(self.encoder_char_index)
        self.decoder_vocabulary_size = len(self.decoder_char_index)
        self.max_encoder_sequence_length = max([len(sequence) for sequence in self.X])
        self.max_decoder_sequence_length = max([len(sequence) for sequence in self.y])
        print('Number of samples:', self.n)
        print('Number of unique encoder tokens:', self.encoder_vocabulary_size)
        print('Number of unique decoder tokens:', self.decoder_vocabulary_size)
        print('Max sequence length for encoding:', self.max_encoder_sequence_length)
        print('Max sequence length for decoding:', self.max_decoder_sequence_length)
        (self.X_tr, self.X_val, 
         self.y_tr, self.y_val) = train_test_split(
            self.X, 
            self.y,
            random_state=42
        )
        
    def construct_dataset(self):
        encoder_input_data = np.zeros(
            (self.n, self.max_encoder_sequence_length, self.encoder_vocabulary_size),
            dtype='float32')
        decoder_input_data = np.zeros(
            (self.n, self.decoder_vocabulary_size, self.decoder_vocabulary_size),
            dtype='float32')
        decoder_target_data = np.zeros(
            (self.n, self.decoder_vocabulary_size, self.decoder_vocabulary_size),
            dtype='float32')
        for i, (X_i, y_i) in enumerate(zip(self.X, self.y)):
            for t, char in enumerate(X_i):
                encoder_input_data[i, t, self.encoder_char_index[char]] = 1.
            for t, char in enumerate(y_i):
                decoder_input_data[i, t, self.decoder_char_index[char]] = 1.
                if t > 0:
                    decoder_target_data[i, t - 1, self.decoder_char_index[char]] = 1.
        (self.encoder_input_data_tr, self.encoder_input_data_val, 
         self.decoder_input_data_tr, self.decoder_input_data_val,
         self.decoder_target_data_tr, self.decoder_target_data_val) = train_test_split(
            encoder_input_data, 
            decoder_input_data, 
            decoder_target_data,
            random_state=42
        )
    
    """
    ENCODER LAYERS:
        - define a Input Keras object in self.encoder_inputs
        - apply a LSTM layer on self.encoder_inputs to get the last state_h and state_c
        - stack those states into an array self.encoder_states
    DECODER LAYERS:
        - define an Input Keras object in self.decoder_inputs
        - define a LSTM layer in self.decoder_lstm, make sure you set return_sequences=True
        to be able to return all hidden states
        - apply this LSTM layer on self.decoder_inputs with the states initialized with self.encoder_states
        and output all the hidden states in self.decoder_all_hdec
        - define a Dense layer in self.decoder_dense with a softmax activation, and output the results 
        in decoder_outputs using self.decoder_all_hdec as inputs
    MODEL DEFINITION:
        - now you can build your global Model:
        Model([self.encoder_inputs, self.decoder_inputs], decoder_outputs)
    """
    def design_and_compile_training_model(self, batch_size=64, latent_dim=256):
        # Hyperparameters
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        # Encoder layers
        # TODO:
        self.encoder_inputs = None
        self.encoder_states = None
        # Decoder layers
        # TODO:
        self.decoder_inputs = None
        self.decoder_lstm = None
        self.decoder_all_hdec = None
        self.decoder_dense = None
        decoder_outputs = None
        # Model definition and compilation
        if all(tensor is not None for tensor in [self.encoder_inputs, self.decoder_inputs, decoder_outputs]):
            self.training_model = Model([self.encoder_inputs, self.decoder_inputs], decoder_outputs)
            self.training_model.compile(optimizer='adam', loss='categorical_crossentropy')
            self.training_model.summary()
        else:
            print("Inputs and outputs of the model are not correctly defined!")
        
    def train(self, epochs=15):
        # Hyperparameters
        self.epochs = epochs
        # Model actual training
        if self.training_model is not None:
            self.training_model.fit(
                [self.encoder_input_data_tr, self.decoder_input_data_tr], self.decoder_target_data_tr,
                batch_size=self.batch_size,
                epochs=self.epochs,
                validation_data=(
                    [self.encoder_input_data_val, self.decoder_input_data_val], self.decoder_target_data_val
                )
            )
    
    """
    ENCODER MODEL:
        - create a Keras Model self.inference_encoder_model 
        with self.encoder_inputs as inputs and self.encoder_states as output
    DECODER MODEL:
        - define two Input Keras objects: one for the h_state and the other for the c_state, then stack
        them into a decoder_states_inputs array
        - reuse the already trained self.decoder_lstm layer with self.decoder_inputs as input
        and decoder_states_inputs as initial_state
            - you should get three outputs: decoder_all_hdec, decoder_state_h and decoder_state_c
        - again, stack the outputed decoder_state_h and decoder_state_c into a decoder_states array
        - now reuse the already trained self.decoder_dense layer with decoder_all_hdec as input,
        and store the output into decoder_outputs
        - you can finally create a Keras Model self.inference_decoder_model
        with [self.decoder_inputs] + decoder_states_inputs as inputs 
        and [decoder_outputs] + decoder_states as output
    """
    def design_inference_model(self):
        if self.training_model is None:
            print("No training model has been defined yet!")
            return None
        # Encoder model
        # TODO:
        self.inference_encoder_model = None
        # Decoder model
        ## Inputs: latent variables from the encoder
        # TODO:
        decoder_states_inputs = None
        ## Decoding using the LSTM trained layer from the decoder
        # TODO:
        decoder_states = None
        ## Get outputs using the Dense trained layer from the decoder
        # TODO:
        decoder_outputs = None
        ## Define the whole decoding model
        # TODO:
        self.inference_decoder_model = None
        
    def decode_sequence(self, input_sequence):
        if self.inference_encoder_model is None or self.inference_decoder_model is None:
            print("Inference models have not been designed yet!")
            return None
        states_value = self.inference_encoder_model.predict(input_sequence)
        target_sequence = np.zeros((1, 1, self.decoder_vocabulary_size))
        target_sequence[0, 0, self.decoder_char_index[self.GO]] = 1.
        decoded_sentence = ''
        while len(decoded_sentence) <= self.max_decoder_sequence_length:
            output_tokens, h, c = self.inference_decoder_model.predict(
                [target_sequence] + states_value
            )
            states_value = [h, c]
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = self.decoder_char_index_inversed[sampled_token_index]
            decoded_sentence += sampled_char
            if sampled_char == self.EOS:
                break
            target_sequence = np.zeros((1, 1, self.decoder_vocabulary_size))
            target_sequence[0, 0, sampled_token_index] = 1.
        return decoded_sentence

# <codecell>

seq2seq = Seq2seq(X, y)

# <codecell>

seq2seq.design_and_compile_training_model(latent_dim=64)

# <codecell>

seq2seq.train(epochs=1)

# <codecell>

seq2seq.design_inference_model()

# <codecell>

if seq2seq.inference_encoder_model is not None and seq2seq.inference_decoder_model is not None:
    for sequence_index in range(10):
        input_sequence = seq2seq.encoder_input_data_val[sequence_index: sequence_index + 1]
        decoded_sentence = seq2seq.decode_sequence(input_sequence)
        print('-')
        raw_input_sequence = "".join(
            [seq2seq.encoder_char_index_inversed[np.argmax(token)] for token in np.squeeze(input_sequence)][::-1]
        )
        print('Input sentence:', seq2seq.X_val[sequence_index][::-1])
        print('Decoded sentence:', decoded_sentence)

# <markdowncell>

# # II - Sequence to sequence model with attention mechanism

# <codecell>

from keras.layers import Activation, dot, concatenate, TimeDistributed

class Seq2seqAttention(Seq2seq):
    def __init__(self, X, y):
        # Attention layers hyperparameters
        self.latent_attention_dim = 64
        # All hidden states from the encoder that must now be stored
        self.encoder_outputs = None
        # Attention layers and states
        self.dense_tanh = None
        self.dense_final = None
        # Seq2Seq class initialization
        super(Seq2seqAttention, self).__init__(X, y)
    
    """
    ENCODER LAYERS:
        - define a Input Keras object in self.encoder_inputs       
        - apply a LSTM layer with return_sequences=True on self.encoder_inputs 
        to get (self.encoder_outputs, state_h, state_c)
        - stack state_h and state_c into an array self.encoder_states
    DECODER LAYERS:
        - define an Input Keras object in self.decoder_inputs
        - define a LSTM layer in self.decoder_lstm, make sure you set return_sequences=True
        to be able to return all hidden states
        - apply this LSTM layer on self.decoder_inputs with the states initialized with self.encoder_states
        and output all the hidden states in self.decoder_all_hdec
    ATTENTION LAYERS:
        - apply a dot product between self.decoder_all_hdec and self.encoder_outputs along their last
        dimension (the latent one), then a softmax activation, and store the result into attention
        - compute the context tensor with a dot product between attention and self.encoder_outputs
        - concatenate the result with self.decoder_all_hdec
        - define the two final Dense layers: 
            - the first with tanh activation and self.latent_attention_dim size
            - the second with softmax activation and self.decoder_vocabulary_size
        - output the final result into attention_outputs
    MODEL DEFINITION:
        - now you can build your global Model:
        Model([self.encoder_inputs, self.decoder_inputs], attention_outputs)
    """
    def design_and_compile_training_model(self, batch_size=64, latent_dim=256, latent_attention_dim=64):
        # Hyperparameters
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.latent_attention_dim = latent_attention_dim
        # Encoder layers
        # TODO:
        self.encoder_inputs = None
        self.encoder_states = None
        self.encoder_outputs = None
        # Decoder layers
        # TODO:
        self.decoder_inputs = None
        self.decoder_lstm = None
        self.decoder_all_hdec = None
        # Attention layers
        # TODO:
        self.dense_tanh = None
        self.dense_final = None
        attention_outputs = None
        # Model definition and compilation
        if all(tensor is not None for tensor in [self.encoder_inputs, self.decoder_inputs, attention_outputs]):
            self.training_model = Model([self.encoder_inputs, self.decoder_inputs], attention_outputs)
            self.training_model.compile(optimizer='adam', loss='categorical_crossentropy')
            self.training_model.summary()
        else:
            print("Inputs and outputs of the model are not correctly defined!")
        
    """
    ENCODER MODEL:
        - create a Keras Model self.inference_encoder_model 
        with self.encoder_inputs as inputs and self.encoder_states as output
    DECODER MODEL:
        - define two Input Keras objects: one for the h_state and the other for the c_state, then stack
        them into a decoder_states_inputs array
        - reuse the already trained self.decoder_lstm layer with self.decoder_inputs as input
        and decoder_states_inputs as initial_state
            - you should get three outputs: decoder_all_hdec, decoder_state_h and decoder_state_c
        - again, stack the outputed decoder_state_h and decoder_state_c into a decoder_states array        
        - now apply a dot product between decoder_all_hdec and self.encoder_outputs along their last
        dimension (the latent one), then a softmax activation: it is your attention tensor
        - compute the context tensor with a dot product between the attention tensor and self.encoder_outputs
        - concatenate the result with decoder_all_hdec
        - reuse the two trained Denser layers and output the final result into attention_outputs
        - you can finally create a Keras Model self.inference_decoder_model
        with [self.decoder_inputs] + [self.decoder_inputs] + decoder_states_inputs as inputs 
        and [attention_outputs] + decoder_states as output
    """
    def design_inference_model(self):
        if self.training_model is None:
            print("No training model has been defined yet!")
            return None
        # Encoder model
        # TODO:
        self.inference_encoder_model = None
        # Decoder model
        ## Inputs: latent variables from the encoder
        # TODO:
        decoder_states_inputs = None
        ## Decoding using the LSTM trained layer from the decoder
        # TODO:
        decoder_states = None
        ## Get outputs using multiple dot products and softmax activation followed by the two trained Dense layers
        # TODO:
        attention_outputs = None
        ## Define the whole decoding model
        # TODO:
        self.inference_decoder_model = None
        
    def decode_sequence(self, input_sequence):
        if self.inference_encoder_model is None or self.inference_decoder_model is None:
            print("Inference models have not been designed yet!")
            return None
        states_value = self.inference_encoder_model.predict(input_sequence)
        target_sequence = np.zeros((1, 1, self.decoder_vocabulary_size))
        target_sequence[0, 0, self.decoder_char_index[self.GO]] = 1.
        decoded_sentence = ''
        while len(decoded_sentence) <= self.max_decoder_sequence_length:
            output_tokens, h, c = self.inference_decoder_model.predict(
                [input_sequence] + [target_sequence] + states_value
            )
            states_value = [h, c]
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = self.decoder_char_index_inversed[sampled_token_index]
            decoded_sentence += sampled_char
            if sampled_char == self.EOS:
                break
            target_sequence = np.zeros((1, 1, self.decoder_vocabulary_size))
            target_sequence[0, 0, sampled_token_index] = 1.
        return decoded_sentence

# <codecell>

seq2seq_attention = Seq2seqAttention(X, y)

# <codecell>

seq2seq_attention.design_and_compile_training_model(latent_dim=64)

# <codecell>

seq2seq_attention.train(epochs=1)

# <codecell>

seq2seq_attention.design_inference_model()

# <codecell>

if seq2seq_attention.inference_encoder_model is not None and seq2seq_attention.inference_decoder_model is not None:
    for sequence_index in range(10):
        input_sequence = seq2seq_attention.encoder_input_data_val[sequence_index: sequence_index + 1]
        decoded_sentence = seq2seq_attention.decode_sequence(input_sequence)
        print('-')
        raw_input_sequence = "".join(
            [seq2seq_attention.encoder_char_index_inversed[np.argmax(token)] 
             for token in np.squeeze(input_sequence)][::-1]
        )
        print('Input sentence:', seq2seq_attention.X_val[sequence_index][::-1])
        print('Decoded sentence:', decoded_sentence)
