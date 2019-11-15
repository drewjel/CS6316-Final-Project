'''
Created on Nov 13, 2019

@author: andrewelsey
'''
import tensorflow as tf

from typing import List

class OrderedNeuronModel(object):
    def __init__(self, C, nlayers, hidden_size):
        
    
    
    def loss(self):
        pass
    
    def predict(self):
        pass

class OrderedNeuronCell(object):
    def __init__(self, C, nlayers, vocab_size, input_size, hidden_size):
        self.seq_len = tf.placeholder(dtype=tf.int, [1])
        
        self.input = tf.placeholder(dtype=tf.float32, [1, None, input_size])
        
        self.hidden = tf.zeros(shape=[1, 1, hidden_size])
        
        self.layers = [ordered_neuron_layer(hidden_size) for i in range(nlayers - 1)]
        self.layers += [ordered_neuron_layer(input_size)]
        
        self.embedding = tf.keras.layers.Embedding(vocab_size, input_size)
        
        
    def forward_propagate(self, seq_len):
        
        self.input_embedding = self.embedding(self.input)
        
        self.last_cell_state = tf.zeros([1, hidden_size], dtype=tf.float32)
        self.last_hidden_state = tf.zeros([1, hidden_size], dtype=tf.float32)
        self.cell_states = []
        self.hidden_states = []
        for layer in self.layers:
            for i in range(seq_len):
                self.input_i = tf.gather(self.input_embedding, i, axis=1)
                
                self.last_cell_state, self.last_hidden_state = layer.predict(current_cell_state, current_hidden_state)

                self.last_cell_state.ap

class OrderedNeuronLayer(object):
    def __init__(self, C, input_size, hidden_size):
        self.C = C
        self.hidden_size = hidden_size
        
        self.input_weights = tf.keras.layers.Dense(C + hidden_size * 4 + C, 
                                                   bias=True, 
                                                   kernel_initializer='glorot_normal',
                                                   bias_initializer='glorot_normal')
    
        self.hidden_weights = tf.keras.layers.Dense(C + hidden_size * 4 + C, 
                                                   bias=True, 
                                                   kernel_initializer='glorot_normal',
                                                   bias_initializer='glorot_normal')
        
        self.master_input_gate_weights_input = tf.keras.layers.Dense(C, 
                                                   bias=True, 
                                                   kernel_initializer='glorot_normal',
                                                   bias_initializer='glorot_normal')
        self.master_input_gate_weights_hidden = tf.keras.layers.Dense(C, 
                                                   bias=True, 
                                                   kernel_initializer='glorot_normal',
                                                   bias_initializer='glorot_normal')
        
        self.master_forget_gate_weights_input = tf.keras.layers.Dense(C, 
                                                   bias=True, 
                                                   kernel_initializer='glorot_normal',
                                                   bias_initializer='glorot_normal')
        
        self.master_forget_gate_weights_hidden = tf.keras.layers.Dense(C, 
                                                   bias=True, 
                                                   kernel_initializer='glorot_normal',
                                                   bias_initializer='glorot_normal')
        
        self.input_gate_weights_input = tf.keras.layers.Dense(hidden_size, 
                                                   bias=True, 
                                                   kernel_initializer='glorot_normal',
                                                   bias_initializer='glorot_normal')
        self.input_gate_weights_hidden = tf.keras.layers.Dense(hidden_size, 
                                                   bias=True, 
                                                   kernel_initializer='glorot_normal',
                                                   bias_initializer='glorot_normal')
        
        self.forget_gate_weights_input = tf.keras.layers.Dense(hidden_size, 
                                                   bias=True, 
                                                   kernel_initializer='glorot_normal',
                                                   bias_initializer='glorot_normal')
        self.forget_gate_weights_hidden = tf.keras.layers.Dense(hidden_size, 
                                                   bias=True, 
                                                   kernel_initializer='glorot_normal',
                                                   bias_initializer='glorot_normal')
        
        self.cell_state_input = tf.keras.layers.Dense(hidden_size, 
                                                   bias=True, 
                                                   kernel_initializer='glorot_normal',
                                                   bias_initializer='glorot_normal')
        self.cell_state_hidden = tf.keras.layers.Dense(hidden_size, 
                                                   bias=True, 
                                                   kernel_initializer='glorot_normal',
                                                   bias_initializer='glorot_normal')
        
        self.output_gate_input = tf.keras.layers.Dense(hidden_size, 
                                                   bias=True, 
                                                   kernel_initializer='glorot_normal',
                                                   bias_initializer='glorot_normal')
        self.output_gate_hidden = tf.keras.layers.Dense(hidden_size, 
                                                   bias=True, 
                                                   kernel_initializer='glorot_normal',
                                                   bias_initializer='glorot_normal')
        
        
    def predict(self, input, cell_state, hidden):
        self.master_forget_gate_in = self.master_forget_gate_weights_input(input) +\
                                             self.master_forget_gate_weights_hidden(hidden)
        self.master_input_gate_in = self.master_input_gate_weights_input(input) +\
                                            self.master_input_gate_weights_hidden(hidden)
        self.master_forget = self.cumax(master_forget_gate_in)
        self.master_input = 1 - self.cumax(master_input_gate_in)
                                            
        self.cell_state_pre = tf.sigmoid(self.cell_state_input(input) + self.cell_state_hidden(hidden))
        self.input_gate_pre = tf.sigmoid(self.input_gate_weights_input(input) + self.input_gate_weights_hidden(hidden))
        self.output_gate = tf.tanh(self.output_gate_input(input) + self.output_gate_hidden(hidden))
        self.forget_gate_pre = tf.sigmoid(self.forget_gate_weights_input(input) + self.forget_gate_weights_hidden(hidden))
        
        self.w = self.master_forget * self.master_input
        
        self.forget_gate = self.forget_gate_pre * self.w + self.master_forget - self.w
        self.input_gate = self.input_gate_pre * self.w + self.master_input - self.w
        self.cell_state = self.forget_gate * cell_state + self.input_gate * self.cell_state_pre
        self.hidden_state = self.output_gate * tf.tanh(self.cell_state)
        
        return self.cell_state, self.hidden_state
        
        #^ft = ft ◦ ωt + ( ˜ft − ωt) = ˜ft ◦ (ft ◦ ˜it + 1 −˜it) (12)
        #ˆit = it ◦ ωt + (˜it − ωt) = ˜it ◦ (it ◦˜ft + 1 − ˜ft) (13)
        #ct = ˆft ◦ ct−1 +ˆit ◦ cˆt
        
    
    def cumax(self, input):
        return tf.cumsum(tf.nn.softmax(input, dim=-1), axis=-1)    

'''
class ONLSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size, chunk_size, dropconnect=0.):
        super(ONLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.chunk_size = chunk_size
        self.n_chunk = int(hidden_size / chunk_size)

        self.ih = nn.Sequential(
            nn.Linear(input_size, 4 * hidden_size + self.n_chunk * 2, bias=True),
            # LayerNorm(3 * hidden_size)
        )
        self.hh = LinearDropConnect(hidden_size, hidden_size*4+self.n_chunk*2, bias=True, dropout=dropconnect)

        # self.c_norm = LayerNorm(hidden_size)

        self.drop_weight_modules = [self.hh]

    def forward(self, input, hidden,
                transformed_input=None):
        hx, cx = hidden

        if transformed_input is None:
            transformed_input = self.ih(input)

        gates = transformed_input + self.hh(hx)
        cingate, cforgetgate = gates[:, :self.n_chunk*2].chunk(2, 1)
        outgate, cell, ingate, forgetgate = gates[:,self.n_chunk*2:]\
            .view(-1, self.n_chunk*4, self.chunk_size).chunk(4,1)

        cingate = 1. - cumsoftmax(cingate)
        cforgetgate = cumsoftmax(cforgetgate)

        distance_cforget = 1. - cforgetgate.sum(dim=-1) / self.n_chunk
        distance_cin = cingate.sum(dim=-1) / self.n_chunk

        cingate = cingate[:, :, None]
        cforgetgate = cforgetgate[:, :, None]

        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)
        cell = F.tanh(cell)
        outgate = F.sigmoid(outgate)

        # cy = cforgetgate * forgetgate * cx + cingate * ingate * cell

        overlap = cforgetgate * cingate
        forgetgate = forgetgate * overlap + (cforgetgate - overlap)
        ingate = ingate * overlap + (cingate - overlap)
        cy = forgetgate * cx + ingate * cell

        # hy = outgate * F.tanh(self.c_norm(cy))
        hy = outgate * F.tanh(cy)
        return hy.view(-1, self.hidden_size), cy, (distance_cforget, distance_cin)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (weight.new(bsz, self.hidden_size).zero_(),
                weight.new(bsz, self.n_chunk, self.chunk_size).zero_())

    def sample_masks(self):
        for m in self.drop_weight_modules:
            m.sample_mask()
'''

       