'''
Created on Nov 13, 2019

@author: andrewelsey
'''
import tensorflow as tf
from collections import defaultdict
import numpy as np
from typing import List

class OrderedNeuronModel(object):
    def __init__(self, C, nlayers, vocab_size, input_size, hidden_size, lr, batch_size=None):
        self.cell = OrderedNeuronCell(C, nlayers, vocab_size, input_size, hidden_size, batch_size)

        
        splits = []
        if vocab_size > 500000:
            # One Billion
            # This produces fairly even matrix mults for the buckets:
            # 0: 11723136, 1: 10854630, 2: 11270961, 3: 11219422
            splits = [4200, 35000, 180000] 
        elif vocab_size > 75000:
            # WikiText-103
            splits = [2800, 20000, 76000]
        
        self.loss_criterion = SplitCrossEntropyLoss(hidden_size, splits)

        self.targets = tf.placeholder(dtype=tf.int32, shape=(None, 1))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=lr )#,beta1=0)
    
        hidden_states, cell_states = self.cell.forward_propagate()
        hidden_states = tf.concat([tf.reshape(hs, (batch_size, 1, -1)) for hs in hidden_states], axis=1)

        self.decoded = self.cell.decoder(hidden_states)#tf.keras.backend.dot(hidden_states, self.cell.decoder_kernel) + self.cell.decoder_bias

        self.decoded = tf.reshape(self.decoded, (6400, -1))

        #self.logits = tf.nn.softmax(self.decoded, axis=-1)

        self._targets = tf.reshape(self.targets, (6400, 1))
        self._targets = tf.one_hot(self.targets, depth=vocab_size)

        self.loss = tf.nn.softmax_cross_entropy_with_logits(labels=self._targets, logits=self.decoded)#self.loss_criterion.calc_loss(self.cell.decoder_kernel, self.cell.decoder_bias, hidden_states, self.targets)
        self.loss = tf.reduce_mean(self.loss)
        self.step = self.optimizer.minimize(self.loss)
    
    #def loss(self):
        #copy over code from SplitCrossEntropy
    #    hidden_states, cell_states = self.cell.forward_propagate()
    #    
    #    return self.loss_criterion.calc_loss(self.cell.decoder_kernel, self.cell.decoder_bias, hidden_states, self.targets)
    
    def predict(self):
        return self.cell.forward_propagate()

class OrderedNeuronCell(object):
    def __init__(self, C, nlayers, vocab_size, input_size, hidden_size, batch_size):
        self.hidden_size = hidden_size
        self.seq_len = tf.placeholder(dtype=tf.int32, shape=None)
        
        self.input = tf.placeholder(dtype=tf.float32, shape=[batch_size, 80])
        
        self.hidden = tf.zeros(shape=[1, 1, self.hidden_size])
        
        self.layers = [OrderedNeuronLayer(C, input_size, hidden_size)]
        self.layers += [OrderedNeuronLayer(C, hidden_size,hidden_size) for i in range(nlayers - 2)]
        self.layers += [OrderedNeuronLayer(C, hidden_size, input_size)]
        
        self.embedding = tf.keras.layers.Embedding(vocab_size, input_size)
        
        #self.decoder = tf.keras.layers.Dense(input_size, vocab_size)
        
        #self.decoder_kernel = tf.get_variable(name='decoder_kernel', shape=[input_size, vocab_size], dtype=tf.float32, initializer=tf.glorot_normal_initializer(), trainable=True)
        #self.decoder_bias = tf.get_variable(name='decoder_bias', shape=[vocab_size], dtype=tf.float32, initializer=tf.glorot_normal_initializer(), trainable=True)
        self.decoder = tf.keras.layers.Dense(units = vocab_size, kernel_initializer=tf.glorot_normal_initializer(), bias_initializer=tf.glorot_normal_initializer())
        
    def forward_propagate(self):
        
        self.input_embedding = self.embedding(self.input)
        
        self.cell_states = []
        self.hidden_states = []
        for layer in self.layers:
            self.first_cell_state = tf.stop_gradient(tf.zeros([1, layer.hidden_size], dtype=tf.float32))
            self.first_hidden_state = tf.stop_gradient(tf.zeros([1, layer.hidden_size], dtype=tf.float32))

            self.last_cell_state = self.first_cell_state
            self.last_hidden_state = self.first_hidden_state


            for i in range(80):
                self.input_i = tf.gather(self.input_embedding, i, axis=1) \
                                if len(self.hidden_states) <= i else self.hidden_states[i]
                
                self.last_cell_state, self.last_hidden_state = layer.predict(self.input_i, self.last_cell_state, self.last_hidden_state)

                if len(self.cell_states) <= i:
                    self.cell_states.append(self.last_cell_state)
                    self.hidden_states.append(self.last_hidden_state)
                else:
                    self.cell_states[i] = self.last_cell_state
                    self.hidden_states[i] = self.last_hidden_state
        
        return self.hidden_states, self.cell_states

class OrderedNeuronLayer(object):
    def __init__(self, C, input_size, hidden_size):
        self.C = C
        self.input_ize = input_size
        self.hidden_size = hidden_size
        
        self.input_weights = tf.keras.layers.Dense(C + hidden_size * 4 + C, 
                                                   use_bias=True, 
                                                   kernel_initializer='glorot_normal',
                                                   bias_initializer='glorot_normal')
    
        self.hidden_weights = tf.keras.layers.Dense(C + hidden_size * 4 + C, 
                                                   use_bias=True, 
                                                   kernel_initializer='glorot_normal',
                                                   bias_initializer='glorot_normal')
        
        self.master_input_gate_weights_input = tf.keras.layers.Dense(C, 
                                                   use_bias=True, 
                                                   kernel_initializer='glorot_normal',
                                                   bias_initializer='glorot_normal')
        self.master_input_gate_weights_hidden = tf.keras.layers.Dense(C, 
                                                   use_bias=True, 
                                                   kernel_initializer='glorot_normal',
                                                   bias_initializer='glorot_normal')
        
        self.master_forget_gate_weights_input = tf.keras.layers.Dense(C, 
                                                   use_bias=True, 
                                                   kernel_initializer='glorot_normal',
                                                   bias_initializer='glorot_normal')
        
        self.master_forget_gate_weights_hidden = tf.keras.layers.Dense(C, 
                                                   use_bias=True, 
                                                   kernel_initializer='glorot_normal',
                                                   bias_initializer='glorot_normal')
        
        self.input_gate_weights_input = tf.keras.layers.Dense(hidden_size, 
                                                   use_bias=True, 
                                                   kernel_initializer='glorot_normal',
                                                   bias_initializer='glorot_normal')
        self.input_gate_weights_hidden = tf.keras.layers.Dense(hidden_size, 
                                                   use_bias=True, 
                                                   kernel_initializer='glorot_normal',
                                                   bias_initializer='glorot_normal')
        
        self.forget_gate_weights_input = tf.keras.layers.Dense(hidden_size, 
                                                   use_bias=True, 
                                                   kernel_initializer='glorot_normal',
                                                   bias_initializer='glorot_normal')
        self.forget_gate_weights_hidden = tf.keras.layers.Dense(hidden_size, 
                                                   use_bias=True, 
                                                   kernel_initializer='glorot_normal',
                                                   bias_initializer='glorot_normal')
        
        self.cell_state_input = tf.keras.layers.Dense(hidden_size, 
                                                   use_bias=True, 
                                                   kernel_initializer='glorot_normal',
                                                   bias_initializer='glorot_normal')
        self.cell_state_hidden = tf.keras.layers.Dense(hidden_size, 
                                                   use_bias=True, 
                                                   kernel_initializer='glorot_normal',
                                                   bias_initializer='glorot_normal')
        
        self.output_gate_input = tf.keras.layers.Dense(hidden_size, 
                                                   use_bias=True, 
                                                   kernel_initializer='glorot_normal',
                                                   bias_initializer='glorot_normal')
        self.output_gate_hidden = tf.keras.layers.Dense(hidden_size, 
                                                   use_bias=True, 
                                                   kernel_initializer='glorot_normal',
                                                   bias_initializer='glorot_normal')
        
        
    def predict(self, input, cell_state, hidden):
        self.master_forget_gate_in = self.master_forget_gate_weights_input(input) +\
                                             self.master_forget_gate_weights_hidden(hidden)
        self.master_input_gate_in = self.master_input_gate_weights_input(input) +\
                                            self.master_input_gate_weights_hidden(hidden)
        self.master_forget = self.cumax(self.master_forget_gate_in)
        self.master_input = 1 - self.cumax(self.master_input_gate_in)
                                            
        self.cell_state_pre = tf.sigmoid(self.cell_state_input(input) + self.cell_state_hidden(hidden))
        self.input_gate_pre = tf.sigmoid(self.input_gate_weights_input(input) + self.input_gate_weights_hidden(hidden))
        self.output_gate = tf.tanh(self.output_gate_input(input) + self.output_gate_hidden(hidden))
        self.forget_gate_pre = tf.sigmoid(self.forget_gate_weights_input(input) + self.forget_gate_weights_hidden(hidden))
        
        p = self.forget_gate_weights_input(input)
        c = self.forget_gate_weights_hidden(hidden)
        self.w = self.master_forget * self.master_input
        
        self.forget_gate = self.forget_gate_pre * tf.tile(self.w, [1, int(self.hidden_size/self.C)]) + tf.tile(self.master_forget - self.w, [1, int(self.hidden_size/self.C)])
        self.input_gate = self.input_gate_pre * tf.tile(self.w, [1, int(self.hidden_size/self.C)]) + tf.tile(self.master_input - self.w, [1, int(self.hidden_size/self.C)])
        self.cell_state = self.forget_gate * cell_state + self.input_gate * self.cell_state_pre
        self.hidden_state = self.output_gate * tf.tanh(self.cell_state)
        
        return self.cell_state, self.hidden_state
        
        #^ft = ft ◦ ωt + ( ˜ft − ωt) = ˜ft ◦ (ft ◦ ˜it + 1 −˜it) (12)
        #ˆit = it ◦ ωt + (˜it − ωt) = ˜it ◦ (it ◦˜ft + 1 − ˜ft) (13)
        #ct = ˆft ◦ ct−1 +ˆit ◦ cˆt
        
    
    def cumax(self, input):
        return tf.cumsum(tf.nn.softmax(input, dim=-1), axis=-1)   
    
    def loss(self):
        #write split_entropy function 
        pass


class SplitCrossEntropyLoss():
    
    def __init__(self, hidden_size, splits):
        self.hidden_size = hidden_size
        self.splits = [0] + splits + [100 * 1000000]
        self.nsplits = len(self.splits) - 1
        self.stats = defaultdict(list)
        # Each of the splits that aren't in the head require a pretend token, we'll call them tombstones
        # The probability given to this tombstone is the probability of selecting an item from the represented split
        if self.nsplits > 1:
            self.tail_vectors = tf.zeros([self.nsplits-1, hidden_size])
            self.tail_bias = tf.zeros([self.nsplits-1])
            
    def logprob(self, weight, bias, hiddens, splits=None, softmaxed_head_res=None, verbose=False):
        # First we perform the first softmax on the head vocabulary and the tombstones
        if softmaxed_head_res is None:
            start, end = self.splits[0], self.splits[1]
            head_weight = None if end - start == 0 else weight[start:end]
            head_bias = None if end - start == 0 else bias[start:end]
            # We only add the tombstones if we have more than one split
            if self.nsplits > 1:
                head_weight = self.tail_vectors if head_weight is None else torch.cat([head_weight, self.tail_vectors])
                head_bias = self.tail_bias if head_bias is None else torch.cat([head_bias, self.tail_bias])

            # Perform the softmax calculation for the word vectors in the head for all splits
            # We need to guard against empty splits as torch.cat does not like random lists
            #head_res = torch.nn.functional.linear(hiddens, head_weight, bias=head_bias)
            #softmaxed_head_res = torch.nn.functional.log_softmax(head_res, dim=-1)

            head_res = hiddens * head_weight + head_bias
        
            softmaxed_head_res = tf.nn.log_softmax(head_res, -1)


        if splits is None:
            splits = list(range(self.nsplits))

        results = []
        running_offset = 0
        for idx in splits:

            # For those targets in the head (idx == 0) we only need to return their loss
            if idx == 0:
                results.append(softmaxed_head_res[:, :-(self.nsplits - 1)])
                #results.append(tf.gather(softmaxed_head_res, [:-(self.nsplits - 1)], axis=1))

            # If the target is in one of the splits, the probability is the p(tombstone) * p(word within tombstone)
            else:
                start, end = self.splits[idx], self.splits[idx + 1]
                tail_weight =  None if end - start == 0 else weight[start:end]
                tail_bias =  None if end - start == 0 else bias[start:end]
                
                #tail_weight = None if end - start == 0 else tf.gather(weight, [start:end])
                #tail_bias = None if end - start == 0 else tf.gather(bias, [start:end])


                # Calculate the softmax for the words in the tombstone
                #tail_res = torch.nn.functional.linear(hiddens, tail_weight, bias=tail_bias)
                tail_res = hiddens * tail_weight + tail_bias


                # Then we calculate p(tombstone) * p(word in tombstone)
                # Adding is equivalent to multiplication in log space
                #head_entropy = (softmaxed_head_res[:, -idx]).contiguous()
                #tail_entropy = torch.nn.functional.log_softmax(tail_res, dim=-1)
                #results.append(head_entropy.view(-1, 1) + tail_entropy)
                
                head_entropy = tf.gather(softmaxed_head_res, [-idx], axis=1)
                tail_entropy = tf.nn.log_softmax(tail_res, axis=-1)
                results.append(tf.reshape(head_entropy, [-1,1]) + tail_entropy)

        if len(results) > 1:
            #return torch.cat(results, dim=1)
            return tf.concat(results, axis=1)
        return results[0]
    
    def split_on_targets(self, hiddens, targets):
        # Split the targets into those in the head and in the tail
        split_targets = []
        split_hiddens = []

        # Determine to which split each element belongs (for each start split value, add 1 if equal or greater)
        # This method appears slower at least for WT-103 values for approx softmax
        #masks = [(targets >= self.splits[idx]).view(1, -1) for idx in range(1, self.nsplits)]
        #mask = torch.sum(torch.cat(masks, dim=0), dim=0)
        ###
        # This is equally fast for smaller splits as method below but scales linearly
        mask = None
        zeros = tf.zeros(tf.shape(targets))
        ones = tf.ones(tf.shape(targets))
        
        for idx in range(1, self.nsplits):
            partial_mask = tf.where(targets >= self.splits[idx], ones, zeros)
            mask = mask + partial_mask if mask is not None else partial_mask
        ###
        #masks = torch.stack([targets] * (self.nsplits - 1))
        #mask = torch.sum(masks >= self.split_starts, dim=0)
        for idx in range(self.nsplits):
            # If there are no splits, avoid costly masked select
            if self.nsplits == 1:
                split_targets, split_hiddens = [targets], [hiddens]
                continue
            # If all the words are covered by earlier targets, we have empties so later stages don't freak out
            if sum(len(t) for t in split_targets) == len(targets):
                split_targets.append([])
                split_hiddens.append([])
                continue
            # Are you in our split?
            tmp_mask = tf.equal(mask, idx)
            split_targets.append(tf.boolean_mask(targets, tmp_mask))
            #hiddens.masked_select(tmp_mask.unsqueeze(1).expand_as(hiddens)).view(-1, hiddens.size(1))
            split_hiddens.append(tf.boolean_mask(hiddens, tmp_mask))
        return split_targets, split_hiddens
    
    #criterion(model.decoder.weight, model.decoder.bias, output, targets).data
    def calc_loss(self, weight, bias, hiddens, targets):
        
        total_loss = None
        #if len(hiddens.size()) > 2: hiddens = hiddens.view(-1, hiddens.size(2))

        hiddens2 = tf.concat(hiddens, axis=0)
        hiddens2 = tf.reshape(hiddens, (-1, tf.shape(hiddens)[-1]))

        split_targets, split_hiddens = self.split_on_targets(hiddens, targets)

        # First we perform the first softmax on the head vocabulary and the tombstones
        start, end = self.splits[0], self.splits[1]
        head_weight = None if end - start == 0 else weight[start:end]
        head_bias = None if end - start == 0 else bias[start:end]
        
        #head_weight = None if end - start == 0 else tf.gather(weight, [start:end])
        #head_bias = None if end - start == 0 else tf.gather(bias, [start:end])

        # We only add the tombstones if we have more than one split
        if self.nsplits > 1:
            #head_weight = self.tail_vectors if head_weight is None else torch.cat([head_weight, self.tail_vectors])
            #head_bias = self.tail_bias if head_bias is None else torch.cat([head_bias, self.tail_bias])
            head_weight = self.tail_vectors if head_weight is None else tf.concat([head_weight, self.tail_vectors], axis=0)
            head_bias = self.tail_bias if head_bias is None else tf.concat([head_bias, self.tail_bias], axis=0)


        # Perform the softmax calculation for the word vectors in the head for all splits
        # We need to guard against empty splits as torch.cat does not like random lists
        #combo = torch.cat([split_hiddens[i] for i in range(self.nsplits) if len(split_hiddens[i])])
        #combo = tf.concat([split_hiddens[i] for i in range(self.nsplits) if len(split_hiddens[i])], axis=0)
        combo = tf.concat(hiddens, axis=0)

        ###
        #all_head_res = torch.nn.functional.linear(combo, head_weight, bias=head_bias)
        
        all_head_res = tf.keras.backend.dot(combo, head_weight) + head_bias
        #softmaxed_all_head_res = torch.nn.functional.log_softmax(all_head_res, dim=-1)
        
        softmaxed_all_head_res = tf.nn.log_softmax(all_head_res, -1)

        running_offset = 0
        for idx in range(self.nsplits):
            # If there are no targets for this split, continue
            if split_targets[idx] is None: continue#len(split_targets[idx]) == 0: continue

            # For those targets in the head (idx == 0) we only need to return their loss
            if idx == 0:
                softmaxed_head_res = softmaxed_all_head_res[running_offset:running_offset + len(split_hiddens[idx])]
                #softmaxed_head_res = tf.gather(softmaxed_all_head_res, [running_offset:running_offset + len(split_hiddens[idx])])
                
                #entropy = -torch.gather(softmaxed_head_res, dim=1, index=split_targets[idx].view(-1, 1))
                entropy = -tf.gather(softmaxed_head_res, indices=tf.reshape(split_targets[idx], [-1,1]))
                
            # If the target is in one of the splits, the probability is the p(tombstone) * p(word within tombstone)
            else:
                softmaxed_head_res = softmaxed_all_head_res[running_offset:running_offset + len(split_hiddens[idx])]
                #softmaxed_head_res = tf.gather(softmaxed_all_head_res, [running_offset:running_offset + len(split_hiddens[idx])])
                
                tail_res = self.logprob(weight, bias, split_hiddens[idx], splits=[idx], softmaxed_head_res=softmaxed_head_res)

                #head_entropy = softmaxed_head_res[:, -idx]
                
                head_entropy = tf.gather(softmaxed_head_res, [-idx], axis=1)
                
                
                # All indices are shifted - if the first split handles [0,...,499] then the 500th in the second split will be 0 indexed
                #indices = (split_targets[idx] - self.splits[idx]).view(-1, 1)
                
                indices = tf.reshape((split_targets[idx] - self.splits[idx]), [-1,1])
                
                # Warning: if you don't squeeze, you get an N x 1 return, which acts oddly with broadcasting
                #tail_entropy = torch.gather(torch.nn.functional.log_softmax(tail_res, dim=-1), dim=1, index=indices).squeeze()
                
                tail_entropy = tf.gather(tf.nn.log_softmax(tail_res, dim=-1), indices, axis=1)
                
                entropy = -(head_entropy + tail_entropy)
            ###
            running_offset += len(split_hiddens[idx])
            total_loss = tf.reduce_sum(entropy) #if total_loss is None else total_loss + tf.reduce_sum(entropy)

        return total_loss
