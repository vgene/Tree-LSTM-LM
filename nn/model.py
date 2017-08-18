#!/usr/bin/python
# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import legacy_seq2seq

import numpy as np
import math

class Model():
    def __init__(self, args, training=True):
        self.args = args
        if not training:
            args.batch_size = 1
            args.seq_length = 1

        if args.model == 'rnn':
            cell_fn = rnn.BasicRNNCell
        elif args.model == 'gru':
            cell_fn = rnn.GRUCell
        elif args.model == 'lstm':
            cell_fn = rnn.BasicLSTMCell
        elif args.model == 'nas':
            cell_fn = rnn.NASCell
        else:
            raise Exception("model type not supported: {}".format(args.model))


        # building cells
        cells = []
        for _ in range(args.num_layers):
            cell = cell_fn(args.rnn_size)
            if training and (args.output_keep_prob < 1.0 or args.input_keep_prob < 1.0):
                cell = rnn.DropoutWrapper(cell,
                                          input_keep_prob=args.input_keep_prob,
                                          output_keep_prob=args.output_keep_prob)
            cells.append(cell)

        self.cell = cell = rnn.MultiRNNCell(cells, state_is_tuple=True)

        self.input_data = tf.placeholder(
            tf.int32, [args.batch_size, args.seq_length])
        self.targets = tf.placeholder(
            tf.int32, [args.batch_size, args.seq_length])
        self.initial_state = cell.zero_state(args.batch_size, tf.float32)

        with tf.variable_scope('rnnlm'):
            softmax_w = tf.get_variable("softmax_w",
                                        [args.rnn_size, args.vocab_size])
            softmax_b = tf.get_variable("softmax_b", [args.vocab_size])

        embedding = tf.get_variable("embedding", [args.vocab_size, args.rnn_size])
        inputs = tf.nn.embedding_lookup(embedding, self.input_data)

        # dropout beta testing: double check which one should affect next line
        if training and args.output_keep_prob:
            inputs = tf.nn.dropout(inputs, args.output_keep_prob)

        inputs = tf.split(inputs, args.seq_length, 1)
        inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

        def loop(prev, _):
            prev = tf.matmul(prev, softmax_w) + softmax_b
            prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
            return tf.nn.embedding_lookup(embedding, prev_symbol)

        outputs, last_state = tf.nn.dynamic_rnn(cell, tf.nn.embedding_lookup(embedding, self.input_data), initial_state=self.initial_state, scope='rnnlm')

        output = tf.reshape(tf.concat(outputs, 1), [-1, args.rnn_size])


        # Get logits, probs, loss and train_loss
        self.logits = tf.matmul(output, softmax_w) + softmax_b
        self.probs = tf.nn.softmax(self.logits)
        loss = legacy_seq2seq.sequence_loss_by_example(
                [self.logits],
                [tf.reshape(self.targets, [-1])],
                [tf.ones([args.batch_size * args.seq_length])])
        self.cost = tf.reduce_sum(loss) / args.batch_size / args.seq_length
        with tf.name_scope('cost'):
            self.cost = tf.reduce_sum(loss) / args.batch_size / args.seq_length
        self.final_state = last_state
        self.lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),
                args.grad_clip)
        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

        # instrument tensorboard
        tf.summary.histogram('logits', self.logits)
        tf.summary.histogram('loss', loss)
        tf.summary.scalar('train_loss', self.cost)

    def sample(self, sess, chars, vocab, num=200, prime='The ', sampling_type=1):
        #
        print(self.input_data)
        state = sess.run(self.cell.zero_state(1, tf.float32))
        for char in prime[:-1]:
            x = np.zeros((1, 1))
            x[0, 0] = vocab[char]
            feed = {self.input_data: x, self.initial_state: state}
            [state] = sess.run([self.final_state], feed)

        # Roulette
        def weighted_pick(weights):
            t = np.cumsum(weights)
            s = np.sum(weights)
            return(int(np.searchsorted(t, np.random.rand(1)*s)))

        ret = prime
        char = prime[-1]
        for n in range(num):
            x = np.zeros((1, 1))
            x[0, 0] = vocab[char]
            feed = {self.input_data: x, self.initial_state: state}
            [probs, state] = sess.run([self.probs, self.final_state], feed)
            p = probs[0]

            # Choosing the most possible one
            if sampling_type == 0:
                sample = np.argmax(p)
            # Choosing the most possible one within a word and choose randomly before a word
            elif sampling_type == 2:
                if char == ' ':
                    sample = weighted_pick(p)
                else:
                    sample = np.argmax(p)
            elif sampling_type == 3:
                if char == u'，' or char == u'。' or char == u"\n" or char == u' ':
                    sample = weighted_pick(p)
                    #ret += ''
                else:
                    sample = np.argmax(p)

            elif sampling_type == 4:
                if char == ';':
                    sample = weighted_pick(p)
                else:
                    sample = np.argmax(p)
            # Choosing randomly
            else:  # sampling_type == 1 default:
                sample = weighted_pick(p)

            pred = chars[sample]
            ret += pred
            char = pred
        return ret

    def getPerplexity(self, sess, chars, vocab, prime):

        if len(prime) == 0:
            return 0
        
        print('Total Length:',len(prime))

        state = sess.run(self.cell.zero_state(1, tf.float32))

        sumLogProb = 0

        char = prime[0]
        x = np.zeros((1, 1))
        x[0, 0] = vocab[char]
        feed = {self.input_data: x, self.initial_state: state}
        [probs, state] = sess.run([self.probs, self.final_state], feed)
        p = probs[0]

        import tqdm
        for char in tqdm.tqdm(prime[1:]):
            # if the char is not in the vocabulary
            # set as (1/len(vocab))
            # TODO: this should not happen
            try:
                sumLogProb += math.log(p[vocab[char]])
            except Exception as e:
                sumLogProb += math.log(1/len(vocab))
            x = np.zeros((1, 1))
            x[0, 0] = vocab[char]
            feed = {self.input_data: x, self.initial_state: state}
            [probs, state] = sess.run([self.probs, self.final_state], feed)
            p = probs[0]

        return math.exp(-sumLogProb/len(prime))

