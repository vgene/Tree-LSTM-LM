from __future__ import print_function
import codecs
import os
import time
import collections
import torch
import numpy as np

DEBUG_MODE = False

class Reader(object):
    """
        Provider an input interface, iterator will yield characters sequences or word sequences
    """
    def __init__(self, filepath):
        """ Read file and set self.file """
        if not os.path.exists(filepath):
            raise EnvironmentError("File not exist")

        with codecs.open(filepath, "r", encoding="utf-8") as f:
            raw_file = f.read()

        counter = collections.Counter(raw_file)
        # sort all data
        count_pairs = sorted(counter.items(), key=lambda x: -x[1])
        # get character in appearing count sequence
        self.charaters, _ = zip(*count_pairs)
        self.vocab_size = len(self.charaters)
        self.vocab = dict(zip(self.charaters, range(len(self.charaters))))
        self.vocab_size = len(self.charaters)
        self.file = np.array(list(map(self.vocab.get, raw_file)))

        if DEBUG_MODE:
            print(self.charaters)
            print(self.vocab_size)

    @property
    def file_length(self):
        return len(self.file)

    def iterator_char(self, batch_size, seq_length):
        data = np.array(self.file, dtype=np.int32)
        data_len = len(data)

        batch_len = data_len // batch_size

        final_data = np.zeros([batch_size, batch_len], dtype=np.int32)
        for i in range(batch_size):
            final_data[i] = data[batch_len *i:batch_len*(i+1)]

        epoch_size = (batch_len-1) // seq_length

        if epoch_size == 0:
            raise ValueError("Epoch size equals zero")

        for i in range(epoch_size):
            inputs = final_data[:, i*seq_length:(i+1)*seq_length]
            targets = final_data[:, i*seq_length+1:(i+1)*seq_length+1]
            yield inputs, targets

class Saver(object):
    """
        Provide an output interface
    """
    def __init__(self, logpath):
        if not (os.path.exists(logpath) and os.path.isdir(logpath)):
            raise EnvironmentError("Log Path Not Find")
        self.logpath = logpath

    def save(self, model):
        cur_time = time.strftime("%B%d-%H%M%S")
        filepath = os.path.join(self.logpath, cur_time+".pth")
        torch.save(model, filepath)

        print("Save as file: {}".format(filepath))
