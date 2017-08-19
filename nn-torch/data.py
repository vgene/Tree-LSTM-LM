from __future__ import print_function
import unidecode
import string
import random
import time
import torch
import math
import os
import numpy as np

DEBUG_MODE = True

class Reader():

    def __init__(self, filepath):
        """ Read file and set self.file """
        if not os.path.exists(filepath):
            raise EnvironmentError("File not exist")

        self.raw_file = unidecode.unidecode(open(filepath).read())
        self.charaters = list(set(self.raw_file))
        self.vocab_size = len(self.charaters)
        self.file = []
    
        for i in self.raw_file:
            self.file.append(self.charaters.index(i))

        if DEBUG_MODE:
            print(self.charaters)
            print(self.vocab_size)

    @property
    def file_length(self):
        return len(self.file)

    def iterator_char(self, batch_size, seq_length):
        data = np.zeros(self.file_length, dtype=np.int32)
        data_len = len(data)

        batch_len = data_len // batch_size

        final_data = np.zeros([batch_size, batch_len], dtype = np.int32)
        for i in range(batch_size):
            final_data[i] = data[batch_len *i:batch_len*(i+1)]

        epoch_size = (batch_len-1) // seq_length

        if (epoch_size == 0):
            raise ValueError("Epoch size equals zero")

        for i in range(epoch_size):
            inputs = final_data[:, i*seq_length:(i+1)*seq_length]
            targets = final_data[:, i*seq_length+1:(i+1)*seq_length+1]
            yield inputs, targets