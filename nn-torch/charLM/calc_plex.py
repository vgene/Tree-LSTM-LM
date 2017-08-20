import os
import codecs
import torch
from torch.autograd import Variable
from torch.nn.functional import softmax
from tqdm import tqdm
import argparse
import math


class Tester(object):
    """
        Parse Test File and calculate metrics like perplexity etc,.
    """
    def __init__(self, filename, batch_size, mapping):

        self.filename = filename 
        self.batch_size = batch_size
        self.mapping = dict(zip(mapping, range(len(mapping))))

        with codecs.open(self.filename, 'r', encoding="utf-8") as f:
            self.test_files = f.readlines()

        assert(self.batch_size <= len(self.test_files))
        # TODO: NO!
        self.test_files = self.test_files[0:self.batch_size]
        self.max_length = len(max(self.test_files, key=len))
        self.length = list(map(len, self.test_files))

        tensor = torch.zeros([self.batch_size, self.max_length]).long()
        # assert(len(self.test_files) == self.batch_size) #TODO:Really batch it with batch_size

        for idx, file_str in enumerate(self.test_files):
            if idx == self.batch_size:
                break
            try:
                tensor[idx][:len(file_str)] = torch.LongTensor(list(map(self.mapping.get, file_str)))
            except Exception as e:
                print(file_str)
                continue

        self.tensor = tensor

    def calc_perplexity(self, model, cuda=False):

        model.seq_length = 1
        hidden = model.init_hidden(self.batch_size)

        prime_input = Variable(self.tensor, volatile=True)

        if cuda:
            hidden = tuple(h.cuda() for h in hidden) # For LSTM especially
            prime_input = prime_input.cuda()

        log_per = 0
        for p in tqdm(range(self.max_length-1)):
            output, hidden = model(prime_input[:, p], hidden)

            # print(output)
            output = output.view(-1, model.vocab_size)
            for i in range(self.batch_size):
                if p >= self.length[i]-1:
                    continue
                #probs = softmax(output[i])
                #log_per += math.log(probs.data[model.mapping.index(self.test_files[i][p+1])])

        log_per /= sum(self.length[0:self.batch_size])
        print(log_per)
        perplexity = 1/math.exp(log_per)
        return perplexity

def main():
    import sys
    reload(sys)
    sys.setdefaultencoding("utf-8")
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--model', type=str)
    argparser.add_argument('--test_file', type=str)
    argparser.add_argument('--cuda', action='store_true')
    args = argparser.parse_args()

    model = torch.load(args.model)
    print(model.vocab_size)
    batch_size = 500
    tester = Tester(args.test_file, batch_size, model.mapping)
    perplexity = tester.calc_perplexity(model, cuda=args.cuda)
    print("Test File: {}, Perplexity:{}".format(args.test_file, perplexity))

if __name__ == "__main__":
    main()
