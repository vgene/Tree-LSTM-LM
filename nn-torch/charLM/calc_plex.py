import os
import codecs
import torch
from torch.autograd import Variable
from torch.nn.functional import softmax
from tqdm import tqdm
import argparse
import math


class BatchProvider(object):
    def __init__(self):
        pass
def char_tensor(string, all_characters):
    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        try:
            tensor[c] = all_characters.index(string[c])
        except:
            continue
    return tensor

def calc_perplexity(model, test_str, cuda=False):

    model.seq_length = 1
    hidden = model.init_hidden(1)

    tensor = char_tensor(test_str, model.mapping)
    prime_input = Variable(tensor.unsqueeze(0), volatile=True)

    if cuda:
        hidden = tuple(h.cuda() for h in hidden)
        prime_input = prime_input.cuda()

    for p in tqdm(range(len(test_str)-1)):
        #output, hidden = model(prime_input[:,p], hidden)
        if (p == 0):
            log_per = 0
        else:
            # print(output)
            output = output.view(-1, model.vocab_size)
            output = softmax(output[0])

            log_per += math.log(output.data[model.mapping.index(test_str[p+1])])
        if (p == 20000):
            break
    log_per /= len(test_str)
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
    with codecs.open(args.test_file, 'r', encoding="utf-8") as f:
        test_str = f.read()

    perplexity = calc_perplexity(model, test_str, cuda=args.cuda)
    print("Test File: {}, Perplexity:{}".format(args.test_file, perplexity))

if __name__ == "__main__":
    main()
