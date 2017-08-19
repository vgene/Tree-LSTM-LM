#!/usr/bin/env python
# https://github.com/spro/char-rnn.pytorch

import os
import argparse
import torch
from torch.autograd import Variable

def char_tensor(string, all_characters):
    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        try:
            tensor[c] = all_characters.index(string[c])
        except:
            continue
    return tensor

def generate(model, prime_str='A', predict_len=100, temperature=0.8, cuda=False):
    hidden = model.init_hidden(1)
    tensor = char_tensor(prime_str, model.mapping)
    prime_input = Variable(tensor.unsqueeze(0))
    #print(prime_input)
    if cuda:
        hidden = tuple(h.cuda() for h in hidden)
        prime_input = prime_input.cuda()
    predicted = prime_str
    model.seq_length = 1

    #print(hidden)
    #print(prime_input[:,0])
    # Use priming string to "build up" hidden state
    for p in range(len(prime_str) - 1):
        _, hidden = model(prime_input[:,p], hidden)
        
    inp = prime_input[:,-1]
    
    for p in range(predict_len):
        output, hidden = model(inp, hidden)
        
        # Sample from the network as a multinomial distribution
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]

        # Add predicted character to string and use as next input
        predicted_char = model.mapping[top_i]
        predicted += predicted_char
        inp = Variable(char_tensor(predicted_char, model.mapping).unsqueeze(0))
        if cuda:
            inp = inp.cuda()

    return predicted

# Run as standalone script
if __name__ == '__main__':

# Parse command line arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument('filename', type=str)
    argparser.add_argument('-p', '--prime_str', type=str, default='A')
    argparser.add_argument('-l', '--predict_len', type=int, default=100)
    argparser.add_argument('-t', '--temperature', type=float, default=0.8)
    argparser.add_argument('--cuda', action='store_true')
    args = argparser.parse_args()

    model = torch.load(args.filename)
    del args.filename
    print(generate(model, **vars(args)))
