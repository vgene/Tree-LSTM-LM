from __future__ import print_function

import argparse
import tqdm
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from data import Reader, Saver
from charlm import CharLM

def repackage_hidden(h, use_cuda=False):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        if use_cuda:
            return Variable(h.data.cuda())
        else:
            return Variable(h.data)
    else:
        return tuple(repackage_hidden(v, use_cuda) for v in h)

def run_epoch(model, reader, criterion, is_train=False, use_cuda=False, lr=0.01):
    """
        reader: data provider
        criterion: loss calculation 
    """
    if is_train:
        model.train()
    else:
        model.eval()

    epoch_size = ((reader.file_length // model.batch_size)-1) // model.seq_length

    hidden = model.init_hidden()

    iters = 0
    costs = 0
    model.optimizer.zero_grad()
    for steps, (inputs, targets) in enumerate(reader.iterator_char(model.batch_size, model.seq_length)):
 
        inputs = Variable(torch.from_numpy(inputs.astype(np.int64)).transpose(0,1).contiguous())
        targets = Variable(torch.from_numpy(targets.astype(np.int64)).transpose(0,1).contiguous())
        if use_cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()
        targets = torch.squeeze(targets.view(-1, model.batch_size*model.seq_length))
        hidden = repackage_hidden(hidden, use_cuda=use_cuda)
        outputs, hidden = model(inputs, hidden)

        loss = criterion(outputs.view(-1, model.vocab_size), targets)
        costs += loss.data[0] * model.seq_length

        perplexity = np.exp(costs/((steps+1)*model.seq_length))
        #print("Iter {}/{},Perplexity:{}".format(steps+1, epoch_size, perplexity))

        if is_train:
            loss.backward()
            model.optimizer.step()

    return perplexity
        

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--layer_num", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seq_length", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--dropout_prob", type=float, default=0)
    parser.add_argument("--path", type=str, default='./data/test.txt')
    parser.add_argument("--log", type=str, default="./log/")
    parser.add_argument("--cuda", action="store_true")

    args = parser.parse_args()

    if args.cuda:
        print("Try to use CUDA")
        if torch.cuda.is_available():
            print("CUDA Found, use CUDA!")
        else:
            print("Sorry, No CUDA, fall back to CPU Mode")
            args.cuda = False
    else:
        print("Not using CUDA")
        args.cuda = False

    # Check path and log
    if not (os.path.exists(args.path) and os.path.isfile(args.path)):
        raise EnvironmentError("Data path not found")

    reader = Reader(args.path)
    args.vocab_size = reader.vocab_size

    # Create log path is not exist
    if os.path.exists(args.log):
        if not os.path.isdir(args.log):
            raise EnvironmentError("Log path exists but not a directory")
    else:
        os.mkdir(args.log)

    saver = Saver(args.log)

    model = CharLM(args, reader.charaters)
    if args.cuda:
        model.cuda()

    criterion = nn.CrossEntropyLoss()

    try:
        for epoch in tqdm.tqdm(range(args.epoch)):
            start_time = time.time()
            perplexity = run_epoch(model, reader, criterion, is_train=True, use_cuda=args.cuda)
            time_interval= time.time() - start_time
            print("Epoch:{}/{}, Perplexity:{}, Time:{}".format(epoch, args.epoch,
                                            perplexity, time_interval))

        saver.save(model)
    except KeyboardInterrupt:
        print("Stop and Save...")
        saver.save(model)

if __name__ == "__main__":
    main()
