import torch
import os
import re

ptb_path = "./data/"

vocab_idx = 0
vocab_map = {}

def tokenize(text):
    """ Mimic Lua's stringx.split() behavior. """
    return re.findall(r"\S+", text)  # Split by whitespace but keep everything else intact.

def load_data(fname):
    global vocab_idx
    
    print(f"\n[Loading Data] File: {fname}")
    
    # Read file and replace newlines with <eos>
    with open(fname, 'r', encoding='utf-8') as f:
        data = f.read()
    
    data = data.replace('\n', '<eos>')  # Match Lua behavior
    data = data.split()  # Equivalent to stringx.split in Lua
            
    x = torch.zeros(len(data), dtype=torch.long)
    
    for i in range(len(data)):
        if data[i] not in vocab_map:
            vocab_idx += 1
            vocab_map[data[i]] = vocab_idx
        x[i] = vocab_map[data[i]]
    return x

def replicate(x_inp, batch_size):
    """ Replicates and shifts data exactly like Lua """
    s = x_inp.size(0)
    x = torch.zeros((s // batch_size, batch_size), dtype=torch.long)

    for i in range(batch_size):
        start = round(i * s / batch_size)  # Mimic Lua rounding
        finish = start + x.size(0)
        x[:, i] = x_inp[start:finish]
    return x

def traindataset(batch_size):
    x = load_data(os.path.join(ptb_path, "ptb.train.txt"))
    x = replicate(x, batch_size)
    return x

def testdataset(batch_size):
    x = load_data(os.path.join(ptb_path, "ptb.test.txt"))
    x = x.view(-1, 1).expand(-1, batch_size).clone()
    return x

def validdataset(batch_size):
    x = load_data(os.path.join(ptb_path, "ptb.valid.txt"))
    x = replicate(x, batch_size)
    return x

datasets = {
    "traindataset": traindataset,
    "validdataset": validdataset,
    "testdataset": testdataset
}