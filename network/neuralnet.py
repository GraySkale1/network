import torch
import torch.nn as nn
from torch.nn import functional as F
import configparser
import pickle
import os.path
import torch.cuda


class model(nn.Module):
    def __init__(self, layers:list, context_size:int, char_set_len:int, modelname:str):
        super(model, self).__init__()

        settings = configparser.ConfigParser()
        settings.read('network_settings.ini')

        self.name = modelname

        self.context = context_size
        self.flatten_size = context_size*layers[0]
        

        self.layers = nn.ModuleList()

        #creates first layers that embedd input values
        self.embedding = nn.Embedding(num_embeddings=char_set_len, embedding_dim=layers[0])

        
        #linear gives weight and bias in one
        layer = nn.Linear(context_size*layers[0], layers[1], bias=True)
        self.layers.append(layer)

        #adds layers to nn.Modulelist
        for i in range(len(layers) - 2):
            layer = nn.Linear(layers[i+1], layers[i+2], bias=True)
            self.layers.append(layer)

        self.final_layer = nn.Linear(layers[-1], char_set_len, bias=True)

        self.optimiser = torch.optim.SGD(params=self.parameters(), lr=float(settings['backprop']['lr']))

    def propagate(self, x):
        #moves x to GPU
        embed = self.embedding(x)
        #flattens 3d matrix to 2d
        h = torch.sigmoid(embed.view(-1, self.flatten_size))
        del embed

        for layer in self.layers:
            h = torch.sigmoid(F.relu(layer(h)))

        output = self.final_layer(h)
        del h
        #release excess memory
        #torch.cuda.empty_cache()

        return output
    
    def save(self):
        path = os.path.join(os.path.dirname(__file__), '..\models\\')
        with open(path + self.name + '.model', 'wb') as model:
            pickle.dump(self, model)

            

def load(modelname:str):
    path = os.path.join(os.path.dirname(__file__), '..\models\\')
    with open(path + modelname + '.model', 'rb') as model:
        return pickle.load(model)
