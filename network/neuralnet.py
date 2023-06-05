import torch
import torch.nn as nn
import configparser
import pickle
import os.path


class model(nn.Module):
    #                 self.classes
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    class layer(nn.Module):
        def __init__(self, x, y, normalisation=False):
            self.weight = torch.randn((x , y), requires_grad=True)
            self.bias = torch.randn(y, requires_grad=True)
            self.normal = normalisation

        def __call__(self):
            return [self.weight, self.bias]


        def feed(self, x):
            out = x@self.weight + self.bias

            if self.normal == True:
                return torch.sigmoid(out)
            else:
                return out
            
    class embedding(nn.Module):
        def __init__(self, x, y, matrix_view_y, normalisation=False):
            self.weight = torch.randn((x, y), requires_grad=True)
            self.normal = normalisation
            self.view_y = matrix_view_y


        def __call__(self):
            return [self.weight]
        
        def feed(self, x): #x must be tensor of ints
            out = (self.weight[x]).view(-1, self.view_y)

            if self.normal == True:
                return torch.sigmoid(out)
            else:
                return out
    
    #             class atributes
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __init__(self, layers:tuple, context_size:int, char_set_len:int, modelname:str):
        super(model, self).__init__()

        settings = configparser.ConfigParser()
        settings.read('network_settings.ini')

        self.param_obj = [] #holds each layer's object

        self.name = modelname

        #embed + first layer need to be defined specifically due to matrix mult compatibility
        self.param_obj.append(self.embedding(char_set_len, layers[0], matrix_view_y=context_size*layers[0], normalisation=True))
        self.param_obj.append(self.layer(context_size*layers[0], layers[1], normalisation=True))

        for i, dimension in enumerate(layers):
            if i <= 1: #layers used for embed + first layer are skipped
                continue

            #[layer-1] used for x so that layers fit in order
            self.param_obj.append(self.layer(layers[i-1], dimension, normalisation=True))

        #final matrix must have rows of length of 'char_set_len' for cross entropy later
        self.param_obj.append(self.layer(layers[-1], char_set_len, normalisation=False))

        

        self.parameter = []

        for obj in self.param_obj:
            self.parameter.append(obj.weight)
            if isinstance(obj, model.layer):
                self.parameter.append(obj.bias)

        for matrix in self.parameter:
            nn.parameter.Parameter(matrix, requires_grad=True) 

        #creates optimiser object
        self.optimiser = torch.optim.SGD(self.parameter, lr=float(settings['backprop']['lr']), dampening=0.5)

    def propagate(self, x): #iterates through all layers 
        for layer in self.param_obj:
            temp = layer.feed(x)
            x = temp
        return x
    
    def save(self):
        path = os.path.join(os.path.dirname(__file__), '..\models\\')
        with open(path + self.name + '.model', 'wb') as model:
            pickle.dump(self, model)

            

def load(modelname:str):
    path = os.path.join(os.path.dirname(__file__), 'models\\')
    with open(path + modelname + '.model', 'r') as model:
        return pickle.load(model)


            


        

