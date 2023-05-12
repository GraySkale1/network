import torch
import torch.nn as nn

class model(nn.Module):
    #                 classes
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    class layer():
        def __init__(self, x, y, normalisation=False):
            self.weight = torch.randn(x , y)
            self.bias = torch.randn(y)
            self.normal = normalisation

        def __call__(self):
            return [self.weight, self.bias]


        def feed(self, x):
            out = x@self.weight + self.bias

            if self.normal == True:
                return torch.sigmoid(out)
            else:
                return out
            
    class embedding():
        def __init__(self, x, y, normalisation=False):
            self.weight = torch.randn(x, y)
            self.normal = normalisation

        def __call__(self):
            return [self.embedding]
        
        def feed(self, x):
            out = self.weight[x]
            if self.normal == True:
                return torch.sigmoid(out)
            else:
                return out
            
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __init__(self, layers=tuple(), input_size=int(), char_set_len=int()):
        super().__init__()

        self.param = []

        self.param.append(self.embedding(char_set_len, layers[0], normalisation=True))
        self.param.append(self.layer(input_size*layers[0], layers[1], normalisation=True))

        for i, dimension in enumerate(layers):
            if i <= 1:
                continue

            if i == (len(layers) - 1):
                self.param.append(self.layer(dimension, char_set_len))
                break
            self.param.append(self.layer(layers[i-1], dimension))



            


        

