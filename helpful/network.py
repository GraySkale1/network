import torch
import torch.nn as nn

class model(nn.Module):
    #                 self.classes
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
        
        def feed(self, x): #x must be tensor of ints!!!
            out = self.weight[x]
            if self.normal == True:
                return torch.sigmoid(out)
            else:
                return out
    
    #             class atributes
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __init__(self, layers=tuple(), input_size=int(), char_set_len=int()):
        super().__init__()

        self.param = [] #holds each layer's object

        #embed + first layer need to be defined specifically due to matrix mult compatibility
        self.param.append(self.embedding(char_set_len, layers[0], normalisation=True))
        self.param.append(self.layer(input_size*layers[0], layers[1], normalisation=True))

        for i, dimension in enumerate(layers):
            if i <= 1: #layers used for embed + first layer are skipped
                continue

            if i == (len(layers) - 1):
                #final layer needs to have as many outputs as characters present
                self.param.append(self.layer(dimension, char_set_len)) 
                break

            #[layer-1] used for x so that layers fit in order
            self.param.append(self.layer(layers[i-1], dimension, normalisation=True))

        #adds each element from param to model.parameters()
        for obj in self.param:
            nn.parameter.Parameter(obj.weight, requires_grad=True)
            if isinstance(obj, self.layer):
                nn.parameter.Parameter(obj.bias, requires_grad=True)

        def propagate(x):
            for layer in self.param:
                x = layer.feed(x)
            return x

            
            


            


        

