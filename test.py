from network import neuralnet as net
import torch
import tiktoken
import configparser


device = 'cuda:0' if torch.cuda.is_available else 'cpu'

settings = configparser.ConfigParser()
settings.read('network_settings.ini')

context = int(settings['network']['context'])
iterations = int(settings['backprop']['iterations'])

tokenisation_type = settings['network']['tokenisation_type']


with open('data/dataset.txt', 'r') as _string:
    text = _string.read()

enc = tiktoken.get_encoding(tokenisation_type)


token_text = enc.encode(text)

model_name = str(input('Name of model:\n'))


test = net.load(model_name)

random_batches_index = torch.randint(context,  len(token_text) + 1, (iterations,))

inp = enc.encode(str(input('input text: ')))

data = token_text[random_batches_index[0] - context + len(inp):random_batches_index[0]] + inp


for i in range(iterations):
    test_set = torch.tensor(data).to(device)
    #returns the index of the highest value in the logit tensor
    logit = torch.argmax(torch.sigmoid(test.propagate(test_set)))
    print(logit)
    #print(enc.decode([logit]), end='')
    data.append(logit)
    data.pop(0)
    