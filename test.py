from network import neuralnet as net
import torch
import tiktoken

tokenisation_type = "cl100k_base"
context = 500
iterations = 100

enc = tiktoken.get_encoding(tokenisation_type)

model_name = str(input('Name of model:\n'))




test = net.load(model_name)


data = enc.encode('\n') * context


for i in range(iterations):
    test_set = torch.tensor(data)
    #returns the index of the highest value in the logit tensor
    logit = int(torch.argmax(torch.softmax(test.propagate(test_set), dim=0)))
    print(enc.decode([logit]), end='')
    data.append(logit)
    data.pop(0)
    