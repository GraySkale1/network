import network.neuralnet as net
from torch.nn import functional as F
import torch
from network import graphics
import os
import configparser
import tiktoken


enc = tiktoken.get_encoding("cl100k_base")


settings = configparser.ConfigParser()
settings.read('network_settings.ini')

context = 500
iterations = 1000
batch_size = 1000
bar_length = 50
print_interval = 400
save_interval = 100
loss_memory = 100
epoch_length = 100
epoch_print_interval = 50

data_folder = os.path.join(os.path.dirname(__file__), 'data\\')

with open(data_folder + 'dataset.txt', 'r') as dataset:
    raw_data = dataset.read()

#added spacer for dynamic context
raw_data = ' '*context + raw_data

#char_set = sorted(set(raw_data))


#token_encode = {j:i for i,j in enumerate(char_set)}

#data = [token_encode[char] for char in raw_data]
data = enc.encode(raw_data)
char_size = max(data)

# x: 2d array of every possible sequensial list of characters of length 'context'
# y: 1d array with the next character for each list in x

loading = graphics.progress(bar_length)

interval = round((len(data) - context) / bar_length)

token_x, token_y = [], []
print('Loading dataset:')
for i in range(len(data) - context): #update and print_interval seperate as print operation is very slow
    token_x.append(data[i:i+context])
    token_y.append(data[i+context])

    if i % interval == 0:
        loading.update()

    if i % print_interval == 0:
        print(loading)
token_y
#converts token_y to tensor on seperate thread so loading screen works
#token_y = Thread(target= convert_to_tensor, args=(token_y,)).start()


batches = len(token_y)

print('New or load model (n/l)')

train = net.model(layers=(12,400,600,400,200,500,356,370), context_size=context, char_set_len=char_size, modelname='gaming')

random_batches_index = torch.randint(0, batches - batch_size, (iterations,))

#shift line back
print("\033[3A")
print(f'Current Epoch Length: {iterations}')

#stores loss for average
loss_level = []
epoch = graphics.progress(epoch_length)
for i in range(iterations):
    #create batch
    batch_x = token_x[random_batches_index[i]:random_batches_index[i] + batch_size]
    batch_y = token_y[random_batches_index[i]:random_batches_index[i] + batch_size]
    batch_y = torch.LongTensor(batch_y)

    #feed forward
    logits = train.propagate(batch_x)
    
    loss = F.cross_entropy(logits, batch_y)

    loss_level.append(loss.item())
    
    loss.backward()

    train.optimiser.step()


    if i % (iterations // epoch_length) and i != 0:
        average = sum(loss_level) / len(loss_level)
        epoch.update(ex_info=f'loss: {average}')

    if i % (iterations // epoch_print_interval) and i != 0:
        print(epoch)

    train.optimiser.step()
    if i % save_interval:
        train.save()
    
    if len(loss_level) > loss_memory:
        loss_level.pop(0)

