import network.neuralnet as net
from torch.nn import functional as F
import torch
from network import graphics
import os
import configparser
import tiktoken
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = 'max_split_size_mb:512'


# Check if CUDA is available
if torch.cuda.is_available():
    # Initialize CUDA
    device = torch.device("cuda")

    cuda_id = torch.cuda.current_device()
    print(torch.cuda.get_device_name(cuda_id))

settings = configparser.ConfigParser()
settings.read('network_settings.ini')

context = int(settings['network']['context'])
iterations = int(settings['backprop']['iterations'])
batch_size = int(settings['network']['batch_size'])
bar_length = int(settings['graphics']['bar_length'])
print_interval = int(settings['graphics']['print_interval'])
save_interval = int(settings['network']['save_interval'])
loss_memory = int(settings['graphics']['loss_memory'])
epoch_bar_length = int(settings['graphics']['epoch_bar_length'])
epoch_print_interval = int(settings['graphics']['epoch_print_interval'])
layer_set = [5,1000,500,700,800]
tokenisation_type = settings['network']['tokenisation_type']
epoch_left = iterations


enc = tiktoken.get_encoding(tokenisation_type)

data_folder = os.path.join(os.path.dirname(__file__), 'data\\')

with open(data_folder + 'dataset.txt', 'r') as dataset:
    raw_data = dataset.read()

#added spacer for dynamic context
raw_data = ' '*context + raw_data

#char_set = sorted(set(raw_data))


#token_encode = {j:i for i,j in enumerate(char_set)}

#data = [token_encode[char] for char in raw_data]
data = enc.encode(raw_data)
char_size = max(set(data)) + 1

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

token_x = torch.tensor(token_x, dtype=torch.int64)
token_y = torch.tensor(token_y, dtype=torch.int64)
#converts token_y to tensor on seperate thread so loading screen works
#token_y = Thread(target= convert_to_tensor, args=(token_y,)).start()


batches = len(token_y)

train = net.model(layers=layer_set, context_size=context, char_set_len=char_size, modelname='gaming').to(device)

random_batches_index = torch.randint(0, batches - batch_size - 1, (iterations,))

#shift line back
print("\033[3A")
print(f'Current Epoch Length: {iterations}')

#stores loss for average
loss_level = []
epoch = graphics.progress(epoch_bar_length)
print(epoch)
for i in range(iterations):
    #create batch
    batch_x = token_x[random_batches_index[i]:random_batches_index[i] + batch_size].to(device)
    batch_y = token_y[random_batches_index[i]:random_batches_index[i] + batch_size].to(device)

    #feed forward

    logits = train.propagate(batch_x)
    
    loss = F.cross_entropy(logits, batch_y)

    torch.cuda.empty_cache()

    loss_level.append(loss.item())
    
    loss.backward()

    train.optimiser.step()
    
    print(loss.item())

    del logits

    train.optimiser.step()
    if i % save_interval:
        train.save()
    
    if len(loss_level) > loss_memory:
        loss_level.pop(0)
train.save()

