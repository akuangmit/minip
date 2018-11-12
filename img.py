import gc
import sys
import pickle
from torch.autograd import Variable
import torch
import torch.nn as nn
torch.backends.cudnn.benchmark=True
import matplotlib.pyplot as plt
import numpy as np

import dataset

model_path = './'
# Parameters
num_epochs = 3
output_period = 100
batch_size = 100

val_loader, test_loader = dataset.get_val_test_loaders(batch_size)

for batch_num, (inputs, labels) in enumerate(test_loader, 1):
    lol = inputs[99].numpy()
    plt.imshow(np.transpose(lol, (1,2,0)))
    plt.show()
    break

# print('Starting evaluation on', model_path)
# run()
# print('Evaluation terminated for', model_path)
