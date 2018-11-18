import gc
import sys
import pickle
from torch.autograd import Variable
import torch
import torch.nn as nn
torch.backends.cudnn.benchmark=True

import dataset
from models.AKATNet import *

def run():
    # Parameters
    num_epochs = 10
    output_period = 100
    batch_size = 100

    # setup the device for running
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = resnet_18()
    model = model.to(device)

    val_loader, _ = dataset.get_val_test_loaders(batch_size)
    epoch = 1

    # run validation on different models (USER INPUT)
    model.load_state_dict(torch.load('models/trialTwo/model.10'))
    model.eval()

    # Calculate classification error and Top-5 Error
    # on validation dataset here
    top5Correct = 0
    tot = 0
    for batch_num, (inputs, labels) in enumerate(val_loader, 1):
        inputs = inputs.to(device)

        outputs = model(inputs)
        prediction = outputs.to('cpu')

        for i in range(batch_size):
            curr = prediction[i].unsqueeze(1)
            _, cls5 = torch.topk(curr, 5, dim=0)

            cls5List = list(cls5.numpy())
            if labels[i] in cls5:
                top5Correct += 1
            tot += 1
    print('top 5 percent error for validation set:', 1-(top5Correct/(tot*1.0)))
    gc.collect()

print('Starting challenge')
run()
print('Challenge terminated')
