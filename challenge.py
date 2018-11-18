import gc
import os
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

    val_loader, test_loader = dataset.get_val_test_loaders(batch_size)
    epoch = 1

    # know what model is before calling this file bc of validation
    model.load_state_dict(torch.load('models/trialTwo/model.10'))
    model.eval()

    # INITializes variables
    top5Correct = 0
    tot = 0
    dictOldNewLabels = {}
    f = open('testResults.txt', 'w')

    # HANDLES data_loader error
    dir = './data/train'
    count = 0
    # get the order that data_loader sees the classes
    subdirs = [x[0] for x in os.walk(dir)]
    subdirs.sort(key=lambda x:str(x))
    # remaps indices of the data_loader to the real classes
    ind = len(dir)
    subdirs= subdirs[1:]
    for subdir in subdirs:
        dictOldNewLabels[str(count)] = subdir[ind+1:]
        count+=1

    # CALCULATE classification error and Top-5 Error
    filePathNumber = 0
    for batch_num, (inputs, labels) in enumerate(test_loader, 1):
        inputs = inputs.to(device)

        outputs = model(inputs)
        prediction = outputs.to('cpu')

        for i in range(batch_size):
            lineArray = []
            filePathNumber+=1
            filePath = 'test/'+str(filePathNumber).zfill(8)+".jpg"
            lineArray.append(filePath)
            curr = prediction[i].unsqueeze(1)
            _, cls5 = torch.topk(curr, 5, dim=0)

            cls5List = list(cls5.numpy())
            for category in cls5List:
                actualCategory = dictOldNewLabels[str(category[0])]
                lineArray.append(str(actualCategory))

            line = " ".join(lineArray)+"\n"
            if filePathNumber==10000:
                line = " ".join(lineArray) # don't append the "\n"
            f.write(line)
    gc.collect()

print('Starting challenge')
run()
print('Challenge terminated')
