import gc
import os
import sys
import pickle
from torch.autograd import Variable
import torch
import torch.nn as nn
torch.backends.cudnn.benchmark=True

import dataset
# from models.AlexNet import *
# from models.ResNet import *
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


    model.load_state_dict(torch.load('models/trialTwo/model.10')) #we should know what this model is before calling this file
    model.eval()

    # Calculate classification error and Top-5 Error
    # on training and validation datasets here
    top5Correct = 0
    tot = 0
    dictOldNewLabels = {}
    f = open('testResults.txt', 'w')
    dir = './data/train'
    count = 0
    subdirs = [x[0] for x in os.walk(dir)]
    subdirs.sort(key=lambda x:str(x))
    ind = len(dir)
    subdirs= subdirs[1:]
    for subdir in subdirs:
        dictOldNewLabels[str(count)] = subdir[ind+1:]
        count+=1
    #print(dictOldNewLabels)
    # dictOldNewLabels = {'2': '10', '11': '3'} etc!
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
                # categoryString = str(category[0])
                # if len(categoryString)==1:
                #     categoryString = "0"+categoryString
                # lineArray.append(categoryString)
                lineArray.append(str(actualCategory))
            line = " ".join(lineArray)+"\n"
            if filePathNumber==10000:
                line = " ".join(lineArray) #don't append the "\n"
            f.write(line)
    # print('top 5 percent error for testset:', 1-(top5Correct/(tot*1.0)))
    gc.collect()

print('Starting challenge')
run()
print('Challenge terminated')
