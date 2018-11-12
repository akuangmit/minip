import gc
import sys
import pickle
from torch.autograd import Variable
import torch
import torch.nn as nn
torch.backends.cudnn.benchmark=True

import dataset
from models.AlexNet import *
from models.ResNet import *

model_path = './'
def run():
    # Parameters
    num_epochs = 3
    output_period = 100
    batch_size = 100

    # setup the device for running
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = resnet_18()
    model = model.to(device)

    train_loader, val_loader = dataset.get_data_loaders(batch_size)
    num_train_batches = len(train_loader)

    trainAccLog = []
    valAccLog = []
    epoch = 1
    while epoch <= num_epochs:
        model.load_state_dict(torch.load(model_path + 'model.' + str(epoch)))
        model.eval()

        top1Correct = 0
        top5Correct = 0
        tot = 0
        for batch_num, (inputs, labels) in enumerate(train_loader, 1):
            inputs = inputs.to(device)

            outputs = model(inputs)
            prediction = outputs.to('cpu')
            for i in range(batch_size):
                curr = prediction[i].unsqueeze(1)
                _, cls = torch.max(curr, dim=0)
                _, cls5 = torch.topk(curr, 5, dim=0)
                if labels[i] in cls:
                    top1Correct += 1
                if labels[i] in cls5:
                    top5Correct += 1
                tot += 1
        trainAccLog.append((top1Correct, top5Correct, tot))
        with open(model_path + 'trainAcc.pkl', 'wb') as f:
            pickle.dump(trainAccLog, f)
            print('for training: top1/5', top1Correct, top5Correct)

        # Calculate classification error and Top-5 Error
        # on training and validation datasets here
        top1Correct = 0
        top5Correct = 0
        tot = 0
        for batch_num, (inputs, labels) in enumerate(val_loader, 1):
            inputs = inputs.to(device)

            outputs = model(inputs)
            prediction = outputs.to('cpu')
            for i in range(batch_size):
                curr = prediction[i].unsqueeze(1)
                _, cls5 = torch.topk(curr, 5, dim=0)
                if labels[i] in cls5[0]:
                    top1Correct += 1
                if labels[i] in cls5:
                    top5Correct += 1
                tot += 1
        valAccLog.append((top1Correct, top5Correct, tot))
        with open(model_path + 'valAcc.pkl', 'wb') as f:
            pickle.dump(valAccLog, f)
            print('for validation: top1/5', top1Correct, top5Correct)

        gc.collect()
        epoch += 1

print('Starting evaluation on', model_path)
run()
print('Evaluation terminated for', model_path)
