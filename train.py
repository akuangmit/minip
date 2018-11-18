import gc
import sys
import pickle
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler
torch.backends.cudnn.benchmark=True

import dataset
from models.AlexNet import *
from models.ResNet import *

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

    criterion = nn.CrossEntropyLoss().to(device)
    # TODO: optimizer is currently unoptimized
    # there's a lot of room for improvement/different optimizers
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    trainAccLog = []
    valAccLog = []
    epoch = 1
    while epoch <= num_epochs:
        running_loss = 0.0
        for param_group in optimizer.param_groups:
            print('Current learning rate: ' + str(param_group['lr']))
        model.train()

        for batch_num, (inputs, labels) in enumerate(train_loader, 1):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()
            running_loss += loss.item()

            if batch_num % output_period == 0:
                print('[%d:%.2f] loss: %.3f' % (
                    epoch, batch_num*1.0/num_train_batches,
                    running_loss/output_period
                    ))
                running_loss = 0.0
                gc.collect()

        gc.collect()
        # save after every epoch
        torch.save(model.state_dict(), "models/model.%d" % epoch)

        # torch.load('./models/model1.1')
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
        with open('trainAcc.pkl', 'wb') as f:
            pickle.dump(trainAccLog, f)
            print('percent error for training: top1/5', 1-(top1Correct/(tot*1.0)), 1-top5Correct/(tot*1.0))
        # Calculate classification error and Top-5 Error
        # on training and validation datasets here
        model.eval()
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
        with open('valAcc.pkl', 'wb') as f:
            pickle.dump(valAccLog, f)
            print('percent error for validation: top1/5', 1-(top1Correct/(tot*1.0)), 1-top5Correct/(tot*1.0))

        gc.collect()
        epoch += 1

print('Starting training and validation')
run()
print('Training and validation terminated')
