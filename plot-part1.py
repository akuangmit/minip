import csv
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

trainAccs = defaultdict(list)
validAccs = defaultdict(list)
reader = csv.DictReader(open('p1results.csv'))
for row in reader:
    method = row['Method']
    trainAcc = float(row['% Training'])
    validAcc = float(row['% Validation'])
    trainAccs[method].append(trainAcc)
    validAccs[method].append(validAcc)

n_groups = 3
tbase = trainAccs['Base']
vbase = validAccs['Base']

method = 'LR (0.01)'
method = 'LR (0.0001)'
method = 'LR/WD (0.01)'
# method = 'Adagrad'
# method = 'Adam'
tmethod = trainAccs[method]
vmethod = validAccs[method]

# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.25
# plt.ylim([0.1,0.5])

rects1 = plt.bar(index, tbase, bar_width,
                 color='b',
                 label='Base')

rects2 = plt.bar(index + bar_width, tmethod, bar_width,
                 color='g',
                 label=method)
 
plt.xlabel('Epochs')
plt.ylabel('% Top-5 Correct')
plt.title('Training Accuracies over Epochs')
plt.xticks(index + bar_width, (1,2,3))
plt.legend()

plt.tight_layout()
plt.show()
