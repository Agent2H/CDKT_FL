# from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from tqdm import trange
import numpy as np
import random
import json
import os
import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset.data), shuffle=False)
testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset.data), shuffle=False)

for _, train_data in enumerate(trainloader, 0):
    trainset.data, trainset.targets = train_data
for _, train_data in enumerate(testloader, 0):
    testset.data, testset.targets = train_data

random.seed(1)
np.random.seed(1)
NUM_USERS = 100  # should be muitiple of 10
NUM_LABELS = 30
# Setup directory for train/test data
train_path = '../../cifar_train100.json'
test_path = './data/test/cifar_test100.json'
public_path= './data/public/public_data100.json'
dir_path = os.path.dirname(train_path)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
dir_path = os.path.dirname(test_path)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
dir_path = os.path.dirname(public_path)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)


cifa_data_image = []
cifa_data_label = []

cifa_data_image.extend(trainset.data.cpu().detach().numpy())
cifa_data_image.extend(testset.data.cpu().detach().numpy())
cifa_data_label.extend(trainset.targets.cpu().detach().numpy())
cifa_data_label.extend(testset.targets.cpu().detach().numpy())
cifa_data_image = np.array(cifa_data_image)
cifa_data_label = np.array(cifa_data_label)

cifa_data = []
public_cifar_data=[]
for i in trange(100):
    idx = cifa_data_label == i
    # cifa_data.append(cifa_data_image[idx])
    cifa_data.append(cifa_data_image[idx][:int(len(cifa_data_image[idx])*0.93)])
    public_cifar_data.append(cifa_data_image[idx][int(len(cifa_data_image[idx]) * 0.93):])

print("\nNumb samples of each label:\n", [len(v) for v in cifa_data])
print("\nNumb samples of each label:\n", [len(v) for v in public_cifar_data])
users_lables = []

###### CREATE USER DATA SPLIT #######
# Assign 100 samples to each user
X = [[] for _ in range(NUM_USERS)]
y = [[] for _ in range(NUM_USERS)]
X_public = []
y_public = []
idx = np.zeros(100, dtype=np.int64)
for user in range(NUM_USERS):
    for j in range(NUM_LABELS):  # 30 labels for each users
        # l = (2*user+j)%10
        l = (user*10 + j) % 100
        print("L:", l)
        X[user] += cifa_data[l][idx[l]:idx[l] + 11].tolist()
        y[user] += (l * np.ones(11)).tolist()
        idx[l] += 10
    # print(len(X[user]))
    # print(len(y[user]))

print("IDX1:", idx)  # counting samples for each labels

# Assign remaining sample by power law
user = 0
props = np.random.lognormal(
    0, 2., (100, NUM_USERS, NUM_LABELS))  # last 5 is 5 labels
props = np.array([[[len(v) - NUM_USERS]] for v in cifa_data]) * \
        props / np.sum(props, (1, 2), keepdims=True)
# print("here:",props/np.sum(props,(1,2), keepdims=True))
# props = np.array([[[len(v)-100]] for v in mnist_data]) * \
#    props/np.sum(props, (1, 2), keepdims=True)
# idx = 1000*np.ones(10, dtype=np.int64)
# print("here2:",props)
for user in trange(NUM_USERS):
    for j in range(NUM_LABELS):  # 4 labels for each users
        # l = (2*user+j)%10
        l = (user*10 + j) % 100
        num_samples = int(props[l, user // int(NUM_USERS / 10), j])

        numran1 = random.randint(5, 50)
        numran2 = random.randint(1, 5)
        num_samples = (num_samples) * numran2 + numran1
        if (NUM_USERS <= 20):
            num_samples = num_samples * 2
        # print("num of samples", num_samples, " :", j)
        if idx[l] + num_samples < len(cifa_data[l]):
            X[user] += cifa_data[l][idx[l]:idx[l] + num_samples].tolist()
            y[user] += (l * np.ones(num_samples)).tolist()
            idx[l] += num_samples
            # print("check len os user:", user, j,
            #       "len data", len(X[user]), num_samples)



for k in trange(100):
    X_public += public_cifar_data[k].tolist()
    y_public += (k*np.ones(len(public_cifar_data[k]))).tolist()
    # print(len(public_mnist_data[k]))
print("length public",len(X_public))

print("IDX2:", idx)  # counting samples for each labels


# Create data structure
train_data = {'users': [], 'user_data': {}, 'num_samples': []}
test_data = {'users': [], 'user_data': {}, 'num_samples': []}
public_data = {'public_data':{},'num_samples_public':[]}
public_data["public_data"] = {'x': X_public, 'y': y_public}
public_data['num_samples_public'].append(len(y_public))
# Setup 5 users
# for i in trange(5, ncols=120):

for i in range(NUM_USERS):
    uname = 'f_{0:05d}'.format(i)

    # X_train, X_test, y_train, y_test = train_test_split(X[i], y[i], train_size=0.75, stratify=y[i])
    # print("length of X", len(X[i]))
    # print("length of Y", len(y[i]))
    X_train, X_test, y_train, y_test = train_test_split(X[i], y[i], train_size=0.8, stratify=y[i])

    train_data["user_data"][uname] = {'x': X_train, 'y': y_train}
    train_data['users'].append(uname)
    train_data['num_samples'].append(len(y_train))

    test_data['users'].append(uname)
    test_data["user_data"][uname] = {'x': X_test, 'y': y_test}
    test_data['num_samples'].append(len(y_test))

print("Num_samples:", train_data['num_samples'])
print("Total_samples:", sum(train_data['num_samples'] + test_data['num_samples']))

with open(train_path, 'w') as outfile:
    json.dump(train_data, outfile)
with open(test_path, 'w') as outfile:
    json.dump(test_data, outfile)
with open(public_path, 'w') as outfile:
    json.dump(public_data, outfile)
print("Finish Generating Samples")