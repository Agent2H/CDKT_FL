# from sklearn.datasets import fetch_mldata
from tqdm import trange
import numpy as np
import random
import json
import os
from tensorflow.examples.tutorials.mnist import input_data

random.seed(1)
np.random.seed(1)
NUM_USERS = 100
NUM_LABELS = 2

# Setup directory for train/test data
train_path = './data/train/fashion_train.json'
test_path = './data/test/fashion_test.json'
public_path= './data/public/public_data.json'
dir_path = os.path.dirname(train_path)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
dir_path = os.path.dirname(test_path)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

# Import Fashion MNIST
fashion_data = input_data.read_data_sets(
    'data/fashion', source_url='http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/')
# fashion_data = tfds.load(name="fashion_mnist")

print("hello")
#fashion_full = list(zip(fashion_data.train, fashion_data.test))
print(type(fashion_data))
mnist_data_image = []
mnist_data_lable = []
mnist_data_image.extend(fashion_data.train.images)
mnist_data_image.extend(fashion_data.test.images)
mnist_data_lable.extend(fashion_data.train.labels)
mnist_data_lable.extend(fashion_data.test.labels)

mu = np.mean(mnist_data_image)
sigma = np.std(mnist_data_image)
nom_Fashion_data = (mnist_data_image - mu)/(sigma+0.001)
nom_Fashion_lable = np.array(mnist_data_lable)
mnist_data = []
public_fmnist_data = []
for i in trange(10):
    idx = nom_Fashion_lable == i
    # mnist_data.append(nom_Fashion_data[idx])
    mnist_data.append(nom_Fashion_data[idx][:int(len(nom_Fashion_data[idx]) * 0.995)])
    public_fmnist_data.append(nom_Fashion_data[idx][int(len(nom_Fashion_data[idx]) * 0.995):])

print([len(v) for v in mnist_data])
print([len(v) for v in public_fmnist_data])

X_public = []
y_public = []
###### CREATE USER DATA SPLIT #######
# Assign 100 samples to each user
X = [[] for _ in range(NUM_USERS)]
y = [[] for _ in range(NUM_USERS)]
idx = np.zeros(10, dtype=np.int64)
for user in range(NUM_USERS):
    for j in range(NUM_LABELS):  # 3 labels for each users
        # l = (2*user+j)%10
        l = (user + j) % 10
        # print("L:", l)
        X[user] += mnist_data[l][idx[l]:idx[l]+10].tolist()
        y[user] += (l*np.ones(10)).tolist()
        idx[l] += 10

print("IDX1:", idx)  # counting samples for each labels

# Assign remaining sample by power law
user = 0
props = np.random.lognormal(
    0, 1., (10, NUM_USERS, NUM_LABELS))  # last 5 is 5 labels
# print("here:",props/np.sum(props,(1,2), keepdims=True))
props = np.array([[[len(v)-100]] for v in mnist_data]) * \
    props/np.sum(props, (1, 2), keepdims=True)
#idx = 1000*np.ones(10, dtype=np.int64)
# print("here2:",props)
for user in trange(NUM_USERS):
    for j in range(NUM_LABELS):  # 2 labels for each users
        # l = (2*user+j)%10
        l = (user + j) % 10
        # #num_samples = int(props[l,user//int(NUM_USERS/10),j]) *10
        # num_samples = int(props[l, user, j]) * NUM_USERS * 2

        num_samples = int(props[l, user//int(NUM_USERS/10), j])
        num_samples = int(num_samples * 0.55)  # Scale down number of samples

        ## num_samples = min(num_samples,200)
        # print(num_samples)
        if idx[l] + num_samples < len(mnist_data[l]):
            X[user] += mnist_data[l][idx[l]:idx[l]+num_samples].tolist()
            y[user] += (l*np.ones(num_samples)).tolist()
            idx[l] += num_samples


for k in trange(10):
    X_public += public_fmnist_data[k].tolist()
    y_public += (k*np.ones(len(public_fmnist_data[k]))).tolist()
    # print(len(public_mnist_data[k]))
print("length",len(X_public))
print("length",len(y_public))

print("IDX2:", idx)  # counting samples for each labels
# Create data structure
train_data = {'users': [], 'user_data': {}, 'num_samples': []}
test_data = {'users': [], 'user_data': {}, 'num_samples': []}
public_data = {'public_data':{},'num_samples_public':[]}
all_samples=[]
public_data["public_data"] = {'x': X_public, 'y': y_public}
public_data['num_samples_public'].append(len(y_public))
# Setup 5 users
# for i in trange(5, ncols=120):
for i in range(NUM_USERS):
    uname = 'f_{0:05d}'.format(i)

    combined = list(zip(X[i], y[i]))
    random.shuffle(combined)
    X[i][:], y[i][:] = zip(*combined)
    num_samples = len(X[i])
    train_len = int(0.8*num_samples)
    test_len = num_samples - train_len

    train_data['users'].append(uname)
    train_data['user_data'][uname] = {
        'x': X[i][:train_len], 'y': y[i][:train_len]}
    train_data['num_samples'].append(train_len)
    test_data['users'].append(uname)
    test_data['user_data'][uname] = {
        'x': X[i][train_len:], 'y': y[i][train_len:]}
    test_data['num_samples'].append(test_len)
    all_samples.append(train_len + test_len)

print("Numb_training_samples:", train_data['num_samples'])
print("Total_training_samples:",sum(train_data['num_samples']))
print("Numb_testing_samples:", test_data['num_samples'])
print("Total_testing_samples:",sum(test_data['num_samples']))
print("Median of data samples:", np.median(all_samples))

with open(train_path, 'w') as outfile:
    json.dump(train_data, outfile)
with open(test_path, 'w') as outfile:
    json.dump(test_data, outfile)
with open(public_path, 'w') as outfile:
    json.dump(public_data, outfile)
print("Finish Generating Samples")
