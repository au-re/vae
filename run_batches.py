import os
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time
from vae import VAE

'''


'''

# Load MNIST data in a format suited for tensorflow
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
# -------------------------------------------------

# TODO, extract from dataset
nsamples = mnist.train.num_examples
data_dim = 784
dataset_name = 'MNIST'

batch_size          = 100
epochs              = 75
network_structure   = [data_dim, 500, 500, 20]

vae = VAE(network_structure, learning_rate=0.001, batch_size=batch_size)

final_cost = 0.

# training cycle
start = time.time()
for epoch in range(epochs):
    avg_cost = 0.
    total_batch = int(nsamples/batch_size)
    
    for i in range(total_batch):
        sample, _ = mnist.train.next_batch(batch_size)
        cost, label = vae.fit(sample)
        avg_cost += cost / nsamples * batch_size
        final_cost = avg_cost
    
    print 'Dataset', dataset_name, 'Epoch:', '%04d' % (epoch+1), 'Cost:', '{:.9f}'.format(avg_cost)

end = time.time()
runtime = '{:.3f}'.format(end-start) 

print 'Total runtime over the training set:', runtime

# reconstruction
samples, classes = mnist.test.next_batch(batch_size)
reconstructions, labels = vae.reconstruct(samples)

# plot a reconstruction sample
plt.figure(figsize=(8, 12))
title = "Average Cost " + '{:.3f}'.format(final_cost) + " Running Time: " + runtime
plt.suptitle(title)

for i in range(6):
    ax = plt.subplot(6, 2, 2*i + 1)
    cax = ax.imshow(samples[i].reshape(28, 28), vmin=0, vmax=1)
    title = "Input"
    ax.set_title(title)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.colorbar(cax)
    
    ax1 = plt.subplot(6, 2, 2*i + 2)
    cax1 = ax1.imshow(reconstructions[i].reshape(28, 28), vmin=0, vmax=1)
    title = "Reconstruction"
    ax1.set_title(title)
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    plt.colorbar(cax1)

name = '_'.join(map(str, network_structure))
name = dataset_name + '_' + name

# save the file 
if not os.path.exists('img/' + dataset_name):
            os.makedirs('img/' + dataset_name)

plt.savefig('img/' + name +'_batch_'+str(batch_size)+'_epoch_'+str(epochs)+'.png')
plt.close()