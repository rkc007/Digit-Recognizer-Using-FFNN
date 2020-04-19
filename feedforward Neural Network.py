'''
Kaggle Competition:
Digit Recognizer - Learn Computer Vision Fundamentals with the famous MNIST Data
Jingyi Luo
'''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
#import time
import math
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split

os.chdir("/Users/ljyi/Desktop/SYS6016/homework/homework02")

tf.reset_default_graph()

n_inputs = 28*28  # MNIST 784
n_hidden1 = 300   # 300
n_hidden2 = 300   # 100
n_hidden3 = 200
n_hidden4 = 100
n_hidden5 = 100
n_outputs = 10
learning_rate = 0.010 # 0.01
n_epochs = 40 # 40
batch_size = 70 #50

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int32, shape=(None), name="y")
training = tf.placeholder_with_default(False, shape=[], name='training')

#tf.set_random_seed(888)

# ---------------------- Computation Graphs Definition ------------------------
scale = 0.01
keep_prob = 0.8
#kernel_regularizer = tf.contrib.layers.l2_regularizer(scale)
#kernel_regularizer = tf.contrib.layers.l1_regularizer(scale)
with tf.name_scope("dnn"):    # a context manager. "dnn" the name argument
    hidden1 = tf.layers.dense(X, n_hidden1, name="hidden1", activation=tf.nn.relu)    # , kernel_regularizer = kernel_regularizer
#    hidden1_drop = tf.layers.dropout(hidden1, keep_prob, training=training)

    hidden2 = tf.layers.dense(hidden1, n_hidden2, name="hidden2", activation=tf.nn.relu)
#    hidden2_drop = tf.layers.dropout(hidden1_drop, dropout_rate)

    hidden3 = tf.layers.dense(hidden2, n_hidden3, name="hidden3", activation=tf.nn.relu)
#    hidden3_drop = tf.layers.dropout(hidden3, dropout_rate)

    hidden4 = tf.layers.dense(hidden3, n_hidden4, name="hidden4", activation=tf.nn.relu)
#    hidden4_drop = tf.layers.dropout(hidden4, dropout_rate)

#    hidden5 = tf.layers.dense(hidden4, n_hidden5, name="hidden5", activation=tf.nn.relu)
#    hidden5_drop = tf.layers.dropout(hidden5, dropout_rate)

    logits = tf.layers.dense(hidden4, n_outputs, name="logits") # , kernel_regularizer = kernel_regularizer

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits) # y here has one value, it's true label
    loss = tf.reduce_mean(xentropy, name="loss")
    loss_summary = tf.summary.scalar('log_loss', loss)  # output a single scalar value.

# loss context for l1,l2 regularization
#with tf.name_scope("loss"):
#    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits) # y here has one value, it's true label
#    base_loss = tf.reduce_mean(xentropy, name="base_loss")
#    reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
#    loss = tf.add_n([base_loss]+reg_loss, name='loss')
#    loss_summary = tf.summary.scalar('log_loss', loss)

with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)     #logits: predictions, y:targets, k=1. Batch sized.
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    accuracy_summary = tf.summary.scalar('accuracy', accuracy)

# initialize variables and save all variables from training
init = tf.global_variables_initializer()
saver = tf.train.Saver() # save and restore variables.

# ---------------------- Read in data, batch function -------------------------
# read in data
train_r = pd.read_csv('data/train.csv')   # 42000*785    # train_r: train_raw
# train_r.info()
# train_r.isnull().sum().sum()

y_train_r = np.array(train_r['label'])
x_train_r = np.array(train_r.drop(columns=['label'])) # 4200*784
test = pd.read_csv('data/test.csv')                   # 2800*784

# split and scale data
x_train, x_val, y_train, y_val = train_test_split(x_train_r, y_train_r, test_size = 0.1, random_state=8)
x_train = x_train/255   # 37800*784
x_val = x_val/255       # 4200*784
test = test/255         # 28000*784

# function to get bactches
def random_batches(x, y, batch_size, seed=8):
    m = x.shape[0]
    batches = []
    np.random.seed(seed)
    permutation = list(np.random.permutation(m)) # shuffling all the rows
    shuffled_x = x[permutation, :]
    shuffled_y = y[permutation]
    num_batches = math.floor(m/batch_size)
    for k in range(0, num_batches):
        batch_x = shuffled_x[k*batch_size:k*batch_size+batch_size, :]
        batch_y = shuffled_y[k*batch_size:k*batch_size+batch_size]
        mini_batch = (batch_x, batch_y)
        batches.append(mini_batch)

    # the left records to form a incomplete batch by their own
#   if m% batch_size != 0:
#       batch_x = shuffled_x[num_batches*batch_size:m, :]
#       batch_y = shuffled_y[num_batches*batch_size:m, :]
#       mini_batch = (batch_x, batch_y)
#       batches.append(mini_batch)
    return batches

# ----------combine summary nodes ------------
# combine all summary nodes to a single op to generates all the summary data
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('./graphs/train_optimize', tf.get_default_graph())
test_writer = tf.summary.FileWriter('./graphs/test_optimize', tf.get_default_graph())

# ------------------------- Model training ------------------------------------
with tf.Session() as sess:
    init.run()
#    i=0
    for epoch in range(n_epochs):
        all_batches = random_batches(x_train, y_train, batch_size)
        for a_batch in all_batches:
            (X_batch, y_batch) = a_batch
            sess.run(training_op, feed_dict = {X:X_batch, y:y_batch})  # , training: True
        acc_train = accuracy.eval(feed_dict = {X:X_batch, y:y_batch})
        acc_val = accuracy.eval(feed_dict= {X: x_val, y: y_val}) # ,
        print(epoch, "Train accuracy:", acc_train, "Val accuracy:", acc_val)

        # measure validation accuracy, and write validate summaries to FileWriters
        test_summary, acc = sess.run([merged, accuracy], feed_dict={X: x_val, y: y_val})
        test_writer.add_summary(test_summary, epoch)
        print('Accuracy at step %s: %s' % (epoch, acc))

        # run training_op on training data, and add training summaries to FileWriters
        train_summary, _ = sess.run([merged, training_op], feed_dict={X:X_batch, y:y_batch})
        train_writer.add_summary(train_summary, epoch)

    train_writer.close()
    test_writer.close()
    save_path = saver.save(sess, "./model_optimization/my_model_final.ckpt")   #save the whole session

# write computation graph to tensorboard
writer = tf.summary.FileWriter('./graphs/MNIST', tf.get_default_graph())
writer.add_graph(sess.graph)

# ------------------------ Test set prediction --------------------------------
with tf.Session() as sess:
    saver.restore(sess, "./model_optimization/my_model_final.ckpt")
    Z = logits.eval(feed_dict = {X: test})
    y_pred = np.argmax(Z, axis = 1)
print("Predicted classes:", y_pred)

# write to a dataframe to upload to kaggle
output = pd.DataFrame(y_pred, columns=['Label'])
output['ImageId'] = range(0, len(test))

# swap two columns
columnsTitles=["ImageId","Label"]
output=output.reindex(columns=columnsTitles)

# write to csv
output.to_csv("prediction_kaggle.csv", index=False)
