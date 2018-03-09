#By Binbin Zhang
#bzhang25@buffalo.edu
#8/10/2017

#moog-binData-conv2-loc2-15-3-V1_data6


import os
import os.path
import math

import numpy as np
import tensorflow as tf

import read_input
import datetime
from sklearn.metrics import confusion_matrix
import scipy.io as scio
#%%

BATCH_SIZE = 50
LEARNING_RATE = [0.0001]
MAX_STEP = 5000

IMAGE_SIZE = 300
NUM_CLASSES = 3
n_test =3900 #3834 #CNN1-300: 1500

CLASSES_NAMES =['Undermelt','JustRight', 'Overmelt']
 

LOG_TRAIN_DIR = 'Result_log/train_stru-01'+ '_imsz-%d_bz-%d_t1-'%(IMAGE_SIZE,BATCH_SIZE)
LOG_VAL_DIR = 'Result_log/eval_stru-01'+ '_imsz-%d_bz-%d_t1-'%(IMAGE_SIZE,BATCH_SIZE)
data_dir = '../../Data/'


## Imagenet structure
# def inference(images):
#     '''
#     Args:
#         images: 4D tensor [batch_size, img_width, img_height, img_channel]
#     Notes:
#         In each conv layer, the kernel size is:
#         [kernel_size, kernel_size, number of input channels, number of output channels].
#         number of input channels are from previuous layer, if previous layer is THE input
#         layer, number of input channels should be image's channels.
#
#
#     '''
#     # conv1, [5, 5, 3, 96], The first two dimensions are the patch size,
#     # the next is the number of input channels,
#     # the last is the number of output channels
#     with tf.variable_scope('conv1') as scope:
#         weights = tf.get_variable('weights', shape=[11, 11, 3, 96], dtype=tf.float32,
#                                   initializer=tf.truncated_normal_initializer(stddev=0.5, dtype=tf.float32))
#         biases = tf.get_variable('biases', shape=[96], dtype=tf.float32, initializer=tf.constant_initializer(0.1))
#         conv = tf.nn.conv2d(images, weights, strides=[1, 4, 4, 1], padding='SAME')
#         pre_activation = tf.nn.bias_add(conv, biases)
#         conv1 = tf.nn.relu(pre_activation, name=scope.name)
#
#         tf.summary.histogram(scope.name + "/weights", weights)
#         tf.summary.histogram(scope.name + "/biases", biases)
#         tf.summary.histogram(scope.name + "/activations", conv1)
#
#
#
#     # pool1 and norm1
#     with tf.variable_scope('pooling1_lrn') as scope:
#         pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pooling1')
#         norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
#
#     # conv2
#     with tf.variable_scope('conv2') as scope:
#         weights = tf.get_variable('weights', shape=[5, 5, 96, 256], dtype=tf.float32,
#                                   initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
#         biases = tf.get_variable('biases', shape=[256], dtype=tf.float32, initializer=tf.constant_initializer(0.1))
#         conv = tf.nn.conv2d(norm1, weights, strides=[1, 1, 1, 1], padding='SAME')
#         pre_activation = tf.nn.bias_add(conv, biases)
#         conv2 = tf.nn.relu(pre_activation, name='conv2')
#         tf.summary.histogram(scope.name + "/weights", weights)
#         tf.summary.histogram(scope.name + "/biases", biases)
#         tf.summary.histogram(scope.name + "/activations", conv2)
#
#     # pool2 and norm2
#     with tf.variable_scope('pooling2_lrn') as scope:
#         #norm2 = tf.nn.lrn(conv2, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
#         pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pooling2')
#
#     # conv3
#     with tf.variable_scope('conv3') as scope:
#         weights = tf.get_variable('weights', shape=[3, 3, 256, 384], dtype=tf.float32,
#                                   initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
#         biases = tf.get_variable('biases', shape=[384], dtype=tf.float32, initializer=tf.constant_initializer(0.1))
#         conv = tf.nn.conv2d(pool2, weights, strides=[1, 1, 1, 1], padding='SAME')
#         pre_activation = tf.nn.bias_add(conv, biases)
#         conv3 = tf.nn.relu(pre_activation, name='conv3')
#         tf.summary.histogram(scope.name + "/weights", weights)
#         tf.summary.histogram(scope.name + "/biases", biases)
#         tf.summary.histogram(scope.name + "/activations", conv3)
#
#     # conv4
#     with tf.variable_scope('conv4') as scope:
#         weights = tf.get_variable('weights', shape=[3, 3, 384, 384], dtype=tf.float32,
#                                   initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
#         biases = tf.get_variable('biases', shape=[384], dtype=tf.float32, initializer=tf.constant_initializer(0.1))
#         conv = tf.nn.conv2d(conv3, weights, strides=[1, 1, 1, 1], padding='SAME')
#         pre_activation = tf.nn.bias_add(conv, biases)
#         conv4 = tf.nn.relu(pre_activation, name='conv4')
#
#     # conv5
#     with tf.variable_scope('conv5') as scope:
#         weights = tf.get_variable('weights', shape=[3, 3, 384, 256], dtype=tf.float32,
#                                   initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
#         biases = tf.get_variable('biases', shape=[256], dtype=tf.float32, initializer=tf.constant_initializer(0.1))
#         conv = tf.nn.conv2d(conv4, weights, strides=[1, 1, 1, 1], padding='SAME')
#         pre_activation = tf.nn.bias_add(conv, biases)
#         conv5 = tf.nn.relu(pre_activation, name='conv5')
#
#     # pool2 and norm2
#     with tf.variable_scope('pooling3_lrn') as scope:
#         #norm2 = tf.nn.lrn(conv2, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
#         pool3 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pooling3')
#
#
#     # local3
#     with tf.variable_scope('local3') as scope:
#         reshape = tf.reshape(pool3, shape=[BATCH_SIZE, -1])
#         dim = reshape.get_shape()[1].value
#         weights = tf.get_variable('weights', shape=[dim, 1024], dtype=tf.float32,
#                                   initializer=tf.truncated_normal_initializer(stddev=0.004, dtype=tf.float32))
#         biases = tf.get_variable('biases', shape=[1024], dtype=tf.float32, initializer=tf.constant_initializer(0.1))
#         local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
#
#     # local4
#     with tf.variable_scope('local4') as scope:
#         weights = tf.get_variable('weights', shape=[1024, 192], dtype=tf.float32,
#                                   initializer=tf.truncated_normal_initializer(stddev=0.004, dtype=tf.float32))
#         biases = tf.get_variable('biases', shape=[192], dtype=tf.float32, initializer=tf.constant_initializer(0.1))
#         local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name='local4')
#
#         tf.summary.histogram(scope.name + "/weights", weights)
#         tf.summary.histogram(scope.name + "/biases", biases)
#         tf.summary.histogram(scope.name + "/activations", local4)
#
#
#     # softmax
#     with tf.variable_scope('softmax_linear') as scope:
#         weights = tf.get_variable('softmax_linear', shape=[192, 3], dtype=tf.float32,
#                                   initializer=tf.truncated_normal_initializer(stddev=0.004, dtype=tf.float32))
#         biases = tf.get_variable('biases',shape=[3], dtype=tf.float32, initializer=tf.constant_initializer(0.1))
#         softmax_linear = tf.add(tf.matmul(local4, weights), biases, name='softmax_linear')
#
#     return softmax_linear

#cifar10 similar structure
def inference(images):
    '''
    Args:
        images: 4D tensor [batch_size, img_width, img_height, img_channel]
    Notes:
        In each conv layer, the kernel size is:
        [kernel_size, kernel_size, number of input channels, number of output channels].
        number of input channels are from previuous layer, if previous layer is THE input
        layer, number of input channels should be image's channels.


    '''
    # conv1, [5, 5, 3, 96], The first two dimensions are the patch size,
    # the next is the number of input channels,
    # the last is the number of output channels
    with tf.variable_scope('conv1') as scope:
        weights = tf.get_variable('weights', shape=[3, 3, 3, 32], dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.5, dtype=tf.float32))
        biases = tf.get_variable('biases', shape=[32], dtype=tf.float32, initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(images, weights, strides=[1, 2, 2, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)

        tf.summary.histogram(scope.name + "/weights", weights)
        tf.summary.histogram(scope.name + "/biases", biases)
        tf.summary.histogram(scope.name + "/activations", conv1)



    # pool1 and norm1
    with tf.variable_scope('pooling1_lrn') as scope:
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pooling1')
        norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

    # conv2
    with tf.variable_scope('conv2') as scope:
        weights = tf.get_variable('weights', shape=[3, 3, 32, 64], dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        biases = tf.get_variable('biases', shape=[64], dtype=tf.float32, initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(norm1, weights, strides=[1, 1, 1, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name='conv2')

    # pool2 and norm2
    with tf.variable_scope('pooling2_lrn') as scope:
        #norm2 = tf.nn.lrn(conv2, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME', name='pooling2')

    # local3
    with tf.variable_scope('local3') as scope:
        reshape = tf.reshape(pool2, shape=[BATCH_SIZE, -1])
        dim = reshape.get_shape()[1].value
        weights = tf.get_variable('weights', shape=[dim, 384], dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.004, dtype=tf.float32))
        biases = tf.get_variable('biases', shape=[384], dtype=tf.float32, initializer=tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

    # local4
    with tf.variable_scope('local4') as scope:
        weights = tf.get_variable('weights', shape=[384, 192], dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.004, dtype=tf.float32))
        biases = tf.get_variable('biases', shape=[192], dtype=tf.float32, initializer=tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name='local4')

        tf.summary.histogram(scope.name + "/weights", weights)
        tf.summary.histogram(scope.name + "/biases", biases)
        tf.summary.histogram(scope.name + "/activations", local4)


    # softmax
    with tf.variable_scope('softmax_linear') as scope:
        weights = tf.get_variable('softmax_linear', shape=[192, 3], dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.004, dtype=tf.float32))
        biases = tf.get_variable('biases',shape=[3], dtype=tf.float32, initializer=tf.constant_initializer(0.1))
        softmax_linear = tf.add(tf.matmul(local4, weights), biases, name='softmax_linear')

    return softmax_linear

# %%

def losses(logits, labels):
    with tf.variable_scope('loss') as scope:
        labels = tf.cast(labels, tf.int64)

        # to use this loss fuction, one-hot encoding is needed!
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits \
            (logits=logits, labels=labels, name='xentropy_per_example')

        #        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits\
        #                        (logits=logits, labels=labels, name='xentropy_per_example')

        loss = tf.reduce_mean(cross_entropy, name='loss')
        tf.summary.scalar(scope.name + '/loss', loss)

    return loss


# %% Train the model on the training data
# you need to change the training data directory below

def evaluation(logits, labels):
    with tf.variable_scope('accuracy') as scope:
        labels = tf.argmax(labels, 1)  # one_hot decode
        correct = tf.nn.in_top_k(logits, labels, 1)
        correct = tf.cast(correct, tf.float16)
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar(scope.name + '/accuracy', accuracy)
    return accuracy


def plot_confusion_matrix(cls_test,cls_pred):
    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # Get the confusion matrix using sklearn.
    # cm = confusion_matrix(y_true=cls_test,  # True class for test-set.
    #                       y_pred=cls_pred)  # Predicted class.

    cm = confusion_matrix(y_true=cls_test,  # True class for test-set.
                          y_pred=cls_pred)  # Predicted class.

    # Print the confusion matrix as text.
    for i in range(NUM_CLASSES):
        # Append the class-name to each line.
        class_name = "({}) {}".format(i, CLASSES_NAMES[i])
        print(cm[i, :], class_name)

    # Print the class-numbers for easy reference.
    class_numbers = [" ({0})".format(i) for i in range(NUM_CLASSES)]
    print("".join(class_numbers))

def train_n_eval():

    val_dir = data_dir

    images, labels = read_input.read_data(data_dir=data_dir,
                                                is_train=True,
                                                batch_size=BATCH_SIZE,
                                                shuffle=True)
    images_val, labels_val = read_input.read_data(data_dir=val_dir,
                                                is_train=False,
                                                batch_size=BATCH_SIZE,
                                                shuffle=False)

    logits = inference(images)

    loss = losses(logits, labels)


    for LEARNING_RT in LEARNING_RATE:
        print('*****Starting run for lLEARNING_RATE = %.0E*****' %LEARNING_RT)
        log_train_dir = LOG_TRAIN_DIR + '_lr_%.0E'%(LEARNING_RT)
        log_val_dir = LOG_VAL_DIR +'_lr_%.0E'%(LEARNING_RT)
        checkpoint_dir = log_train_dir

        starttime = datetime.datetime.now()

        optimizer = tf.train.AdamOptimizer(LEARNING_RT)

        global_step = tf.Variable(initial_value=0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)

        accuracy = evaluation(logits, labels)

        x=tf.placeholder(tf.float32,shape=[BATCH_SIZE,IMAGE_SIZE, IMAGE_SIZE, 3], name='x')
        y_=tf.placeholder(tf.int32, shape=[BATCH_SIZE,NUM_CLASSES], name='y_true')
        y_true_cls = tf.argmax(y_, dimension=1)


        saver = tf.train.Saver(tf.global_variables())
        summary_op = tf.summary.merge_all()

        sess = tf.Session()

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        train_writer = tf.summary.FileWriter(log_train_dir, sess.graph)
        val_writer=tf.summary.FileWriter(log_val_dir, sess.graph)

        try:
            print("Trying to restore last checkpoint ...")

            # Use TensorFlow to find the latest checkpoint - if any.
            last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=checkpoint_dir)

            # Try and load the data in the checkpoint.
            saver.restore(sess, save_path=last_chk_path)

            # If we get to this point, the checkpoint was successfully loaded.
            print("Restored checkpoint from:", last_chk_path)

        except:
            # If the above failed for some reason, simply
            # initialize all the variables for the TensorFlow graph.
            print("Failed to restore checkpoint. Initializing variables instead.")
            #session.run(tf.global_variables_initializer())
            init = tf.global_variables_initializer()
            sess.run(init)


        try:
            for step in np.arange(MAX_STEP):
                if coord.should_stop():
                    break
                tra_image,tra_label=sess.run([images,labels])

                global_step_i= sess.run(global_step)

                _, tra_loss,tra_acc = sess.run([train_op, loss, accuracy],
                                                 feed_dict={x:tra_image,y_:tra_label})
                # _, loss_value = sess.run([train_op, loss])


                if global_step_i % 50 == 0 or step == 0:
                    print('Step: %d, loss: %.4f,train accuracy = %.2f%%' %(global_step_i, tra_loss,tra_acc * 100.0))
                    summary_str = sess.run(summary_op)
                    train_writer.add_summary(summary_str, global_step_i)


                if global_step_i % 100 == 0 or (step + 1) == MAX_STEP:
                    val_image,val_label = sess.run([images_val,labels_val])
                    val_loss,val_acc = sess.run([loss,accuracy],
                                                feed_dict={x:val_image,y_:val_label})
                    print('<  Step: %d, valid loss: %.4f,valid accuracy = %.2f%%  >' % (global_step_i, val_loss, val_acc * 100.0))
                    summary_str = sess.run(summary_op)
                    val_writer.add_summary(summary_str, global_step_i)

                if global_step_i % 1000 == 0 or (step + 1) == MAX_STEP:
                    checkpoint_path = os.path.join(log_train_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=global_step)



        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            coord.request_stop()

        coord.join(threads)
        sess.close()

        endtime = datetime.datetime.now()
        print("Time usage: " + str(endtime - starttime))


def evaluate_once():
    with tf.Graph().as_default():


        val_dir = data_dir
        for LEARNING_RT in LEARNING_RATE:
            log_train_dir = LOG_TRAIN_DIR + '_lr_%.0E'%(LEARNING_RT)
            checkpoint_dir = log_train_dir
            cls_pred = []
            cls_true = []

            # reading test data
            images, labels = read_input.read_data(data_dir=val_dir,
                                                        is_train=False,
                                                        batch_size=BATCH_SIZE,
                                                        shuffle=False)

            logits = inference(images)
            y_pred_cls = tf.argmax(logits, dimension=1)
            labels = tf.argmax(labels, 1)  # one_hot decode
            top_k_op = tf.nn.in_top_k(logits, labels, 1)
            saver = tf.train.Saver(tf.global_variables())

            with tf.Session() as sess:

                print("Reading checkpoints...")
                ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    print('Loading success, global_step is %s' % global_step)
                else:
                    print('No checkpoint file found')
                    #return

                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=sess, coord=coord)

                try:
                    num_iter = int(math.ceil(n_test / BATCH_SIZE))
                    true_count = 0
                    total_sample_count = num_iter * BATCH_SIZE
                    step = 0

                    predict_label_all = np.zeros((1, 40))
                    true_label_all = np.zeros((1, 40))

                    while step < num_iter and not coord.should_stop():
                        y_pred, y_true, predictions = sess.run([y_pred_cls,labels,top_k_op])
                        # print("step:", step)
                        # print("y_pred: ", y_pred)
                        # print("y_true: ", y_true)


                        true_count += np.sum(predictions)
                        step += 1
                        precision = true_count / total_sample_count

                        cls_pred = np.append(cls_pred,y_pred)
                        cls_true = np.append(cls_true, y_true)


                    print('precision = %.3f' % precision)
                    print('Confusion Matrix:')
                    plot_confusion_matrix(cls_true, cls_pred)
                    return np.reshape(cls_pred,(3900,1)),np.reshape(cls_true,(3900,1))
                except Exception as e:
                    coord.request_stop(e)
                finally:
                    coord.request_stop()
                    coord.join(threads)


# import scipy.io as scio
# def confusion_matrix():
#     with tf.Graph().as_default():
#
#
#         val_dir = data_dir
#         test_dir = val_dir
#        # n_test = 21600
#         for LEARNING_RT in LEARNING_RATE:
#             log_train_dir = LOG_TRAIN_DIR + '_lr_%.0E'%(LEARNING_RT)
#             log_dir = log_train_dir
#
#             # reading test data
#             images, labels = read_input.read_data(data_dir=test_dir,
#                                                         is_train=False,
#                                                         batch_size=BATCH_SIZE,
#                                                         shuffle=False)
#
#             logits = inference(images)
#             labels = tf.argmax(labels, 1) # one_hot decode
#             value,id=tf.nn.top_k(logits)
#             predict=id
#             # top_k_op = tf.nn.in_top_k(logits, labels, 1)
#             saver = tf.train.Saver(tf.global_variables())
#
#             with tf.Session() as sess:
#
#                 print("Reading checkpoints...")
#                 ckpt = tf.train.get_checkpoint_state(log_dir)
#                 if ckpt and ckpt.model_checkpoint_path:
#                     global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
#                     saver.restore(sess, ckpt.model_checkpoint_path)
#                     print('Loading success, global_step is %s' % global_step)
#                 else:
#                     print('No checkpoint file found')
#                     return
#
#                 coord = tf.train.Coordinator()
#                 threads = tf.train.start_queue_runners(sess=sess, coord=coord)
#
#                 try:
#                     num_iter = int(math.ceil(n_test / BATCH_SIZE))
#                     print(num_iter)
#                     true_count = 0
#                     total_sample_count = num_iter * BATCH_SIZE
#                     step = 0
#                     predict_label_all=np.zeros((BATCH_SIZE,1))
#                     true_label_all=np.zeros((1,BATCH_SIZE))
#                     while step < num_iter and not coord.should_stop():
#                         predict_label,true_label=sess.run([predict,labels])
#                         predict_label_all=np.hstack((predict_label_all,predict_label))
#                         true_label_all=np.vstack((true_label_all,true_label))
#                         # print(predict_label)
#                         print(true_label)
#                         # predictions = sess.run([top_k_op])
#                         # true_count += np.sum(predictions)
#                         step += 1
#                         # precision = true_count / total_sample_count
#                     # print('precision = %.3f' % precision)
#                     return np.reshape(np.transpose(predict_label_all[:,1:]),(1,n_test)),np.reshape(true_label_all[1:,:],(1,n_test))
#                 except Exception as e:
#                     coord.request_stop(e)
#                 finally:
#                     coord.request_stop()
#                     coord.join(threads)


#
# #
starttime1 = datetime.datetime.now()
print('*****Training**************************************')
train_n_eval()
endtime1 = datetime.datetime.now()
print('Training total time: ',endtime1-starttime1)
print('*****Evaluation************************************')
print("Check after training:")
# evaluate_once()

starttime2 = datetime.datetime.now()
#
# predict_all,true_all=confusion_matrix()
# save_dir = 'C:\\Users\\bzhang25\\Google Drive\\CNN-testing\\aa'
# scio.savemat(save_dir, {'predict':predict_all,'true':true_all})
predict_all,true_all=evaluate_once()
save_dir = 'Result/test_result_Stru-01'
scio.savemat(save_dir, {'predict':predict_all,'true':true_all})
endtime2 = datetime.datetime.now()
print('Testing total time: ',endtime2-starttime2)



