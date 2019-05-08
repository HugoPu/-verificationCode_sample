import os
import time
import tensorflow as tf

from config import Config as config
from utils.data_utils import get_next_batch
from model import crack_captcha_cnn, X, Y, keep_prob, cal_accuracy, cal_confidence

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def train_crack_captcha_cnn():

    logits = crack_captcha_cnn()
    # loss
    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output, Y))
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=Y))
        tf.summary.scalar('loss', loss)  # 可视化loss常量

    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(0.001, global_step, 400, 0.99, staircase=True)
    # optimizer 为了加快训练 learning_rate应该开始大，然后慢慢衰
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)

    accuracy, confident_scale, correct_scale = cal_accuracy(logits)

    saver = tf.train.Saver()
    run_config = tf.ConfigProto(log_device_placement=True)
    run_config.gpu_options.per_process_gpu_memory_fraction = 0.4
    with tf.Session(config=run_config) as sess:

        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter(config.OUTPUT, sess.graph)

        sess.run(tf.global_variables_initializer())

        step = 0
        max_acc = 0
        while True:
            batch_x, batch_y = get_next_batch(64, config, is_training=True)
            _, loss_ = sess.run([optimizer, loss], feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.75})
            print(step, loss_)

            # writer.add_summary(summary,step)
            # 每100 step计算一次准确率
            if step % 100 == 0:
                batch_x_test, batch_y_test = get_next_batch(100, config, is_training=False)
                summary, acc, confi_scale, cor_scale = sess.run(
                    [merged, accuracy, confident_scale, correct_scale],
                    feed_dict={X: batch_x_test, Y: batch_y_test, keep_prob: 1.})
                print(step, acc, confi_scale, cor_scale)

                writer.add_summary(summary, step)

                if max_acc < acc:
                    saver.save(sess, config.OUTPUT + "/model", global_step=step)
                    max_acc = acc

                # 如果准确率大于50%,保存模型,完成训练
                if step == config.NUM_EPOCH:
                    break

            step += 1



if __name__ == '__main__':
    start = time.clock()

    train_crack_captcha_cnn()

    end = time.clock()
    print('Running time: %s Seconds' % (end - start))
