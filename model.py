import tensorflow as tf

from config import Config as config

with tf.name_scope('input'):
    X = tf.placeholder(tf.float32, [None, config.IMAGE_HEIGHT * config.IMAGE_WIDTH])
    Y = tf.placeholder(tf.float32, [None, config.NUM_CLASSES])
keep_prob = tf.placeholder(tf.float32)  # dropout

def crack_captcha_cnn(w_alpha=0.01, b_alpha=0.1):
    x = tf.reshape(X, shape=[-1, config.IMAGE_HEIGHT, config.IMAGE_WIDTH, 1])

    # w_c1_alpha = np.sqrt(2.0/(IMAGE_HEIGHT*IMAGE_WIDTH)) #
    # w_c2_alpha = np.sqrt(2.0/(3*3*32))
    # w_c3_alpha = np.sqrt(2.0/(3*3*64))
    # w_d1_alpha = np.sqrt(2.0/(8*32*64))
    # out_alpha = np.sqrt(2.0/1024)

    # 3 conv layer
    w_c1 = tf.Variable(w_alpha * tf.random_normal([3, 3, 1, 32]))
    b_c1 = tf.Variable(b_alpha * tf.random_normal([32]))
    conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv1 = tf.nn.dropout(conv1, keep_prob)

    w_c2 = tf.Variable(w_alpha * tf.random_normal([3, 3, 32, 64]))
    b_c2 = tf.Variable(b_alpha * tf.random_normal([64]))
    conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv2 = tf.nn.dropout(conv2, keep_prob)

    w_c3 = tf.Variable(w_alpha * tf.random_normal([3, 3, 64, 64]))
    b_c3 = tf.Variable(b_alpha * tf.random_normal([64]))
    conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv3 = tf.nn.dropout(conv3, keep_prob)

    # Fully connected layer
    w_d = tf.Variable(w_alpha * tf.random_normal([8 * 20 * 64, 1024]))
    b_d = tf.Variable(b_alpha * tf.random_normal([1024]))
    dense = tf.reshape(conv3, [-1, w_d.get_shape().as_list()[0]])
    dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
    dense = tf.nn.dropout(dense, keep_prob)

    with tf.name_scope('w_out'):
        w_out = tf.Variable(w_alpha * tf.random_normal([1024, config.NUM_CLASSES]))

    with tf.name_scope('b_out'):
        b_out = tf.Variable(b_alpha * tf.random_normal([config.NUM_CLASSES]))
    out = tf.add(tf.matmul(dense, w_out), b_out)
    # out = tf.nn.softmax(out)
    return out

def cal_confidence(logits, ):
    logits = tf.reshape(logits, [-1, config.MAX_NUM_CHARS, config.CHAR_SET_LEN])
    confidence = tf.keras.layers.Softmax()(logits)
    min_confidence = tf.keras.layers.minimum(confidence)
    is_confident = tf.math.greater(min_confidence, 0.5)
    return is_confident

def cal_accuracy(logits):
    predict = tf.reshape(logits, [-1, config.MAX_NUM_CHARS, config.CHAR_SET_LEN])
    max_idx_p = tf.argmax(predict, 2)
    max_idx_l = tf.argmax(tf.reshape(Y, [-1, config.MAX_NUM_CHARS, config.CHAR_SET_LEN]), 2)
    correct_pred = tf.reduce_all(tf.equal(max_idx_p, max_idx_l), 1)

    softmax = tf.keras.layers.Softmax()(predict)
    max_softmax = tf.reduce_max(softmax, axis=-1)
    min_confidence = tf.reduce_min(max_softmax, axis=-1)
    is_confident = tf.math.greater(min_confidence, config.CONFIDENCE_THRESHOLD)
    num_correct_pred = tf.reduce_sum(tf.cast(correct_pred, tf.float32))
    num_greater_thre = tf.reduce_sum(tf.cast(is_confident, tf.float32))
    correct_pred_confi = tf.reduce_sum(tf.cast(tf.equal(is_confident, correct_pred), tf.float32))


    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

    with tf.name_scope('correct_and_confident'):
        correct_and_confident = correct_pred_confi / num_greater_thre
        tf.summary.scalar('correct_and_confident', correct_and_confident)

    return accuracy, correct_and_confident