import tensorflow as tf
a = tf.constant([[[1,2],[3,4]],[[5,6],[7,8]]])
test1 = tf.reduce_sum(a,axis=0)
test2 = tf.reduce_sum(a,axis=1)
test3 = tf.reduce_sum(a,axis=2)
with tf.Session() as sess:
    print(sess.run(test1))
    print('/r               ')
    print(sess.run(test2))
    print('/r               ')
    print(sess.run(test3))