import tensorflow as tf

for e in tf.compat.v1.train.summary_iterator("/home/nlp/amit/sign-language/transcription/pose_to_video/pix_to_pix/logs/fit/20231118-221156/events.out.tfevents.1700345516.9ca74b8dca9d.1.0.v2"):
    for v in e.summary.value:
        print(v.tag)
        print(v.simple_value)
    print('-' * 10)