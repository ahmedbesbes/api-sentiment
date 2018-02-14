import tensorflow as tf
from mem_absa.config_mem import Configure
from mem_absa.load_data import read_data, init_word_embeddings, read_raw
from mem_absa.load_data import read_vocabulary
from mem_absa.model import MemN2N

import spacy

fr_nlp=spacy.load("fr")
path=".."
configure=Configure()
FLAGS=configure.get_flags(path)

from pyfasttext import FastText

wiki_model=FastText()
wiki_model.load_model(FLAGS.pathFasttext)


def main(_):
    configure.pp.pprint(FLAGS.__flags)
    source_count=[]
    source_word2idx={}
    max_sent_length=read_vocabulary(fr_nlp, FLAGS.train_data, source_count, source_word2idx)
    print(max_sent_length)

    print('loading pre-trained word vectors...')
    FLAGS.pre_trained_context_wt=init_word_embeddings(wiki_model, source_word2idx, FLAGS.nbwords)
    # pad idx has to be 0
    FLAGS.pre_trained_context_wt[FLAGS.pad_idx, :]=0
    # FLAGS.pre_trained_target_wt = init_word_embeddings(target_word2idx)


    # with tf.Session() as sess:
    model=MemN2N(FLAGS)
    model.build_model()
    saver=tf.train.Saver(tf.trainable_variables())

    if FLAGS.load_model:
        print('Loading Model...')

        ckpt=tf.train.get_checkpoint_state(FLAGS.pathModel)
        saver.restore(model.sess, ckpt.model_checkpoint_path)
        print("Model loaded")

        if FLAGS.txt_file:
            test_data=read_data(fr_nlp, FLAGS.test_data, source_count, source_word2idx)
        else:
            test_data=read_raw(fr_nlp, FLAGS.test_data, source_count, source_word2idx)

        f=open("./word_id.txt", "w")
        for c, v in source_word2idx.items():
            f.write(str(c) + " " + str(v) + "\n")

        # read_sample
        FLAGS.pre_trained_context_wt=init_word_embeddings(wiki_model, source_word2idx, FLAGS.nbwords)
        FLAGS.pre_trained_context_wt[FLAGS.pad_idx, :]=0
        # model.predict(test_data, source_word2idx)
        test_loss, test_acc=model.test(test_data, source_word2idx, True)
        print('test_loss=%.4f;test-acc=%.4f;' % (test_loss, test_acc))

    else:
        print('training...')

        train_data=read_data(fr_nlp, FLAGS.train_data, source_count, source_word2idx)

        if FLAGS.txt_file:
            test_data=read_data(fr_nlp, FLAGS.test_data, source_count, source_word2idx)
        else:
            test_data=read_raw(fr_nlp, FLAGS.test_data, source_count, source_word2idx)

        FLAGS.pre_trained_context_wt=init_word_embeddings(wiki_model, source_word2idx, FLAGS.nbwords)
        FLAGS.pre_trained_context_wt[FLAGS.pad_idx, :]=0

        model.sess.run(model.A.assign(model.pre_trained_context_wt))
        model.sess.run(model.B.assign(model.pre_trained_context_wt))
        model.sess.run(model.ASP.assign(model.pre_trained_context_wt))

        max_train=0
        for idx in range(FLAGS.nepoch):
            print('epoch ' + str(idx) + '...')
            train_loss, train_acc=model.train(train_data, source_word2idx)
            print('train-loss=%.4f;train-acc=%.4f' % (train_loss, train_acc))
            test_loss, test_acc=model.test(test_data, source_word2idx, False)
            print('test-loss=%.4f;test-acc=%.4f' % (test_loss, test_acc))
            # save_path = saver.save(self.sess, "./models/model.ckpt", global_step=idx)
            # print("Model saved in file: %s" % save_path)
            # save the best model
            if max_train < train_acc:
                max_train=train_acc
                saver.save(model.sess, FLAGS.pathModel + '/model')


if __name__ == '__main__':
    tf.app.run()
