import pprint
import tensorflow as tf


class Configure():
    def __init__(self):
        self.pp=pprint.PrettyPrinter()

    def get_flags(self, path):
        # path='..'
        flags=tf.app.flags
        self.FLAGS=flags.FLAGS
        # flags.DEFINE_string("pretrain_file", path+"/mem_absa/data/glove.6B.300d.txt", "pre-trained glove vectors file path ["+path+"/mem_absa/data/glove.6B.300d.txt]")
        # flags.DEFINE_string("pretrain_file", path+"/mem_absa/data/model.bin", "pre-trained glove vectors file path ["+path+"/mem_absa/data/model.bin]")

        #flags.DEFINE_string("train_data", path + "/mem_absa/data/Restaurants_Train-fr.xml",
        #                         "train gold data set path [" + path + "/mem_absa/data/Restaurants_Train-fr.xml]")

        self.FLAGS.test_data=path + "/mem_absa/data/Restaurants_Gold-fr.xml"
        # flags.DEFINE_string("test_data", path+"/mem_absa/data/Museums_Gold-fr.xml",
        #                    "test gold data set path ["+path+"/mem_absa/data/Museums_Gold-fr.xml]")
        # flags.DEFINE_string("test_data", path+"/data/pj_sent.raw",
        #                    "test gold data set path ["+path+"/data/pj_sent.raw]")

        self.FLAGS.test_samples=path + "/sequence_tagging/data/reviews.txt"

        self.FLAGS.test_aspects=path + "/sequence_tagging/data/aspects.txt"

        # flags.DEFINE_string("test_data", "data/Museums_gold-fr.xml", "test gold data set path [./data/Museums_gold-fr.xml]")

        #self.FLAGS.pathFasttext=path + "/data/model_pyfasttext100.bin"

        self.FLAGS.pathModel=path + "/mem_absa/models/test_model"

        self.FLAGS.show=False

        self.FLAGS.load_model=False

        self.FLAGS.load_samples=False

        self.FLAGS.txt_file=True


        # FLAGS.pad_idx = source_word2idx['<pad>']
        self.FLAGS.train_data=path + "/mem_absa/data/Restaurants_Train-fr.xml"

        self.FLAGS.edim=100

        self.FLAGS.lindim=75

        self.FLAGS.nhop=7

        self.FLAGS.batch_size=128

        self.FLAGS.nepoch=150

        self.FLAGS.init_lr=0.01

        self.FLAGS.init_std=0.05

        self.FLAGS.max_grad_norm=10

        self.FLAGS.pad_idx=0

        self.FLAGS.init_hid=0.1

        self.FLAGS.nbwords=7000
        # FLAGS.mem_size = train_data[4] if train_data[4] > test_data[4] else test_data[4]

        self.FLAGS.mem_size=500
        # print("FLAGS.mem_size : ",FLAGS.mem_size)
        return self.FLAGS
