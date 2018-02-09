import pprint
import tensorflow as tf

pp=pprint.PrettyPrinter()

flags=tf.app.flags
path='/Users/nkooli/Documents/docs/avis/review_analysis_pj'

flags.DEFINE_integer("edim", 100, "internal state dimension [100]")
flags.DEFINE_integer("lindim", 75, "linear part of the state [75]")
flags.DEFINE_integer("nhop", 7, "number of hops [7]")
flags.DEFINE_integer("batch_size", 128, "batch size to use during training [128]")
flags.DEFINE_integer("nepoch", 150, "number of epoch to use during training [100]")
flags.DEFINE_float("init_lr", 0.01, "initial learning rate [0.01]")
flags.DEFINE_float("init_hid", 0.1, "initial internal state value [0.1]")
flags.DEFINE_float("init_std", 0.05, "weight initialization std [0.05]")
flags.DEFINE_float("max_grad_norm", 10, "clip gradients to this norm [50]")
#flags.DEFINE_string("pretrain_file", path+"/mem_absa/data/glove.6B.300d.txt", "pre-trained glove vectors file path ["+path+"/mem_absa/data/glove.6B.300d.txt]")
#flags.DEFINE_string("pretrain_file", path+"/mem_absa/data/model.bin", "pre-trained glove vectors file path ["+path+"/mem_absa/data/model.bin]")
flags.DEFINE_string("train_data", path+"/mem_absa/data/Restaurants_Train-fr.xml",
                    "train gold data set path ["+path+"/mem_absa/data/Restaurants_Train-fr.xml]")

flags.DEFINE_string("test_data", path+"/mem_absa/data/Restaurants_Gold-fr.xml",
                    "test gold data set path ["+path+"/mem_absa/data/Restaurants_Gold-fr.xml]")
#flags.DEFINE_string("test_data", path+"/mem_absa/data/Museums_Gold-fr.xml",
#                    "test gold data set path ["+path+"/mem_absa/data/Museums_Gold-fr.xml]")
#flags.DEFINE_string("test_data", path+"/data/pj_sent.raw",
#                    "test gold data set path ["+path+"/data/pj_sent.raw]")

flags.DEFINE_string("test_samples", path+"/sequence_tagging/data/reviews.txt",
                    "test samples data set path ["+path+"/sequence_tagging/data/reviews.txt]")
flags.DEFINE_string("test_aspects", path+"/sequence_tagging/data/aspects.txt",
                    "test aspects data set path  ["+path+"/sequence_tagging/data/aspects.txt]")

# flags.DEFINE_string("test_data", "data/Museums_gold-fr.xml", "test gold data set path [./data/Museums_gold-fr.xml]")
flags.DEFINE_boolean("show", False, "print progress [False]")
flags.DEFINE_boolean("load_model", False, "loading model [False]")
flags.DEFINE_boolean("load_samples", False, "loading file samples [True]")
flags.DEFINE_boolean("txt_file", True, "loading txt file samples [True]")
FLAGS=flags.FLAGS
# FLAGS.pad_idx = source_word2idx['<pad>']
FLAGS.pad_idx=0

FLAGS.nwords=7000
# FLAGS.mem_size = train_data[4] if train_data[4] > test_data[4] else test_data[4]
FLAGS.mem_size=500
# print("FLAGS.mem_size : ",FLAGS.mem_size)

