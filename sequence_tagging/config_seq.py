import os
import tensorflow as tf
from sequence_tagging.model.general_utils import get_logger
from sequence_tagging.model.data_utils import get_trimmed_fasttext_vectors, load_vocab, get_processing_word


path_log="log.txt"
class Config():
    def __init__(self):

        """Initialize hyperparameters and load vocabs

        Args:
            load_embeddings: (bool) if True, load embeddings into
                np array, else None

        """
        # load if requested (default)


    def load(self,flags):
        """Loads vocabulary, processing functions and embeddings

        Supposes that build_data.py has been run successfully and that
        the corresponding files have been created (vocab and trimmed GloVe
        vectors)

        """
        # 1. vocabulary
        self.vocab_words=load_vocab(flags.filename_words)
        self.vocab_tags=load_vocab(flags.filename_tags)
        self.vocab_chars=load_vocab(flags.filename_chars)

        self.nwords=len(self.vocab_words)
        self.nchars=len(self.vocab_chars)
        self.ntags=len(self.vocab_tags)

        # 2. get processing functions that map str -> id
        self.processing_word=get_processing_word(self.vocab_words, self.vocab_chars, lowercase=True,
                                                 chars=flags.use_chars)
        self.processing_tag=get_processing_word(self.vocab_tags, lowercase=False, allow_unk=False)

        # 3. get pre-trained embeddings
        self.embeddings=(get_trimmed_fasttext_vectors(flags.filename_trimmed) if flags.use_pretrained else None)

    def get_flags(self,path,load=True):
        # general config
        self.flags=tf.app.flags

        # directory for training outputs


        self.dir_output=path + "/results/test/"

        self.flags.DEFINE_string("dir_output",path + "/results/test/", " ")

        if not os.path.exists(self.dir_output):
            os.makedirs(self.dir_output)


        # create instance of logger
        self.logger=get_logger(self.dir_output+path_log)



        self.flags.DEFINE_string("dir_model",self.dir_output + "model.weights/", " ")


        self.flags.DEFINE_string("path_log", self.dir_output + "log.txt", " ")


        # embeddings
        self.flags.DEFINE_integer("dim_word", 100, "word embedding dimension")
        self.flags.DEFINE_integer("dim_char", 100, "character embedding dimension")

        # glove files
        #    filename_glove = "data/glove.6B.{}d.txt".format(dim_word)
        # trimmed embeddings (created from glove_filename with build_data.py)
        # filename_trimmed = "data/glove.6B.{}d.trimmed.npz".format(dim_word)
        self.dim_word=100

        self.flags.DEFINE_string("filename_trimmed",path + "/data/embedding.{}d.trimmed.npz".format(self.dim_word), " ")
        self.flags.DEFINE_boolean("use_pretrained", True, " ")


        self.flags.DEFINE_string("filename_train", path + "/data/train.txt", " ")

        self.flags.DEFINE_string("filename_test",path + "/data/test.txt", " ")
        self.flags.DEFINE_string("filename_dev", path + "/data/test.txt", " ")

        self.flags.DEFINE_integer("max_iter", None, "word embedding dimension")# if not None, max number of examples in Dataset

        # vocab (created from dataset with build_data.py)

        self.flags.DEFINE_string("filename_words", path + "/data/words.txt", " ")


        self.flags.DEFINE_string("filename_tags", path + "/data/tags.txt", " ")

        self.flags.DEFINE_string("filename_chars", path + "/data/chars.txt", " ")

        self.flags.DEFINE_string("filename_results", path + "/data/results.txt", " ")

        self.flags.DEFINE_string("filename_samples", path + "/data/reviews.txt", " ")
        self.flags.DEFINE_string("filename_aspects", path + "/data/aspects.txt", " ")

        # training
        self.flags.DEFINE_boolean("train_embeddings", True, " ")

        self.flags.DEFINE_integer("nepochs", 100, " ")
        self.flags.DEFINE_float("dropout", 0.5, " ")
        self.flags.DEFINE_integer("batchsize", 10, " ")
        self.flags.DEFINE_string("lr_method", "RMSProp"," ")

        self.flags.DEFINE_float("lr", 0.015, " ")
        self.flags.DEFINE_float("lr_decay", 0.9, " ")
        self.flags.DEFINE_integer("clip", 2, " ")
        self.flags.DEFINE_integer("nepoch_no_imprv", 5, " ")

        # model hyperparameters
        self.flags.DEFINE_integer("hidden_size_char", 50, "lstm on chars")
        self.flags.DEFINE_integer("hidden_size_lstm", 200, "lstm on word embeddings")


        # NOTE: if both chars and crf, only 1.6x slower on GPU
        self.flags.DEFINE_boolean("use_crf", True, " ") #if crf, training is 1.7x slower on CPU
        self.flags.DEFINE_boolean("use_chars", True, " ")  # if char embedding, training is 3.5x slower on CPU
        self.FLAGS=self.flags.FLAGS

        if load:
            self.load(self.FLAGS)

        return self.FLAGS