import os


from sequence_tagging.model.general_utils import get_logger
from sequence_tagging.model.data_utils import get_trimmed_fasttext_vectors, load_vocab, get_processing_word
path='/Users/nkooli/Documents/docs/avis/review_analysis_pj/sequence_tagging'

class Config():
    def __init__(self, load=True):
        """Initialize hyperparameters and load vocabs

        Args:
            load_embeddings: (bool) if True, load embeddings into
                np array, else None

        """
        # directory for training outputs
        if not os.path.exists(self.dir_output):
            os.makedirs(self.dir_output)

        # create instance of logger
        self.logger = get_logger(self.path_log)

        # load if requested (default)
        if load:
            self.load()


    def load(self):
        """Loads vocabulary, processing functions and embeddings

        Supposes that build_data.py has been run successfully and that
        the corresponding files have been created (vocab and trimmed GloVe
        vectors)

        """
        # 1. vocabulary
        self.vocab_words = load_vocab(self.filename_words)
        self.vocab_tags  = load_vocab(self.filename_tags)
        self.vocab_chars = load_vocab(self.filename_chars)

        self.nwords     = len(self.vocab_words)
        self.nchars     = len(self.vocab_chars)
        self.ntags      = len(self.vocab_tags)

        # 2. get processing functions that map str -> id
        self.processing_word = get_processing_word(self.vocab_words,
                self.vocab_chars, lowercase=True, chars=self.use_chars)
        self.processing_tag  = get_processing_word(self.vocab_tags,
                lowercase=False, allow_unk=False)

        # 3. get pre-trained embeddings
        self.embeddings = (get_trimmed_fasttext_vectors(self.filename_trimmed)
                if self.use_pretrained else None)


    # general config
    dir_output = path+"/results/test/"
    dir_model  = dir_output + "model.weights/"
    path_log   = dir_output + "log.txt"

    # embeddings
    dim_word = 100
    dim_char = 100

    # glove files
#    filename_glove = "data/glove.6B.{}d.txt".format(dim_word)
    # trimmed embeddings (created from glove_filename with build_data.py)
    #filename_trimmed = "data/glove.6B.{}d.trimmed.npz".format(dim_word)
    filename_trimmed=path+"/data/embedding.{}d.trimmed.npz".format(dim_word)
    use_pretrained = True

    # dataset
    # filename_dev = "data/coNLL/eng/eng.testa.iob"
    # filename_test = "data/coNLL/eng/eng.testb.iob"
    # filename_train = "data/coNLL/eng/eng.train.iob"

    filename_train  = path+"/data/train.txt" # train
    filename_dev=filename_test= path+"/data/pj_seq.txt" # test

    max_iter = None # if not None, max number of examples in Dataset

    # vocab (created from dataset with build_data.py)
    filename_words = path+"/data/words.txt"
    filename_tags = path+"/data/tags.txt"
    filename_chars = path+"/data/chars.txt"

    filename_results=path+"/data/results.txt"
    filename_samples=path+"/data/reviews.txt"
    filename_aspects=path+"/data/aspects.txt"

    # training
    train_embeddings = True
    nepochs          = 100
    dropout          = 0.5
    batch_size       = 10
    lr_method        = "RMSProp"
    lr               = 0.015
    lr_decay         = 0.9
    clip             = 2 # if negative, no clipping
    nepoch_no_imprv  = 5

    # model hyperparameters
    hidden_size_char = 50 # lstm on chars
    hidden_size_lstm = 200 # lstm on word embeddings

    # NOTE: if both chars and crf, only 1.6x slower on GPU
    use_crf = True # if crf, training is 1.7x slower on CPU
    use_chars = False # if char embedding, training is 3.5x slower on CPU
