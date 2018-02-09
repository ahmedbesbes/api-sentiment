from sequence_tagging.model.ner_model import NERModel
from sequence_tagging.config_seq import Config
from sequence_tagging.model.data_utils import CoNLLDataset
import tensorflow as tf1

import numpy as np
import re
from mem_absa.config_mem import FLAGS
from mem_absa.load_data import init_word_embeddings
from mem_absa.load_data import read_sample, read_vocabulary
from mem_absa.mapping import mapping_sentiments
from mem_absa.model import MemN2N
import configuration as confi
from datetime import datetime

# create instance of config
config=Config()


def load_tagging_model():
    # build model tagging sequence
    model_tag=NERModel(config)
    model_tag.build()
    CoNLLDataset(config.filename_test, config.processing_word, config.processing_tag, config.max_iter)

    model_tag.restore_session(config.dir_model)
    return model_tag


def load_sentiment_model(fr_nlp):
    tf1.reset_default_graph()
    # build model sentiment classification
    source_count=[]
    source_word2idx={}
    read_vocabulary(fr_nlp, FLAGS.train_data, source_count, source_word2idx)
    FLAGS.pre_trained_context_wt=init_word_embeddings(source_word2idx, FLAGS.nwords)
    FLAGS.pre_trained_context_wt[FLAGS.pad_idx, :]=0
    model_sa=MemN2N(FLAGS)
    model_sa.build_model()

    model_sa.restore_session(tf1, confi.model_path)
    return model_sa, source_count, source_word2idx


# returns the elapsed milliseconds since the start of the program
def millis(start_time):
    dt=datetime.now() - start_time
    ms=(dt.days * 24 * 60 * 60 + dt.seconds) * 1000 + dt.microseconds / 1000.0
    return ms


def sentiment_analysis(model_tag, model_sa, source_count, source_word2idx, sentence, fr_nlp):
    start_time=datetime.now()
    sentence_nlp=fr_nlp(sentence)
    words_raw=[]
    words_raw.extend([sp.text for sp in sentence_nlp])
    interval1=millis(start_time)
    print("word processing spacy :", interval1)
    preds, aspects=model_tag.predict(words_raw)
    interval2=millis(start_time)
    print("aspect extraction :", (interval2 - interval1))
    if len(aspects) > 0:
        # model_sa, source_count, source_word2idx=load_sentiment_model()
        aspect_words=np.array(aspects)[:, 0]
        aspect_categories=np.array(aspects)[:, 1]

        test_data=read_sample(fr_nlp, sentence, aspect_words, source_count, source_word2idx)
        interval31=millis(start_time)
        print("31 :", (interval31 - interval2))

        FLAGS.pre_trained_context_wt=init_word_embeddings(source_word2idx, FLAGS.nwords)
        interval32=millis(start_time)
        print("init 32 :", (interval32 - interval31))

        FLAGS.pre_trained_context_wt[FLAGS.pad_idx, :]=0
        interval33=millis(start_time)
        print("33 :", (interval33 - interval32))

        interval3=millis(start_time)
        print("embedding & indexation :", (interval3 - interval2))
        predictions=model_sa.predict(test_data, source_word2idx)

        interval4=millis(start_time)
        print("sentiment analysis :", (interval4 - interval3))

        samples={}
        opinions=[]
        for asp, cat, pred in zip(aspect_words, aspect_categories, predictions):
            print(asp, " : ", str(cat), " =>", mapping_sentiments(pred), end=" ; exemple : ")
            sample=[s.strip() for s in re.split('[\.\?!,;:]', sentence) if
                    re.sub(' ', '', asp.lower()) in re.sub(' ', '', s.lower())][0]
            print(sample)
            samples[str(cat) + '_' + str(pred)]=sample
            opinion=[asp, str(cat), mapping_sentiments(pred), sample]
            opinions.append(opinion)

            # summary review
            summury=[]
            categories=['SERVICE', 'AMBIANCE', 'QUALITE', 'PRIX', 'GENERAL', 'LOCALISATION']
            for categ in categories:
                exists=False
                total=0
                val=0
                for asp, cat, pred in zip(aspect_words, aspect_categories, predictions):

                    if str(cat) == categ:
                        exists=True
                        total+=1
                        val+=pred
                if exists:
                    sum=[categ, mapping_sentiments(round(val / total))]
                    print(categ, " ", mapping_sentiments(round(val / total)), "; exemple : ", end=" ")
                    summury.append(sum)
        interval5=millis(start_time)
        print("average aspects & summary review :", (interval5 - interval4))

    interval6=millis(start_time)
    print("Total :", interval6)

    return opinions, summury
