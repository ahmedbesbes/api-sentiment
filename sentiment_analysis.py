from sequence_tagging.config_seq import Config
from sequence_tagging.model.ner_model import NERModel
from sequence_tagging.model.data_utils import CoNLLDataset
from sequence_tagging.evaluate import align_data
import tensorflow as tf1

import numpy as np
import re

from mem_absa.load_data import init_word_embeddings
from mem_absa.load_data import read_sample, read_vocabulary
from mem_absa.mapping import mapping_sentiments
from mem_absa.model import MemN2N
from mem_absa.config_mem import Configure

from datetime import datetime



def load_tagging_model():
    # build model tagging sequence
    config=Config()
    FLAGS1=config.get_flags("./sequence_tagging")
    # build model
    model_tag=NERModel(config, FLAGS1)
    model_tag.build()

    CoNLLDataset(FLAGS1.filename_test, config.processing_word, config.processing_tag, FLAGS1.max_iter)

    model_tag.restore_session(FLAGS1.dir_model)
    return model_tag


def load_sentiment_model(fr_nlp,wiki_model):
    tf1.reset_default_graph()
    # build model sentiment classification
    source_count=[]
    source_word2idx={}
    path="."
    configure=Configure()
    FLAGS2=configure.get_flags(path)
    read_vocabulary(fr_nlp, FLAGS2.train_data , source_count, source_word2idx)
    FLAGS2.pre_trained_context_wt=init_word_embeddings(wiki_model,source_word2idx, FLAGS2.nbwords)
    FLAGS2.pre_trained_context_wt[FLAGS2.pad_idx, :]=0
    model_sa=MemN2N(FLAGS2)
    model_sa.build_model()

    model_sa.restore_session(tf1, FLAGS2.pathModel)
    return model_sa,FLAGS2, source_count, source_word2idx


# returns the elapsed milliseconds since the start of the program
def millis(start_time):
    dt=datetime.now() - start_time
    ms=(dt.days * 24 * 60 * 60 + dt.seconds) * 1000 + dt.microseconds / 1000.0
    return ms


def sentiment_analysis(model_tag, model_sa, FLAGS,source_count, source_word2idx, review, fr_nlp, wiki_model):
    #start_time=datetime.now()


    samples={}
    opinions=[]
    summury=[]

    all_aspect_words, all_aspect_categories, all_predictions=[],[],[]

    #sentences=review.split(".?!")
    sentences=re.split('\.+|\!|\?', review)
    for sentence in sentences:
        sentence_nlp=fr_nlp(sentence)
        words_raw=[]
        words_raw.extend([sp.text for sp in sentence_nlp])

        #interval1=millis(start_time)
        #print("word processing spacy :", interval1)

        preds, aspects=model_tag.predict(words_raw)
        #to_print=align_data({"input": words_raw, "output": preds})
        #for key, seq in to_print.items():
        #    model_tag.config.logger.info(seq)

        #interval2=millis(start_time)
        #print("aspect extraction :", (interval2 - interval1))


        if len(aspects) > 0:
            # model_sa, source_count, source_word2idx=load_sentiment_model()
            aspect_words=np.array(aspects)[:, 0]
            aspect_categories=np.array(aspects)[:, 1]
            aspect_idx=np.array(aspects)[:, 2]

            all_aspect_words.extend(aspect_words)
            all_aspect_categories.extend(aspect_categories)




            test_data=read_sample(fr_nlp, sentence, aspect_words,aspect_idx, source_count, source_word2idx)
            #interval31=millis(start_time)
            #print("31 :", (interval31 - interval2))

            FLAGS.pre_trained_context_wt=init_word_embeddings(wiki_model,source_word2idx, FLAGS.nbwords)
            #interval32=millis(start_time)
            #print("init 32 :", (interval32 - interval31))

            FLAGS.pre_trained_context_wt[FLAGS.pad_idx, :]=0
            #interval33=millis(start_time)
            #print("33 :", (interval33 - interval32))

            #interval3=millis(start_time)
            #print("embedding & indexation :", (interval3 - interval2))
            predictions=model_sa.predict(test_data, source_word2idx)

            #interval4=millis(start_time)
            #print("sentiment analysis :", (interval4 - interval3))
            all_predictions.extend(predictions)

            for asp, cat, idx, pred in zip(aspect_words, aspect_categories,aspect_idx, predictions):
                print(asp, " : ", str(cat), " =>", mapping_sentiments(pred), end=" ; exemple : ")
                sample=[s.strip() for s in re.split('[\.\?!,;:]', sentence) if
                        re.sub(' ', '', asp.lower()) in re.sub(' ', '', s.lower())][0]
                print(sample)
                samples[str(cat) + '_' + str(pred)]=sample
                opinion=[asp, str(cat), str(idx),str(int(idx)+len(asp)), mapping_sentiments(pred), sample]
                opinions.append(opinion)

    if len(all_aspect_words)>0:

        # summary review
        print("\n------SUMMARY REVIEW-------")
        categories=['SERVICE', 'AMBIANCE', 'QUALITE', 'PRIX', 'GENERAL', 'LOCALISATION']
        for categ in categories:
            exists=False
            total=0
            val=0
            for asp, cat, pred in zip(all_aspect_words, all_aspect_categories, all_predictions):

                if str(cat) == categ:
                    exists=True
                    total+=1
                    val+=pred
            if exists:
                sum=[categ, mapping_sentiments(round(val / total))]
                print(categ, " ", mapping_sentiments(round(val / total)), "; exemple : ", end=" ")
                summury.append(sum)
                try:
                    print(samples[categ + '_' + str(int(round(val / total)))])
                except:
                    print("conflict sentiments")
        #interval5=millis(start_time)
        #print("average aspects & summary review :", (interval5 - interval4))

        #interval6=millis(start_time)
        #print("Total :", interval6)
    else:
        print("PAS D'ASPECTS !")

    return opinions, summury
