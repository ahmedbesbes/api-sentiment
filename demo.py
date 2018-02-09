from sequence_tagging.model.ner_model import NERModel
from sequence_tagging.config_seq import Config
from sequence_tagging.model.data_utils import CoNLLDataset
from sequence_tagging.evaluate import align_data
import tensorflow as tf1

import numpy as np
import re
from mem_absa.config_mem import FLAGS
from mem_absa.load_data import init_word_embeddings
from mem_absa.load_data import read_sample, read_vocabulary
from mem_absa.mapping import mapping_sentiments
from mem_absa.model import MemN2N
import mysql.connector
import configuration as configuration
import spacy

# fr dictionary loading
fr_nlp=spacy.load("fr")
config=Config()

# DB connection
conn=mysql.connector.connect(host="localhost", user="root", password="root", database="resolution")


# build model tagging sequence
def load_tagging_model():
    model_tag=NERModel(config)
    model_tag.build()
    CoNLLDataset(config.filename_test, config.processing_word, config.processing_tag, config.max_iter)
    model_tag.restore_session(config.dir_model)
    return model_tag


# build model sentiment classification
def load_sentiment_model():
    tf1.reset_default_graph()
    source_count=[]
    source_word2idx={}
    read_vocabulary(FLAGS.train_data, source_count, source_word2idx)
    FLAGS.pre_trained_context_wt=init_word_embeddings(source_word2idx, FLAGS.nwords)
    FLAGS.pre_trained_context_wt[FLAGS.pad_idx, :]=0
    model_sa=MemN2N(FLAGS)
    model_sa.build_model()
    model_sa.restore_session(tf1, configuration.model_path)
    return model_sa, source_count, source_word2idx


def main():
    model_tag=load_tagging_model()
    model_sa, source_count, source_word2idx=load_sentiment_model()

    model_tag.logger.info("""
    This is an interactive mode.
    To exit, enter 'exit'.
    To select random Pages Jaunes review, enter 'select'.
    You can enter a review like
    input> Accueil chaleureux; prix correct
    """)

    while True:

        sentence=input("input> ")
        # sentence=remove_accent(sentence)
        sentence_nlp=fr_nlp(sentence)
        words_raw=[]
        words_raw.extend([sp.text for sp in sentence_nlp])

        if words_raw == ["exit"]: break

        # select randomly one review from Pages Jaunes database
        if words_raw == ["select"]:
            # connexion BD
            cursor=conn.cursor()
            # segmentation de la BD
            # limit 5
            cursor.execute("select comment from avis where code_an8='54053000'  order by RAND() limit 1")
            comment=cursor.fetchall()
            sentence=comment[0][0]
            sentence_nlp=fr_nlp(sentence)
            words_raw=[]
            words_raw.extend([sp.text for sp in sentence_nlp])

        preds, aspects=model_tag.predict(words_raw)
        to_print=align_data({"input": words_raw, "output": preds})
        for key, seq in to_print.items():
            model_tag.logger.info(seq)

        if len(aspects) > 0:
            aspect_words=np.array(aspects)[:, 0]
            aspect_categories=np.array(aspects)[:, 1]
            test_data=read_sample(sentence, aspect_words, source_count, source_word2idx)
            FLAGS.pre_trained_context_wt=init_word_embeddings(source_word2idx, FLAGS.nwords)
            FLAGS.pre_trained_context_wt[FLAGS.pad_idx, :]=0

            predictions=model_sa.predict(test_data, source_word2idx)
            samples={}
            for asp, cat, pred in zip(aspect_words, aspect_categories, predictions):
                print(asp, " : ", str(cat), " =>", mapping_sentiments(pred), end=" ; exemple : ")
                sample=[s.strip() for s in re.split('[\.\?!,;:]', sentence) if
                        re.sub(' ', '', asp.lower()) in re.sub(' ', '', s.lower())][0]
                print(sample)
                samples[str(cat) + '_' + str(pred)]=sample

            # summary review
            print("\n------SUMMARY REVIEW-------")
            categories=['SERVICE', 'AMBIANCE', 'QUALITE', 'PRIX', 'GENERAL', 'LOCALISATION']
            for categ in categories:
                exists=False
                total=0
                val=0
                for asp, cat, pred in zip(aspect_words, aspect_categories, predictions):
                    # print(mapping(str(cat)),categ)
                    if str(cat) == categ:
                        exists=True
                        total+=1
                        val+=pred
                if exists:
                    print(categ, " ", mapping_sentiments(round(val / total)), "; exemple : ", end=" ")
                    try:
                        print(samples[categ + '_' + str(int(round(val / total)))])
                    except:
                        print("conflict sentiments")
        else:
            print("PAS D'ASPECTS !")


if __name__ == "__main__":
    main()
