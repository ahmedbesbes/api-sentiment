import numpy as np
import re

from sentiment_analysis import load_sentiment_model, load_tagging_model, sentiment_analysis
from mem_absa.load_data import init_word_embeddings
from mem_absa.load_data import read_sample
from mem_absa.mapping import mapping_sentiments

import configuration as configuration

import spacy
from pyfasttext import FastText

fr_nlp=spacy.load("fr")

wiki_model=FastText()
wiki_model.load_model(configuration.pathFasttext)


def main():
    # sequence tagging
    model_tag=load_tagging_model()
    model_sa, FLAGS, source_count, source_word2idx=load_sentiment_model(fr_nlp, wiki_model)

    csv_file=open(configuration.filename_procsv, 'w')
    csv_file.write("code_etab;comment;note_globale;"
                   "sentiment_SERVICE;target_SERVICE;sample_SERVICE;sentiment_AMBIANCE;target_AMBIANCE;sample_AMBIANCE;sentiment_QUALITY;target_QUALITY;sample_QUALITY;"
                   "sentiment_PRICE;target_PRICE;sample_PRICE;sentiment_GENERAL;target_GENERAL;sample_GENERAL;sentiment_LOCATION;target_LOCATION;sample_LOCATION\n")

    # k=0
    with open(configuration.filename_proreviews, "r") as f:
        for line in f:
            pro=line[0:7]
            note=line[8:9]
            line=line[11:-2]
            csv_file.write(str(pro) + ";\"" + re.sub('\"', '&quot', line.strip()) + "\"" + ";" + str(note))

            review=line.strip()
            sentences=re.split('\.+|\!|\?', review)
            all_aspect_words, all_aspect_categories, all_predictions=[], [], []
            samples={}
            for sentence in sentences:
                sentence_nlp=fr_nlp(sentence)
                words_raw=[]
                words_raw.extend([sp.text for sp in sentence_nlp])
                # print(words_raw)
                _, aspects=model_tag.predict(words_raw)

                # print(res)
                if len(aspects) > 0:
                    aspect_words=np.array(aspects)[:, 0]
                    all_aspect_words.extend(aspect_words)

                    aspect_categories=np.array(aspects)[:, 1]
                    all_aspect_categories.extend(aspect_categories)

                    aspect_idx=np.array(aspects)[:, 2]

                    test_data=read_sample(fr_nlp, review, aspect_words, aspect_idx, source_count, source_word2idx)
                    FLAGS.pre_trained_context_wt=init_word_embeddings(wiki_model, source_word2idx, FLAGS.nbwords)
                    FLAGS.pre_trained_context_wt[FLAGS.pad_idx, :]=0

                    predictions=model_sa.predict(test_data, source_word2idx)
                    # all_predictions.extend(predictions)

                    for asp, cat, pred in zip(aspect_words, aspect_categories, predictions):
                        all_predictions.append(pred)
                        # print(asp, " : ", str(cat), " =>", mapping_sentiments(pred), end=" ; ")
                        sample=[s.strip() for s in re.split('[\.\?!,;:]', review) if
                                re.sub(' ', '', asp.lower()) in re.sub(' ', '', s.lower())][0]
                        # print(sample)
                        samples[str(cat) + '_' + str(pred)]=sample
            # print(all_predictions)
            # summary review
            # print("\n------SUMMARY REVIEW-------")
            categories=['SERVICE', 'AMBIANCE', 'QUALITE', 'PRIX', 'GENERAL', 'LOCALISATION']

            for categ in categories:
                asp_cat=""
                exists=False
                total=0
                val=0
                for asp, cat, pred in zip(all_aspect_words, all_aspect_categories, all_predictions):
                    if str(cat) == categ:
                        exists=True
                        total+=1
                        val+=pred
                        asp_cat+=asp+", "
                if exists:
                    # print(categ,mapping_sentiments(round(val / total)))
                    csv_file.write(";\"" + mapping_sentiments(round(val / total)) + "\";\""+asp_cat[0:-2]+ "\";\"")
                    try:
                        csv_file.write(samples[categ + '_' + str(int(round(val / total)))] + "\"")
                    except:
                        csv_file.write("\"conflict\"")
                        print("conflict")
                else:
                    csv_file.write(";\"-\";\"-\";\"-\"")

            csv_file.write("\n")


if __name__ == "__main__":
    main()
