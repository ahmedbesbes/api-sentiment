
from sequence_tagging.config_seq import Config


from sentiment_analysis import load_sentiment_model, load_tagging_model, sentiment_analysis

import mysql.connector
import configuration as configuration
import spacy
from pyfasttext import FastText
# fr dictionary loading
fr_nlp=spacy.load("fr")

wiki_model=FastText()
wiki_model.load_model(configuration.pathFasttext)

config=Config()

# DB connection
conn=mysql.connector.connect(host="localhost", user="root", password="root", database="resolution")



def main():
    model_tag=load_tagging_model()
    model_sa, flags, source_count, source_word2idx=load_sentiment_model(fr_nlp, wiki_model)

    model_tag.config.logger.info("""
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

        sentiment_analysis(model_tag, model_sa, flags, source_count, source_word2idx, sentence, fr_nlp, wiki_model)

if __name__ == "__main__":
    main()
