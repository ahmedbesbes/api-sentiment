from requests import codes as http_codes
import os

from flask import Flask, jsonify, request, Blueprint
from flask_cors import CORS
from flask_restplus import Api, Resource
from flask_restplus import abort

from commons.logger import logger, configure
from commons import configuration


from sequence_tagging.model.ner_model import NERModel
from sequence_tagging.config_seq import Config
from sequence_tagging.model.data_utils import CoNLLDataset
from sequence_tagging.evaluate import align_data
import tensorflow as tf1

import pickle
import numpy as np
import re
from mem_absa.config_mem import FLAGS,pp,flags
from mem_absa.load_data import init_word_embeddings
from mem_absa.load_data import read_sample, read_vocabulary
from mem_absa.mapping import mapping_sentiments
from mem_absa.model import MemN2N
import configuration as confi


import spacy
fr_nlp= spacy.load("fr")

import traceback

# Chargement de la conf
conf = configuration.load()
script_dir = os.path.dirname(__file__)

# create instance of config
config=Config()


def _init_app(p_conf):
    # Configuration du logger
    configure(conf['log']['level_values'][conf['log']['level']],
              conf['log']['dir'], conf['log']['filename'],
              conf['log']['max_filesize'], conf['log']['max_files'])

    # Load app config into Flask WSGI running instance
    r_app = Flask(__name__)
    r_app.config['API_CONF'] = p_conf

    # Autoriser le Cross-origin (CORS)
    r_app.config['CORS_HEADERS'] = 'Auth-Token, Content-Type, User, Content-Length'
    CORS(r_app, resources={r"/*": {"origins": "*"}})

    # Documentation swagger
    # L'utilisation de blueprint permet de modifier la route associée à la doc
    blueprint = Blueprint('api', __name__)
    r_swagger_api = Api(blueprint, doc='/' + conf['url_prefix'] + '/doc/',
                        title='API',
                        description="Api pour l'analyse de sentiments à base d'aspects")
    r_app.register_blueprint(blueprint)
    r_ns = r_swagger_api.namespace(name=conf['url_prefix'], description="Documentation de l'api")

    return r_app, r_swagger_api, r_ns

def load_tagging_model():
    # build model tagging sequence
    model_tag=NERModel(config)
    model_tag.build()
    CoNLLDataset(config.filename_test, config.processing_word, config.processing_tag, config.max_iter)

    model_tag.restore_session(config.dir_model)
    return model_tag


def load_sentiment_model():
    tf1.reset_default_graph()

    # build model sentiment classification
    source_count=[]
    source_word2idx={}
    read_vocabulary(FLAGS.train_data, source_count, source_word2idx)
    FLAGS.pre_trained_context_wt=init_word_embeddings(source_word2idx, FLAGS.nwords)
    FLAGS.pre_trained_context_wt[FLAGS.pad_idx, :]=0
    model_sa=MemN2N(FLAGS)
    model_sa.build_model()

    model_sa.restore_session(tf1, confi.model_path)
    return model_sa,source_count,source_word2idx


app, swagger_api, ns = _init_app(conf)
model_tag=load_tagging_model()

# Access log query interceptor
@app.before_request
def access_log():
    logger.info("{0} {1}".format(request.method, request.path))


@ns.route('/heartbeat')
class Heart(Resource):
    @staticmethod
    def get():
        """
            Heartbeat
            Est-ce que l'api est en vie ?
        """
        response = {
            'status_code': 200,
            'message': 'Heartbeat'
        }

        return _success(response)


@ns.route("/supervision")
class Supervision(Resource):
    @staticmethod
    def get():
        """
            Retourne la configuration de l'api
        """
        response = None
        try:
            response = app.config['API_CONF']
        except Exception:
            abort(http_codes.SERVER_ERROR, "Erreur interne lors de la récupération de la configuration")

        return _success(response)


# Doc de la route /models
doc_parser = swagger_api.parser()

# Doc de la route /quiquoiou?phrase=<phrase>
doc_parser2 = swagger_api.parser()
doc_parser2.add_argument(name='avis', required=True, type=str, help="Le texte d'opinion à analyser")



@ns.route("/aspectsentiment", endpoint="/aspectsentiment")
@swagger_api.expect(doc_parser2)
class aspectsentiment(Resource):




    @staticmethod
    def get():



        try:
            sentence = request.args.get('avis', None)

        except Exception:
            abort(http_codes.SERVER_ERROR, "Erreur lors du chargement du modèle")

        logger.info("Analyse de {sentence}".format(
            sentence=sentence
        ))

        sentence_nlp=fr_nlp(sentence)
        words_raw=[]
        words_raw.extend([sp.text for sp in sentence_nlp])

        preds, aspects=model_tag.predict(words_raw)

        if len(aspects) > 0:
            model_sa, source_count, source_word2idx=load_sentiment_model()
            aspect_words=np.array(aspects)[:, 0]
            aspect_categories=np.array(aspects)[:, 1]
            test_data=read_sample(sentence, aspect_words, source_count, source_word2idx)
            FLAGS.pre_trained_context_wt=init_word_embeddings(source_word2idx, FLAGS.nwords)
            FLAGS.pre_trained_context_wt[FLAGS.pad_idx, :]=0

            predictions=model_sa.predict(test_data, source_word2idx)
            samples={}
            opinions=[]
            for asp, cat, pred in zip(aspect_words, aspect_categories, predictions):
                print(asp, " : ", str(cat), " =>", mapping_sentiments(pred), end=" ; exemple : ")
                sample=[s.strip() for s in re.split('[\.\?!,;:]', sentence) if re.sub(' ', '', asp.lower())in re.sub(' ', '',s.lower())][0]
                print(sample)
                samples[str(cat) + '_' + str(pred)]=sample
                opinion=[asp,str(cat),mapping_sentiments(pred),sample]
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
                        sum=[categ,mapping_sentiments(round(val / total))]
                        print(categ, " ", mapping_sentiments(round(val / total)), "; exemple : ", end=" ")
                        summury.append(sum)


        response = {
            'aspects': [ {'target': opinion[0], 'category': opinion[1], 'sentiment': opinion[2], 'exemple': opinion[3]} for opinion in opinions ],
            'summury':[ { 'category': sum[0], 'sentiment': sum[1]} for sum in summury ],

        }

        logger.info(response)

        return _success(response)

def _success(response):
    return make_reponse(response, http_codes.OK)

def _failure(exception, http_code=http_codes.SERVER_ERROR):
    try:
        exn = traceback.format_exc(exception)
        logger.info("EXCEPTION: {}".format(exn))
    except:
        logger.info("EXCEPTION: {}".format(exception))
    try:
        data, code = exception.to_tuple()
        return make_reponse(data, code)
    except:
        try:
            data = exception.to_dict()
            return make_reponse(data, exception.http)
        except Exception as exn:
            return make_reponse(None, http_code)


def make_reponse(p_object=None, status_code=200):
    """
        Fabrique un objet Response à partir d'un p_object et d'un status code
    """
    if p_object is None and status_code == 404:
        p_object = {"status": {"status_content": [{"code": "404 - Not Found", "message": "Resource not found"}]}}

    json_response = jsonify(p_object)
    json_response.status_code = status_code
    json_response.content_type = 'application/json;charset=utf-8'
    json_response.headers['Cache-Control'] = 'max-age=3600'
    return json_response

if __name__ == "__main__":
    # Run http REST stack
    logger.info("Run api on {}:{}".format(conf['host'], conf['port']))
    app.run(host=conf['host'], port=int(conf['port']), debug=conf['log']['level'] == "DEBUG")
