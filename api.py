from requests import codes as http_codes
import os

from flask import Flask, jsonify, request, Blueprint
from flask_cors import CORS
from flask_restplus import Api, Resource
from flask_restplus import abort

from commons.logger import logger, configure
from commons import configuration

import configuration as confi

from sentiment_analysis import load_sentiment_model, load_tagging_model, sentiment_analysis

import spacy
from pyfasttext import FastText
# fr dictionary loading
fr_nlp=spacy.load("fr")

wiki_model=FastText()
wiki_model.load_model(confi.pathFasttext)

import traceback

# Chargement de la conf
conf=configuration.load()
script_dir=os.path.dirname(__file__)


def _init_app(p_conf):
    # Configuration du logger
    configure(conf['log']['level_values'][conf['log']['level']], conf['log']['dir'], conf['log']['filename'],
              conf['log']['max_filesize'], conf['log']['max_files'])

    # Load app config into Flask WSGI running instance
    r_app=Flask(__name__)
    r_app.config['API_CONF']=p_conf

    # Autoriser le Cross-origin (CORS)
    r_app.config['CORS_HEADERS']='Auth-Token, Content-Type, User, Content-Length'
    CORS(r_app, resources={r"/*": {"origins": "*"}})

    # Documentation swagger
    # L'utilisation de blueprint permet de modifier la route associée à la doc
    blueprint=Blueprint('api', __name__)
    r_swagger_api=Api(blueprint, doc='/' + conf['url_prefix'] + '/doc/', title='API',
                      description="Api pour l'analyse de sentiments à base d'aspects")
    r_app.register_blueprint(blueprint)
    r_ns=r_swagger_api.namespace(name=conf['url_prefix'], description="Documentation de l'api")

    return r_app, r_swagger_api, r_ns


app, swagger_api, ns=_init_app(conf)
model_tag=load_tagging_model()
model_sa, flags, source_count, source_word2idx=load_sentiment_model(fr_nlp,wiki_model)


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
        response={'status_code': 200, 'message': 'Heartbeat'}

        return _success(response)


@ns.route("/supervision")
class Supervision(Resource):
    @staticmethod
    def get():
        """
            Retourne la configuration de l'api
        """
        response=None
        try:
            response=app.config['API_CONF']
        except Exception:
            abort(http_codes.SERVER_ERROR, "Erreur interne lors de la récupération de la configuration")

        return _success(response)


# Doc de la route /models
doc_parser=swagger_api.parser()

# Doc de la route /quiquoiou?phrase=<phrase>
doc_parser2=swagger_api.parser()
doc_parser2.add_argument(name='avis', required=True, type=str, help="Le texte d'opinion à analyser")


@ns.route("/aspectsentiment", endpoint="/aspectsentiment")
@swagger_api.expect(doc_parser2)
class aspectsentiment(Resource):
    @staticmethod
    def get():

        try:
            sentence=request.args.get('avis', None)

        except Exception:
            abort(http_codes.SERVER_ERROR, "Erreur lors du chargement du modèle")

        logger.info("Analyse de {sentence}".format(sentence=sentence))

        opinions, summury=sentiment_analysis(model_tag, model_sa,flags, source_count, source_word2idx, sentence, fr_nlp,wiki_model)

        response={
            'aspects': [{'target': opinion[0], 'category': opinion[1], 'sentiment': opinion[2], 'exemple': opinion[3]}
                        for opinion in opinions],
            'summury': [{'category': sum[0], 'sentiment': sum[1]} for sum in summury],

        }

        logger.info(response)

        return _success(response)


def _success(response):
    return make_reponse(response, http_codes.OK)


def _failure(exception, http_code=http_codes.SERVER_ERROR):
    try:
        exn=traceback.format_exc(exception)
        logger.info("EXCEPTION: {}".format(exn))
    except:
        logger.info("EXCEPTION: {}".format(exception))
    try:
        data, code=exception.to_tuple()
        return make_reponse(data, code)
    except:
        try:
            data=exception.to_dict()
            return make_reponse(data, exception.http)
        except Exception as exn:
            return make_reponse(None, http_code)


def make_reponse(p_object=None, status_code=200):
    """
        Fabrique un objet Response à partir d'un p_object et d'un status code
    """
    if p_object is None and status_code == 404:
        p_object={"status": {"status_content": [{"code": "404 - Not Found", "message": "Resource not found"}]}}

    json_response=jsonify(p_object)
    json_response.status_code=status_code
    json_response.content_type='application/json;charset=utf-8'
    json_response.headers['Cache-Control']='max-age=3600'
    return json_response


if __name__ == "__main__":
    # Run http REST stack
    logger.info("Run api on {}:{}".format(conf['host'], conf['port']))
    app.run(host=conf['host'], port=int(conf['port']), debug=conf['log']['level'] == "DEBUG")
