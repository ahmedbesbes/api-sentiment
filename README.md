# Aspect based sentiment analysis with Tensorflow

The problem is treated in two phases :
- Aspect extraction
consists of extracting and classifying aspect targets in the review
- Sentiment analysis
consists of assigning a tonality (negative, neutral, positive) to each extracted aspect target

## Install and api launching
```
pip install -r requirements.txt
python api.py
curl --request GET --url http://localhost:9090/api-sentiment-1/doc/
```
## Aspect extraction

1. Build vocab from the data according to the config in `sequence_tagging/model/config.py`.

```
python sequence_tagging/build_data.py
```

2. Train the model with

```
python sequence_tagging/train.py
```


3. Evaluate and interact with the model with
```
python sequence_tagging/evaluate.py
```

## Sentiment analysis

Train/test a model according to the config in `mem_absa/config_mem.py`
```
python mem_absa/train_test.py --show True
```
