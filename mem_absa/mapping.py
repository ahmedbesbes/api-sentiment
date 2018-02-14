"""


"""


def mapping_categories(tok):
    translate={
        'DRINKS#PRICES': 'PRIX',
        'RESTAURANT#GENERAL': 'GENERAL',
        'SERVICE#GENERAL': 'SERVICE',
        'RESTAURANT#MISCELLANEOUS': 'GENERAL',
        'FOOD#QUALITY': 'QUALITE',
        'AMBIENCE#GENERAL': 'AMBIANCE',
        'FOOD#STYLE_OPTIONS': 'QUALITE',
        'LOCATION#GENERAL': 'LOCALISATION',
        'FOOD#PRICES': 'PRIX',
        'RESTAURANT#PRICES': 'PRIX',
        'DRINKS#QUALITY': 'QUALITE',
        'DRINKS#STYLE_OPTIONS': 'QUALITE'}
    categ=''
    try:
        categ=translate[tok]
    except:
        print("category mapping : pas de conversion pour {}".format(tok))
    return categ


def mapping_sentiments(tok):
    translate={
        0:'negative',
        1:'neutral',
        2:'positive'
    }
    sent=''
    try:
        sent=translate[tok]
    except:
        print("sentiment mapping : pas de conversion pour {}".format(phrase))
    return sent
