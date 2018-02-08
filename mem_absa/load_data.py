import os
import pdb
import xml.etree.ElementTree as ET
from collections import Counter
import re
import string
from pyfasttext import FastText
import spacy
import numpy as np

fr_nlp=spacy.load("fr")
path='/Users/nkooli/Documents/docs/avis/review_analysis_pj/mem_absa'
wiki_model=FastText()
wiki_model.load_model(path+'/model_pyfasttext100.bin')


def standardization(item):
    # remove '\n'
    item=item.strip('\n')
    # remove ' '
    item=item.strip()
    # strip punctuation
    item=item.translate(string.punctuation)
    # supprime les espaces de debut / fin et doubles espaces
    # item=" ".join(item.split())

    return item


def _get_data_tuple(sptoks, asp_term, from_idx, to_idx, label, word2idx):
    # Find the ids of aspect term

    # list of aspect words
    aspect_is=[]
    for sptok in sptoks:
        # if sptok.idx >= from_idx and sptok.idx + len(sptok.text) <= to_idx:
        if sptok.idx < to_idx and sptok.idx + len(sptok.text) > from_idx:  # as long as it has intersection
            aspect_is.append(sptok.i)

    assert aspect_is, pdb.set_trace()

    # context word distances to aspect word
    pos_info=[]
    for _i, sptok in enumerate(sptoks):
        pos_info.append(min([abs(_i - i) for i in aspect_is]))

    # sentiment of the aspect
    lab=None
    if label == 'negative':
        lab=0
    elif label == 'neutral':
        lab=1
    elif label == "positive":
        lab=2
    else:
        raise ValueError("Unknown label: %s" % label)

    return pos_info, lab

def _get_data_raw(sptoks, asp_term, from_idx, to_idx, word2idx):
    # Find the ids of aspect term

    # list of aspect words
    aspect_is=[]
    for sptok in sptoks:
        # if sptok.idx >= from_idx and sptok.idx + len(sptok.text) <= to_idx:
        if sptok.idx < to_idx and sptok.idx + len(sptok.text) > from_idx:  # as long as it has intersection
            aspect_is.append(sptok.i)

    assert aspect_is, pdb.set_trace()

    # context word distances to aspect word
    pos_info=[]
    for _i, sptok in enumerate(sptoks):
        pos_info.append(min([abs(_i - i) for i in aspect_is]))

    return pos_info

def read_vocabulary(fname, source_count, source_word2idx):
    if os.path.isfile(fname) == False:
        raise ("[!] Data %s not found" % fname)

    tree=ET.parse(fname)
    root=tree.getroot()

    source_words, target_words, max_sent_len=[], [], 0
    for reviews in root:
        for review in reviews:
            for sentence in review:
                if sentence.find('text').text:
                    sent=sentence.find('text').text
                    # print(sent)
                    # sent=re.sub('[' + string.punctuation + ']', ' ', sent)
                    # sent=" ".join(sent.split())
                    sptoks=fr_nlp(sent.lower())
                    # print(" => ",sptoks)
                    source_words.extend([sp.text.lower() for sp in sptoks])
                    if len(sptoks) > max_sent_len:
                        max_sent_len=len(sptoks)
                    for asp_terms in sentence.iter('Opinions'):
                        for asp_term in asp_terms.findall('Opinion'):
                            if asp_term.get("target") == "conflict": continue  # TODO:
                            if asp_term.get("target") == "NULL": continue  # TODO:

                            target=asp_term.get('target')
                            # print(target)
                            # target=re.sub('[' + string.punctuation + ']', ' ', target)
                            # target=" ".join(target.split())
                            t_sptoks=fr_nlp(target.lower())
                            # print(" => ",t_sptoks)
                            # t_sptoks = fr_nlp(asp_term.get('target'))
                            target_words.extend([sp.text.lower().strip() for sp in t_sptoks])
    if len(source_count) == 0:
        source_count.append(['<pad>', 0])
    source_count.extend(Counter(source_words + target_words).most_common())
    # print(source_count)
    source_count=sorted(source_count, key=lambda x: (x[1], x[0]), reverse=True)
    # print(source_count)
    for word, _ in source_count:
        if word not in source_word2idx:
            source_word2idx[word]=len(source_word2idx)

    # print(source_word2idx)
    return max_sent_len


def read_vocabRaw(fname, source_count, source_word2idx):
    if os.path.isfile(fname) == False:
        raise ("[!] Data %s not found" % fname)
    source_words, target_words, max_sent_len=[], [], 0
    f=open(fname,"r")
    i=0
    for line in f:
        if i==0:
            sent =line.strip()
            sptoks=fr_nlp(sent.lower())
            source_words.extend([sp.text.lower() for sp in sptoks])
            if len(sptoks) > max_sent_len:
                max_sent_len=len(sptoks)
        if i==1:
            target=line.strip()
            t_sptoks=fr_nlp(target.lower())
            target_words.extend([sp.text.lower().strip() for sp in t_sptoks])
            i=-1
        i+=1
    if len(source_count) == 0:
        source_count.append(['<pad>', 0])
    source_count.extend(Counter(source_words + target_words).most_common())
    # print(source_count)
    source_count=sorted(source_count, key=lambda x: (x[1], x[0]), reverse=True)
    # print(source_count)
    for word, _ in source_count:
        if word not in source_word2idx:
            source_word2idx[word]=len(source_word2idx)
    # print(source_word2idx)
    return max_sent_len


def read_data(fname, source_count, source_word2idx):
    max_sent_len=read_vocabulary(fname, source_count, source_word2idx)
    if os.path.isfile(fname) == False:
        raise ("[!] Data %s not found" % fname)

    tree=ET.parse(fname)
    root=tree.getroot()

    source_data, source_loc_data, target_data, target_label=list(), list(), list(), list()
    for reviews in root:
        for review in reviews:
            for sentence in review:
                if sentence.find('text').text:
                    sent=sentence.find('text').text
                    # sent = re.sub('['+string.punctuation+']', ' ', sent)
                    # sent=" ".join(sent.split())
                    sptoks=fr_nlp(sent.lower())
                    # print("sptoks : ",sptoks)
                    if len(sptoks.text.strip()) != 0:
                        idx=[]
                        for sptok in sptoks:
                            # print(sptok)
                            idx.append(source_word2idx[sptok.text])
                        for asp_terms in sentence.iter('Opinions'):
                            for asp_term in asp_terms.findall('Opinion'):
                                if asp_term.get("target") == "conflict": continue  # TODO:
                                if (asp_term.get('target') == 'NULL'): continue  # TODO:
                                t_sptoks=fr_nlp(asp_term.get('target'))
                                # print(t_sptoks)
                                source_data.append(idx)
                                pos_info, lab=_get_data_tuple(sptoks, t_sptoks, int(asp_term.get('from')),
                                                              int(asp_term.get('to')), asp_term.get('polarity'),
                                                              source_word2idx)
                                source_loc_data.append(pos_info)
                                target_data.append([source_word2idx[sp.text.lower().strip()] for sp in t_sptoks])
                                target_label.append(lab)

    print("Read %s aspects from %s" % (len(source_data), fname))
    # souce_data : set of word indexes
    # source_loc_data : context word distances to aspect word
    # target_data : aspect word index
    # target_label : sentiment GT
    # max_sent_len : the maximal number of words by sentence

    return source_data, source_loc_data, target_data, target_label, max_sent_len

def read_raw(fname, source_count, source_word2idx):
    max_sent_len=read_vocabRaw(fname, source_count, source_word2idx)
    if os.path.isfile(fname) == False:
        raise ("[!] Data %s not found" % fname)

    f=open(fname, "r")
    source_data, source_loc_data, target_data, target_label=list(), list(), list(), list()

    i=0
    sent=""
    for line in f:
        if i==0:
            sent=str(line).strip()
            #print("sent ",sent)
        if i==1:
            target=str(line).strip()
            #print("target ", target)
            from_ = int(sent.find("$T$"))
            to_=from_+len(target)
            #print(from_, to_)
            review=sent.replace("$T$", target)
            #print("review : ",review)
            sptoks=fr_nlp(review.lower().strip())

            #print("sptoks : ",sptoks)
            idx, t_sptoks=[], []
            if len(sptoks.text.strip()) != 0:
                for sptok in sptoks:
                    #(sptok)
                    idx.append(source_word2idx[sptok.text])
            t_sptoks=fr_nlp(target)
            source_data.append(idx)
            pos_info=_get_data_raw(sptoks, t_sptoks, from_, to_,source_word2idx)
            source_loc_data.append(pos_info)
            target_data.append([source_word2idx[sp.text.lower().strip()] for sp in t_sptoks])
        if i==2:
            lab=line.strip()
            target_label.append(int(lab)+1)
            i=-1
        i+=1
    print("Read %s aspects from %s" % (len(source_data), fname))
    # souce_data : set of word indexes
    # source_loc_data : context word distances to aspect word
    # target_data : aspect word index
    # target_label : sentiment GT
    # max_sent_len : the maximal number of words by sentence

    return source_data, source_loc_data, target_data, target_label, max_sent_len

def read_sample(text, aspect_words, source_count, source_word2idx):
    # source_data : set of word indexes
    # source_loc_data : context word distances to aspect word
    # target_data : aspect word index
    # target_label : sentiment GT
    source_words, target_words, max_sent_len=[], [], 0

    sptoks=fr_nlp(text)
    source_words.extend([sp.text.lower().strip() for sp in sptoks])
    if len(sptoks) > max_sent_len:
        max_sent_len=len(sptoks)
    for asp_term in aspect_words:
        t_sptoks=fr_nlp(str(asp_term))
        target_words.extend([sp.text.lower().strip() for sp in t_sptoks])

        # if len(source_count) == 0:
        #     source_count.append(['<pad>', 0])
    source_count.extend(Counter(source_words + target_words))
    # source_count=sorted(source_count, key=lambda x: (x[1], x[0]), reverse=True)
    # source_count.sort(reverse=True)
    # sorted(source_count)
    # source_count.sort()

    for word in source_words:
        if word not in source_word2idx:
            source_word2idx[word]=len(source_word2idx)

    source_data, source_loc_data, target_data, target_label=list(), list(), list(), list()

    idx=[]
    for sptok in sptoks:
        idx.append(source_word2idx[sptok.text.lower().strip()])
    for aspect_word in aspect_words:
        t_sptoks=fr_nlp(str(aspect_word))
        source_data.append(idx)

        # calculate pos info ...
        from_idx=text.find(str(aspect_word))
        to_idx=from_idx + len(aspect_word)
        # list of aspect words
        aspect_is=[]
        for sptok in sptoks:
            # if sptok.idx >= from_idx and sptok.idx + len(sptok.text) <= to_idx:
            if sptok.idx < to_idx and sptok.idx + len(sptok.text) > from_idx:  # as long as it has intersection
                aspect_is.append(sptok.i)

        assert aspect_is, pdb.set_trace()

        # context word distances to aspect word
        pos_info=[]
        for _i, sptok in enumerate(sptoks):
            pos_info.append(min([abs(_i - i) for i in aspect_is]))

        lab=1
        source_loc_data.append(pos_info)
        target_data.append([source_word2idx[sp.text.lower().strip()] for sp in t_sptoks])
        target_label.append(lab)
        # target_label=[0,0,0]
    #print("Read %s aspects from %s" % (len(source_data), text))

    # souce_data : set of word indexes
    # source_loc_data : context word distances to aspect word
    # target_data : aspect word index
    # target_label : sentiment GT
    # max_sent_len : the maximal number of words by sentence
    return source_data, source_loc_data, target_data, target_label, max_sent_len

# Fasttext embedding
def init_word_embeddings(word2idx, nwords):
    wt=[]
    f=open("./word_embedding_100.txt", "w")
    for w in word2idx:
        # print(w)
        # remove accents
        w=supprime_accent(w)
        # remove punctuation
       # w=re.sub('[' + string.punctuation + ']', '', w)
        f.write(w + " ")
        liste=np.array(wiki_model[w])
        for val in liste:
            f.write(str(round(val, 5))+" ")
        f.write("\n")
        wt.append(np.array(wiki_model[w]))
    if nwords - len(word2idx) > 0:
        #print("Warning : redefine nwords !! ")
        for i in range(nwords - len(word2idx)):
            wt.append(np.array(wiki_model['']))
    return np.array(wt)


def supprime_accent(ligne):
    """ supprime les accents du texte source """
    accents={'a': ['à', 'ã', 'á', 'â'], 'e': ['é', 'è', 'ê', 'ë'], 'i': ['î', 'ï'], 'u': ['ù', 'ü', 'û'],
             'o': ['ô', 'ö']}
    for (char, accented_chars) in accents.items():
        for accented_char in accented_chars:
            ligne=ligne.replace(accented_char, char)
    return ligne
