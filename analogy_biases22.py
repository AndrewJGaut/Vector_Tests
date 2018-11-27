import logging
from web.datasets.analogy import fetch_google_analogy
from web.embeddings import *
from web.embedding import * 
from web.vocabulary import *

import matplotlib
matplotlib.use('Agg')
#%matplotlib inline
from matplotlib import pyplot as plt
import json
import random
import numpy as np


logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.DEBUG, datefmt='%I:%M:%S')


import debiaswe as dwe
import debiaswe.we as we
from debiaswe.we import WordEmbedding
from debiaswe.data import load_professions
from debiaswe.debias import debias

E = WordEmbedding('./embeddings/w2v_gnews_small.txt')


with open('./data/definitional_pairs.json', "r") as f:
    defs = json.load(f)
#print("definitional", defs)

with open('./data/equalize_pairs.json', "r") as f:
    equalize_pairs = json.load(f)

with open('./data/gender_specific_seed.json', "r") as f:
    gender_specific_words = json.load(f)
#print("gender specific", len(gender_specific_words), gender_specific_words[:10])

debias(E, gender_specific_words, defs, equalize_pairs)

v = Vocabulary(E.words)
w = Embedding(v,E.vecs) #HARD DEBIASED


file = open("analogy_set2words","r")
for line in file.readlines():
    words = line.strip().split(' ')
    w1,w2 = words[0], words[1]
    direction = E.diff(w1, w2)
    print("Question: {} is to {} as {} is to ?".format(w1, "?", w2))
    a_gender = E.best_analogies_dist_thresh(direction, 1, 10)

    for (a,b,c) in a_gender:
        print(str(a)+"-"+str(b)+"-"+str(c))
    print("\n-----------------------------------\n")
file.close()


