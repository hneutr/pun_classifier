from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from functools import reduce
import numpy as np
from nltk import pos_tag
from nltk import download as nltk_download

class PosTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.le = LabelEncoder()

        # Ok this list of tags is horrible but I couldn't find a way through nltk with code to get this list
        # If anyone can figure out a better way to do this, it would be awesome
        # For now I guess this works though
        tags = "$ # . '' ( ) * -- , : ABL ABN ABX AP AT BE BED BEDZ BEG BEM BEN BER BEZ CC CD CS DO DOD DOZ DT DTI DTS DTX EX FW HV HVD HVG HVN IN JJ JJR JJS JJT MD NC NN NN$ NNS NNS$ NNPS NP NP$ NPS NNP NPS$ NR OD PDT PN PN$ POS PP$ PP$$ PPL PPLS PPO PPS PPSS PRP PRP$ QL QLP RB RBR RBS RBT RN RP SYM TO UH VB VBD VBG VBN VBP VBZ WDT WP WP$ WPO WPS WQL WRB"
        tag_list = tags.split(" ")
        self.le.fit(tag_list)


    def fit(self, xs, ys):
        return self

    def transform(self, sentences):
        features = []
        for sentence in sentences:
            tagged_sentence = pos_tag(sentence)
            pos = list(map(lambda x: x[1], tagged_sentence))
          #  pos = self.le.transform(list(map(lambda x: x[1], tagged_sentence)))
          #  pos_str = " ".join(map(lambda x: str(int(x)), list(pos)))
          #  features.append([pos_str])
            features.append(" ".join(pos))

        return features
