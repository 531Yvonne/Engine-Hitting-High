# region imports
from AlgorithmImports import *
# endregion
import nltk
from nltk.tokenize import word_tokenize
from itertools import chain
import re

# Can't have a single file full of sentiment keys and values (due to QuantConnect Size Limit)
# So the sentiment words are split into multiple files and recombined in this code
from sentiment_dict_1 import WordScoreChunk1 as wsc1
from sentiment_dict_2 import WordScoreChunk2 as wsc2
from sentiment_dict_3 import WordScoreChunk3 as wsc3
from sentiment_dict_4 import WordScoreChunk4 as wsc4
from sentiment_dict_5 import WordScoreChunk5 as wsc5


class WordScore:
    def __init__(self):
        ws1 = wsc1.words
        ws2 = wsc2.words
        ws3 = wsc3.words
        ws4 = wsc4.words
        ws5 = wsc5.words

        # combine all python dictionaries
        self.wordScores = dict(chain.from_iterable(d.items()
                               for d in (ws1, ws2, ws3, ws4, ws5)))

    def score(self, article):
        '''Get sentiment score'''
        score = 0
        try:
            words = word_tokenize(article)
            words = re.sub("[^a-zA-Z]", " ", str(words))
        except:
            return 0
        score = sum([self.wordScores[word]
                    for word in words if word in self.wordScores])
        return score
