###### Antonym sources:
#         http://www.michigan-proficiency-exams.com/antonym-list.html
#         http://kidspicturedictionary.com/word-must-know/vocabulary/vocabulary-list-by-opposites-or-antonyms/
#         http://www.myenglishpages.com/site_php_files/vocabulary-lesson-opposites1.php
#      simplifications:
#         to was deleted before verbs:
#             to argue	to agree ->  argue agree
#         worsd with more than one antonyms like
#             attack	defense, protection:
#                 attack	defense
#                 attack	protection
#         
#######################################

from sklearn.base import BaseEstimator, TransformerMixin
from collections import defaultdict

def remov_duplicate(t1, k):
	t2 = [t1[1], t1[0]]
	if t1 not in k:
		if t2 in k:
			pass
		else:
			k.append(t1)

def make_raw_antonym_list():
	antonym_raw = open('features/antonym.txt', 'r').readlines()

	antonym_raw_list = []

	for antonym in antonym_raw:
		antonym = antonym.strip().lower().split('\t')
		if "to " in antonym[0][:3]:
			if "," in antonym[1]:
				first = antonym[0][3:].strip()
				words = antonym[1].split(',')
				for i in range(0, len(words)):
					words[i] = words[i].strip()[3:]
				
				for w in words:
					temp = [first, w]
					remov_duplicate(temp, antonym_raw_list)
				
			else:
				first = antonym[0][3:].strip()
				second = antonym[1][3:].strip()
				temp = [first, second]
				remov_duplicate(temp, antonym_raw_list)
		else:
			if "," in antonym[1]:
				first = antonym[0]
				words = antonym[1].split(',')
				for w in words:
					temp = [first, w.strip()]
					remov_duplicate(temp, antonym_raw_list)
			else:
				temp = [ antonym[0], antonym[1] ]
				remov_duplicate(temp, antonym_raw_list)
				
		
	return antonym_raw_list	
		

class Antonyms (BaseEstimator, TransformerMixin):
    
    def __init__(self):
        pass

    def fit(self, x, y):
        return self
        
    def transform(self, x):
        antonyms = make_raw_antonym_list()
        features =[]
        for pun in x:
            pun = pun.strip().lower()
            temp = {'antonym':0}
            for i in antonyms:
                if i[0] in pun:
                    if i[1] in pun:
                        temp['idiom'] = 1
                    else:
                        pass
            
            features.append(temp)
        
        
        return features





