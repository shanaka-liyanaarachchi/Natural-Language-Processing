import nltk

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

#_______________________STEM
stemmer= PorterStemmer()
phrase="Analysing the functionality of a laser and understanding behavioural patterns"
words= word_tokenize(phrase)
stemmed_words=[]
for word in words:
    stemmed_words.append(stemmer.stem(word))
r= " ".join(stemmed_words)
print("______________________________STEMMED_________________________")
print(r)
print("___________________________________________________________________")

#____________________LEMMATIZE
from nltk.stem import WordNetLemmatizer
lemmatizer= WordNetLemmatizer()

lemmatized_words_N=[]
lemmatized_words_V=[]
for word in words:
    lemmatized_words_N.append(lemmatizer.lemmatize(word,pos='n'))
    lemmatized_words_V.append(lemmatizer.lemmatize(word,pos='v'))
    #pos is the type of word verb, noun
ln= " ".join(lemmatized_words_N)
lv= " ".join(lemmatized_words_V)
print("_____________________________LEMMATIZED_______________________")
print(ln)
print(lv)
print("____________________________________________________________________")


