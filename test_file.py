# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import re
import nltk

import requests
from bs4 import BeautifulSoup

# nltk.download('stopwords')
# nltk.download ('punkt')
from nltk.corpus import stopwords
#from nltk.corpus import webtext
#from nltk.probability import FreqDist
#from nltk.tokenize import sent_tokenize
#from nltk.tokenize import word_tokenize
#nltk.download('wordnet')
from nltk.stem.wordnet import WordNetLemmatizer
# lem = WordNetLemmatizer()

from nltk.tag import pos_tag
# nltk.download('averaged_perceptron_tagger')

#from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.model_selection import train_test_split
#from sklearn.ensemble import RandomForestClassifier
from textblob import TextBlob

## Set the URL you want to webscrape from
url = 'https://www.indiatoday.in/business/budget-2020/story/full-text-of-budget-2020-speech-by-finance-minister-nirmala-sitharaman-1642337-2020-02-01'

## Connect to the URL ##
response = requests.get(url)

## Parse HTML and save to BeautifulSoup object ##
soup = BeautifulSoup(response.text, "html.parser")
# print(type(soup))

# title = soup.title
# print(title)
# text = soup.get_text()
# para = soup.find_all(text=True)

## Extracting only the required text ##

blacklist = [
    'style',
    'script',
    'link',
    'meta'
    # other elements,
]

text_elements = [t for t in soup.find_all(text=True) if t.parent.name not in blacklist]


# print(text_elements)

# print(type(text_elements))

with open("para.txt", 'w') as output:
    output.seek(0)
    output.truncate(0)  #  output.seek(0)
    output.write(str(text_elements))
    



## Removing Noise from the Data ##

stop_words = set(stopwords.words('english'))
newStopWords = ['The','This','In','It','also','would']
new_stop_words = stop_words.union(newStopWords)


# print(new_stop_words)

file1 = open("para.txt")

line = file1.read()  # Use this to read file content as a stream
words = line.split()
for r in words:
    if not r in new_stop_words:
        appendFile = open('filteredtext.txt', 'a')
        appendFile.write(" " + r)
        appendFile.close()



string = open('filteredtext.txt').read()
new_str = re.sub('[^a-zA-Z0-9]', ' ', string) # removing anything other than alphapbets and numbers

new_str = re.sub(r'\s+', ' ', new_str, flags=re.I) # removing spaces

new_str = re.sub(r'\^[a-zA-Z]\s+', ' ', new_str) # removing single chars

new_str = new_str.lower() # converting into lowecase
open('c.txt', 'w').write(new_str)


# print(new_str)

# # Tokenizing the text ##

tokens = nltk.tokenize.word_tokenize(new_str)

## POS tagging ##

def lemmatize_sentence(tokens):
    lemmatizer = WordNetLemmatizer()
    lemmatized_sentence = []
    for word, tag in pos_tag(tokens):
        if tag.startswith('NN'):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'
        lemmatized_sentence.append(lemmatizer.lemmatize(word, pos))
    return lemmatized_sentence

# print(lemmatize_sentence(tokens))


## Bag of Words scheme - converting text to numbers


# To debug error : Iterable over raw text documents expected, string object received
# new_str = [new_str]


# print(new_str)

#vectorizer = TfidfVectorizer (max_features=25000, min_df=7, max_df=.7, stop_words=stopwords.words('english'))
#new_str = vectorizer.fit_transform(tokens).toarray()
#
### Dividing Data into Training and Test Sets ##
#
#X_train, X_test, y_train, y_test = train_test_split(tokens,tokens, test_size=.3,train_size=.7, random_state=150)
#
#print("X_train ****************:",X_train)
#print("X_test ******************:",X_test)
#print("y_train******************:",y_train)
#print("y_test********************:",y_test)
### Training the Model ##
#
#text_classifier = RandomForestClassifier(n_estimators=200, random_state=0)
#text_classifier.fit(X_train, y_train)

## Frequency distribution of tokens ##
    
obj= TextBlob(new_str)

fd = nltk.FreqDist(tokens)

sentiment=obj.sentiment.polarity

print(sentiment)

if sentiment == 0:
  print('The text is neutral')
elif sentiment > 0:
  print('The text is positive')
else:
  print('The text is negative')
  
  
# print(fd)
fd.plot(15, cumulative=False)
print(fd.most_common(15))




# Refreshing the files

with open("filteredtext.txt", 'w') as output:
    output.seek(0)
    output.truncate(0)  #  output.seek(0)


with open("c.txt", 'w') as output:
    output.seek(0)
    output.truncate(0)  #  output.seek(0)
