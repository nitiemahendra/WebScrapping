# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import re
import nltk
import pandas as pd
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
# nltk.download('stopwords')
# nltk.download ('punkt')
from nltk.corpus import stopwords
from nltk.corpus import webtext
from nltk.probability import FreqDist
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
#nltk.download('wordnet')
from nltk.stem.wordnet import WordNetLemmatizer
lem = WordNetLemmatizer()



# Set the URL you want to webscrape from
url = 'https://www.indiatoday.in/business/budget-2020/story/full-text-of-budget-2020-speech-by-finance-minister-nirmala-sitharaman-1642337-2020-02-01'

# Connect to the URL
response = requests.get(url)

# Parse HTML and save to BeautifulSoup object¶
soup = BeautifulSoup(response.text, "html.parser")
# print(type(soup))

# title = soup.title
# print(title)
# text = soup.get_text()
# para = soup.find_all(text=True)
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
    

# re.sub(r'<[^>]+>', '', para)
    
stop_words = set(stopwords.words('english'))
#stop_words= nltk.corpus.stopwords('english')
newStopWords = ['The','This','In','I','It','also','would']
new_stop_words = stop_words.union(newStopWords)


# print(new_stop_words)
file1 = open("para.txt")


line = file1.read()  # Use this to read file content as a stream:
words = line.split()
for r in words:
    if not r in new_stop_words:
        appendFile = open('filteredtext.txt', 'a')
        appendFile.write(" " + r)
        appendFile.close()

string = open('filteredtext.txt').read()
new_str = re.sub('[^a-zA-Z0-9]', ' ', string)
open('c.txt', 'w').write(new_str)

# new_str1= lem.lemmatize(new_str)

# print(new_str1)
"""

def graph():
    inputfile = file2.read()
    tokens = nltk.tokenize.word_tokenize(inputfile)
    fd = nltk.FreqDist(tokens)
    fd.plot(30, cumulative=False)
    fd.close()
    
""
nltk.download('webtext')
wt_words = webtext.words('testing.txt')
data_analysis = nltk.FreqDist(wt_words)
 
# Let's take the specific words only if their frequency is greater than 3.
filter_words = dict([(m, n) for m, n in data_analysis.items() if len(m) > 3])
 
for key in sorted(filter_words):
    print("%s: %s" % (key, filter_words[key]))
 
data_analysis = nltk.FreqDist(filter_words)
 
data_analysis.plot(25, cumulative=False)

"""
    
tokens = nltk.tokenize.word_tokenize(new_str)
#tokens_1 = lem.lemmatize(tokens,"v")

lem_tokens= ' '.join([lem.lemmatize(w) for w in tokens])

# fd = nltk.FreqDist(tokens)
fd = nltk.FreqDist(lem_tokens)

print(fd)
fd.plot(5, cumulative=False)
print(fd.most_common(15))
#print(tokens_1)

# print(new_str)


with open("filteredtext.txt", 'w') as output:
    output.seek(0)
    output.truncate(0)  #  output.seek(0)


with open("c.txt", 'w') as output:
    output.seek(0)
    output.truncate(0)  #  output.seek(0)
