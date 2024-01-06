import nltk
import re
from nltk.probability import FreqDist
import matplotlib.pyplot as plt

#Alterado o id para download
nltk_id = 'inaugural'
nltk.download(nltk_id)

# print(nltk.corpus.inaugural.readme())
washington = nltk.corpus.inaugural.raw('1789-Washington.txt')

# print(washington)

washington_letras_min =  re.findall(r'\b[A-zÀ-úü]+\b', washington.lower())
# print(washington_letras_min)

#Alteracao do stopwords para lingua inglesa
nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('english')

# print(stopwords)

list_english_stopwords = set(stopwords)
washington_letras_min_semstop = [w for w in washington_letras_min if w not in list_english_stopwords]
# print(washington_letras_min_semstop)

porter = nltk.PorterStemmer()
washington_letras_min_semstop_stem = [porter.stem(t) for t in washington_letras_min_semstop]
print(washington_letras_min_semstop)

freq_sem_stem = FreqDist(washington_letras_min_semstop)
freq_com_stem = FreqDist(washington_letras_min_semstop_stem)

print("20 palavras mais frequentes sem stem:")
print(freq_sem_stem.most_common(20))

print("20 palavras mais frequentes com stem:")
print(freq_com_stem.most_common(20))

plt.figure(figsize = (13, 8))
freq_sem_stem.plot(25, title = "Frequência de Palavras - Sem Stemming")

plt.figure(figsize = (13, 8))
freq_com_stem.plot(25, title = "Frequência de Palavras - Com Stemming")