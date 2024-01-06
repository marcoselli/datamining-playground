import nltk
import re
from gensim.models import Word2Vec
import os
from sklearn.decomposition import PCA
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

path = "/home/runner/TP3MODELO-VETORIAL/doc-collections"
os.chdir(path)

content_file = []
for file in sorted(os.listdir()):
  print(file)
  doc = open(file, "r")
  content_file.append(doc.read())

def pre_processing(data):
  return re.findall(r'\b[A-zÀ-úü]+\b', data.lower())

def remove_english_stopwords(data: list):
  nltk.download('stopwords')
  stopwords = nltk.corpus.stopwords.words('english')
  list_english_stopwords = set(stopwords)
  return [word for word in data if word not in list_english_stopwords]  

for i in range(len(content_file)):
  content_file[i] = content_file[i].replace("\n", " ")
  content_file[i] = pre_processing(content_file[i])
  content_file[i] = remove_english_stopwords(content_file[i])
  content_file[i] = set(content_file[i])
  content_file[i] = list(content_file[i])

#Criando um Word2Vec utilizando a lib GENSIM
model = Word2Vec(content_file, vector_size=100, min_count=1)

#Ajuste um modelo PCA 2d para o vetor
vectors = model[model.wv.vocab]
words = list(model.wv.vocab)
pca = PCA(n_components=2)
PCA_result = pca.fit_transform(vectors)

#Preparando um DataFrame
words = pd.DataFrame(words)
PCA_result = pd.DataFrame(PCA_result)
PCA_result['x_values'] =PCA_result.iloc[0:, 0]
PCA_result['y_values'] =PCA_result.iloc[0:, 1]
PCA_final = pd.merge(words, PCA_result, left_index=True, right_index=True)
PCA_final['word'] =PCA_final.iloc[0:, 0]
PCA_data_complet =PCA_final[['word','x_values','y_values']]  

#Treinando o modelo
sns.set_style('ticks')
fig = sns.lmplot(x='x_values', y='y_values',data ='K_means_data',
           fit_reg=False,
           legend=True,
           hue='Cluster')
fig = plt.gcf()
fig.set_size_inches(8, 6)
plt.show()

#Também é possível baixar bases já treinadas, como esse exemplo do Google
model = Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True, norm_only=True)

#Minha dúvida é após ter o Word2Vec devo aplicar o TF-IDF nele???