import nltk
import re
from operator import itemgetter
import os
import util

#Altere para o path correspondente dos arquivos a serem lidos
path = "/home/runner/TP3MODELO-VETORIAL/doc-collections"
os.chdir(path)

content_file = ""
for file in sorted(os.listdir()):
  doc = open(file, "r")
  content_file += doc.read()


def pre_processing(data):
  return re.findall(r'\b[A-zÀ-úü]+\b', data.lower())


def remove_portuguese_stopwords(data: list):
  nltk.download('stopwords')
  stopwords = nltk.corpus.stopwords.words('portuguese')
  list_portuguese_stopwords = set(stopwords)
  return [word for word in data if word not in list_portuguese_stopwords]


porter = nltk.PorterStemmer()

content_file = pre_processing(content_file)
content_file = remove_portuguese_stopwords(content_file)

#Descomente para utilizar o Stemming no BoW
#content_file = [porter.stem(t) for t in content_file]

#Retirando repetição para o BoW
bow = set(content_file)
bow = list(bow)

archives = {}
for file in sorted(os.listdir()):
  doc = open(file, "r")
  content_file = doc.read()
  content_file = pre_processing(content_file)
  content_file = remove_portuguese_stopwords(content_file)
  content_file = set(content_file)
  archives[file] = list(content_file)

doc_frequency_dict = util.generate_frequency(archives, bow)
doc_tf_dict = util.calculate_tf(doc_frequency_dict)
doc_idf_list = util.calculate_idf(doc_tf_dict)
doc_tf_idf_dict = util.calculate_tf_idf(doc_tf_dict, doc_idf_list)

query = input("Digite uma consulta:")
query = pre_processing(query)
query = remove_portuguese_stopwords(query)

#Descomente para utilizar o Stemming na consulta
#query = [porter.stem(t) for t in query]

query = set(query)
query = list(query)

query_dict = {}
query_dict[1] = list(query)

query_frequency_dict = util.generate_frequency(query_dict, bow)
query_tf_dict = util.calculate_tf(query_frequency_dict)
query_tf_idf_dict = util.calculate_tf_idf(query_tf_dict, doc_idf_list)
vectorial_model = util.construct_vectorial_model(doc_tf_idf_dict,
                                                 query_tf_idf_dict[1])

sorted_vectorial_model = dict(
  sorted(vectorial_model.items(), key=itemgetter(1), reverse=True))

print("***Grau de similaridade dos Modelos Vetoriais***")

for file in sorted_vectorial_model:
  print("Arquivo", file, ":", sorted_vectorial_model[file])
