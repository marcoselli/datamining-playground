import util
import os
from operator import itemgetter

#Alterar para o path da pasta a ser lida!
path = "/home/runner/TP3MODELO-VETORIAL/doc-collections"
os.chdir(path)

def create_and_save_bow(path: str):
    content_file = []
    for file in sorted(os.listdir()):
            file_path = f"{path}/{file}"
            content_file += util.read_text_file(file_path)

    words = [word.strip("0123456789,.!:;()[]?@#$") for word in content_file]
    words = [word.replace("'s", "") for word in words]
    unique_items = {word for word in words}
  
    #COLOCAR O PATH DA PASTA ONDE O VOCABULARIO SERÁ CRIADO!
    os.chdir("/home/runner/TP3MODELO-VETORIAL")
    util.save_text_file("vocabulario.txt", unique_items)
   
#Utilize para criar o BoW caso necessário
create_and_save_bow(path)
  
vocabulario = util.read_text_file("/home/runner/TP3MODELO-VETORIAL/vocabulario.txt")
print(vocabulario)

os.chdir(path)
archives = {}
for file in sorted(os.listdir()):
  file_path = f"{path}/{file}"
  archive = util.read_text_file(file_path)
  words = [word.strip("0123456789,.!:;()[]?@#$") for word in archive]
  words = [word.replace("'s", "") for word in words]
  archives[file] = words
  
doc_frequency_dict = util.generate_frequency(archives, vocabulario)
doc_tf_dict = util.calculate_tf(doc_frequency_dict)
doc_idf_list = util.calculate_idf(doc_tf_dict)
doc_tf_idf_dict = util.calculate_tf_idf(doc_tf_dict, doc_idf_list)

query = input("Digite uma consulta:")
words = [word.strip("0123456789,.!:;()[]?@#$") for word in query.split()]
words = [word.replace("'s", "") for word in words]
unique_items = {word for word in words}

query = {}
query[1] = list(unique_items)

query_frequency_dict = util.generate_frequency(query, vocabulario)
query_tf_dict = util.calculate_tf(query_frequency_dict)
query_tf_idf_dict = util.calculate_tf_idf(query_tf_dict, doc_idf_list)
vectorial_model = util.construct_vectorial_model(doc_tf_idf_dict, query_tf_idf_dict[1])

sorted_vectorial_model = dict(sorted(vectorial_model.items(), key=itemgetter(1), reverse=True)) 

print("***Grau de similaridade dos Modelos Vetoriais***")

for file in sorted_vectorial_model:
  print("Arquivo", file, ":", sorted_vectorial_model[file])