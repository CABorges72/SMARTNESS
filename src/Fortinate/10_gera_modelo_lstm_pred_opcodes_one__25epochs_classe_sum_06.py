import numpy as np
import matplotlib.pyplot as plt
import json
import fnmatch
import gzip
import sys,time
from nltk import tokenize
import nltk
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

#from keras.models import Sequential
#from keras.layers import Dense
#from keras.layers import LSTM
from keras.models import load_model


from tensorflow.keras import Sequential, Input, Model
from keras.layers import Dense, Dropout
from keras.layers import LSTM, CuDNNLSTM, MaxPooling1D, Conv1D, Flatten
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#--------------------------

from keras.preprocessing.text import Tokenizer
from numpy import array
from numpy import argmax
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

def load_dict_from_file_gz(gzfile):
    f = gzip.open(gzfile, 'r')
    byte_stream = f.read()
    dict_merge=eval(byte_stream)
    f.close()
    return dict_merge

#  utilizando versão original
#vetor = np.load("/content/drive/MyDrive/maldrive/i2v_opcodes_100.npy") # dim 100
vetor = np.load("/content/drive/MyDrive/maldrive/WGRS-2024/vetorone00.npy") # dim 100
#dim=128
dim=64
#dict_all=load_dict_from_file_gz("/content/drive/MyDrive/maldrive/dict_opcodes_rva_capstone.dic.gz")
dict_all=load_dict_from_file_gz("/content/drive/MyDrive/maldrive/WGRS-2024/sumarios/dicionarios/dict_opcodes_one_sum_06.dic.gz")
#opcodes=np.array( tuple(dict_all.keys()) )
opcodes=np.array( tuple(dict_all['opcode'].values()) )

#dict_all['opcode'].keys()

#vetor.shape

#opcodes.shape

#dict_all['opcode'].values()

#dict_all

opcodes

opcodes=opcodes[0:64]
opcodes

#------------------------------------
doc=str(opcodes.tolist()).replace('[','').replace(']','').replace("'","").replace(',','')
#doc=base_treinamento.tolist()
len(doc)

#doc = "Can I eat the Pizza".lower().split()
doc = doc.lower().split()
# create the tokenizer
#t = Tokenizer()
#------------------------------------


label_encoder = LabelEncoder()
data = label_encoder.fit_transform(doc)
data = array(data)
# one hot encode
encoded = to_categorical(data)
print(encoded)

vetor=encoded
vetor[63]

#Criação de matriz de dados mapeando X->Y anterior->proximo
def create_dataset(elemento, look_back=1):
    dataX, dataY = [], []
    for i in range(len(elemento)-look_back):
        a = elemento[i:(i+look_back)]
        dataX.append(a)
        dataY.append(elemento[i + look_back])
    return np.array(dataX), np.array(dataY)

#label=int(sys.argv[1]) # modelo que será gerado
label=7 # modelo que será gerado
dd=3
look_back = 5
#look_back = 90
epoch = 100 # problema de tempo de execução para famílias com muitas amostras.
#epoch = 1 # problema de tempo de execução para famílias com muitas amostras.
units = dim # dimensão de vetor
batch = 100

model = Sequential()
#model.add(LSTM(units, input_shape=(look_back, dim)))
##model.add(CuDNNLSTM(units, input_shape=(look_back, dim)))
model.add(CuDNNLSTM(units, return_sequences=True,input_shape=(look_back, dim)))
model.add(CuDNNLSTM(units, return_sequences=True))
model.add(CuDNNLSTM(units, return_sequences=True))
model.add(CuDNNLSTM(units))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['acc'])

dir_load="/content/drive/MyDrive/maldrive/modelo100" # windows
caminho="modelos100/"

# a ideia de criar memory foi para que a mattriz a não ultrapassasse o tamanho máximo de memória
# 100 -> 100Mb aprox., teste empírico. calculamos tam_memory por formula aproximada.
memory=100 # quantidade em Mb que pode ocupar cada partição de a
tam_memory=int(memory*(200/units)*10000/76)
model.summary()

def load_opcodes_classe_txt(file_path):
    arquivo_opcodes = file_path
    linha_opcodes = ' '
    linha_list = []
    print(arquivo_opcodes)
    try:
        f = open(arquivo_opcodes,'r')
        linha_opcodes = f.read()
        linha_list = eval(linha_opcodes)
        #print(linha_list)
        f.close()
    except:
        print('O arquivo não existe!')
    return linha_list

def load_opcodes_classe_txt_token(file_path):
    arquivo_opcodes = file_path
    linha_opcodes = ' '
    linha_list = []
    print(arquivo_opcodes)
    try:
        f = open(arquivo_opcodes,'r')
        linha_opcodes = f.read()
        #frase = "Bem vindo ao mundo do PLN"
        token_espaco = tokenize.WhitespaceTokenizer()
        linha_list = token_espaco.tokenize(linha_opcodes)
        #print(token_frase)
        #linha_list = eval(linha_opcodes)
        #print(linha_list)
        f.close()
    except:
        print('O arquivo não existe!')
    return linha_list



"""#Experimento 01
#Classe 06 - Sumario
"""

dim=64

units = dim # dimensão de vetor
# a ideia de criar memory foi para que a mattriz a não ultrapassasse o tamanho máximo de memória
# 100 -> 100Mb aprox., teste empírico. calculamos tam_memory por formula aproximada.
memory=100*3 # quantidade em Mb que pode ocupar cada partição de a
tam_memory=int(memory*(200/units)*10000/76)

tam_memory

cla = 7
#look_back = 5
#dim = 128
#file = '/content/drive/MyDrive/maldrive/arq_todos_opcodes_classe_03_completo.txt'
file = '/content/drive/MyDrive/maldrive/WGRS-2024/sumarios/arq_todos_opcodes_classe_06_sumario.txt'
file_path = file.strip('\n')
lista_opcodes = load_opcodes_classe_txt_token(file_path)

codeSequence = np.array(lista_opcodes)

#tamanho = int(round(len(codeSequence)*80/100,0))
tamanho = int(round(len(codeSequence)*90/100,0))
tamanho = 340000
codeSequence_treino = codeSequence[:tamanho]
codeSequence_teste = codeSequence[(tamanho):374000]

codeSequence_treino.shape

codeSequence_teste.shape

codeSequenceSelection=codeSequence_treino
linhanumerada = np.zeros([len(codeSequenceSelection),dim])
for index,code in enumerate(codeSequenceSelection):
  elemento = np.zeros([1,dim])
  elemento = vetor[(opcodes==code)]
  if (elemento.sum()==0.0):
    elemento = np.zeros([1,dim])
  linhanumerada[index]=elemento

look_back = 5
a, b = create_dataset(linhanumerada,look_back)
history = model.fit(a, b, epochs=25, batch_size=batch,verbose=0)

history.history

#plt.plot(range(1,25), history.history['val_loss'], marker='+')
plt.plot(range(1,26), history.history['loss'], marker='o')

#plt.plot(range(1,101), history.history['val_acc'], marker='+')
plt.plot(range(1,26), history.history['acc'], marker='o')

"""#Teste"""

codeSequenceSelection=codeSequence_teste
codeSequenceSelection.shape

linhanumerada = np.zeros([len(codeSequenceSelection),dim])
for index,code in enumerate(codeSequenceSelection):
  elemento = np.zeros([1,dim])
  elemento = vetor[(opcodes==code)]
  if (elemento.sum()==0.0):
    elemento = np.zeros([1,dim])
  linhanumerada[index]=elemento

look_back = 5
a, b = create_dataset(linhanumerada,look_back)

a.shape

eval = model.evaluate(x=a, y=b)

#a.shape

#b.shape

#aa = np.reshape(a, (a.shape[0], a.shape[1], 1))

predicts = model.predict(a, batch_size=128)

#predicts.shape

#predicts[0]

#b[5]

#new_array[5]

from tensorflow.keras.utils import to_categorical
new_array = to_categorical(np.argmax(predicts, axis=1), 128)
new_array.shape
acertos=0
for i in range(new_array.shape[0]):
  if (b[i].argmax()==new_array[i].argmax()):
    acertos = acertos + 1

print(acertos/new_array.shape[0])

# One 64 ---> 0.6668627739373437

# 374000 ---> 0.7039564641859097

# 363000 ---> 0.6856190331868465

# 440000 ---> 0.5081885235654456

#528000 --->  0.15234920304198354

# 352000 --->  0.633505235192999

# 176000 --->  0.06464520162550798

# 88000 --->   0.05791119449656035

# 44000 --->

# 22000 --->

# 12000 --->

"""# Salvar e modelo"""

ep=25
label = 6
graph = []
model.save("/content/drive/MyDrive/maldrive/WGRS-2024/sumarios/modelo100/model_LSTM_" + str(label) +"_"+str(ep)+ ".h5") # ler modelo de cada epoch
#graph.append(history.history["loss"][0])
graph.append(history.history)
np.save("/content/drive/MyDrive/maldrive/WGRS-2024/sumarios/modelo100/graph"+str(label)+".npy",np.array(graph))

"""#Ler o modelo"""

label=7
model = load_model("/content/drive/MyDrive/maldrive/experimentos/experimento01/sumarios/modelo100/model_LSTM_" + str(label) +"_"+str(25)+ ".h5") # ler modelo de cada epoch
model.summary()

"""


#Teste do modelo 6 como classe 1"""

cla = 1
#look_back = 5
#dim = 128
#file = '/content/drive/MyDrive/maldrive/arq_todos_opcodes_classe_03_completo.txt'
file = '/content/drive/MyDrive/maldrive/WGRS-2024/sumarios/arq_todos_opcodes_classe_01_sumario.txt'
file_path = file.strip('\n')
lista_opcodes = load_opcodes_classe_txt_token(file_path)

codeSequence = np.array(lista_opcodes)

#tamanho = int(round(len(codeSequence)*80/100,0))
tamanho = int(round(len(codeSequence)*90/100,0))
tamanho = 80000
codeSequence_treino = codeSequence[:tamanho]
codeSequence_teste = codeSequence[(tamanho):88000]

codeSequenceSelection=codeSequence_teste
codeSequenceSelection.shape

linhanumerada = np.zeros([len(codeSequenceSelection),dim])
for index,code in enumerate(codeSequenceSelection):
  elemento = np.zeros([1,dim])
  elemento = vetor[(opcodes==code)]
  if (elemento.sum()==0.0):
    elemento = np.zeros([1,dim])
  linhanumerada[index]=elemento

look_back = 5
a, b = create_dataset(linhanumerada,look_back)
a.shape

eval = model.evaluate(x=a, y=b)



"""#Teste do modelo 6 como classe 2"""

cla = 2
#look_back = 5
#dim = 128
#file = '/content/drive/MyDrive/maldrive/arq_todos_opcodes_classe_03_completo.txt'
file = '/content/drive/MyDrive/maldrive/WGRS-2024/sumarios/arq_todos_opcodes_classe_02_sumario.txt'
file_path = file.strip('\n')
lista_opcodes = load_opcodes_classe_txt_token(file_path)

codeSequence = np.array(lista_opcodes)

#tamanho = int(round(len(codeSequence)*80/100,0))
tamanho = int(round(len(codeSequence)*90/100,0))
tamanho = 80000
codeSequence_treino = codeSequence[:tamanho]
codeSequence_teste = codeSequence[(tamanho):88000]

codeSequenceSelection=codeSequence_teste
codeSequenceSelection.shape

linhanumerada = np.zeros([len(codeSequenceSelection),dim])
for index,code in enumerate(codeSequenceSelection):
  elemento = np.zeros([1,dim])
  elemento = vetor[(opcodes==code)]
  if (elemento.sum()==0.0):
    elemento = np.zeros([1,dim])
  linhanumerada[index]=elemento

look_back = 5
a, b = create_dataset(linhanumerada,look_back)
a.shape

eval = model.evaluate(x=a, y=b)

"""#Teste do modelo 6 como classe 3"""

cla = 3
#look_back = 5
#dim = 128
#file = '/content/drive/MyDrive/maldrive/arq_todos_opcodes_classe_03_completo.txt'
file = '/content/drive/MyDrive/maldrive/WGRS-2024/sumarios/arq_todos_opcodes_classe_03_sumario.txt'
file_path = file.strip('\n')
lista_opcodes = load_opcodes_classe_txt_token(file_path)

codeSequence = np.array(lista_opcodes)

#tamanho = int(round(len(codeSequence)*80/100,0))
tamanho = int(round(len(codeSequence)*90/100,0))
tamanho = 80000
codeSequence_treino = codeSequence[:tamanho]
codeSequence_teste = codeSequence[(tamanho):88000]

codeSequenceSelection=codeSequence_teste
codeSequenceSelection.shape

linhanumerada = np.zeros([len(codeSequenceSelection),dim])
for index,code in enumerate(codeSequenceSelection):
  elemento = np.zeros([1,dim])
  elemento = vetor[(opcodes==code)]
  if (elemento.sum()==0.0):
    elemento = np.zeros([1,dim])
  linhanumerada[index]=elemento

look_back = 5
a, b = create_dataset(linhanumerada,look_back)
a.shape

eval = model.evaluate(x=a, y=b)

"""#Teste do modelo 6 como classe 4"""

cla = 4
#look_back = 5
#dim = 128
#file = '/content/drive/MyDrive/maldrive/arq_todos_opcodes_classe_03_completo.txt'
file = '/content/drive/MyDrive/maldrive/WGRS-2024/sumarios/arq_todos_opcodes_classe_04_sumario.txt'
file_path = file.strip('\n')
lista_opcodes = load_opcodes_classe_txt_token(file_path)

codeSequence = np.array(lista_opcodes)

#tamanho = int(round(len(codeSequence)*80/100,0))
tamanho = int(round(len(codeSequence)*90/100,0))
tamanho = 80000
codeSequence_treino = codeSequence[:tamanho]
codeSequence_teste = codeSequence[(tamanho):88000]

codeSequenceSelection=codeSequence_teste
codeSequenceSelection.shape

linhanumerada = np.zeros([len(codeSequenceSelection),dim])
for index,code in enumerate(codeSequenceSelection):
  elemento = np.zeros([1,dim])
  elemento = vetor[(opcodes==code)]
  if (elemento.sum()==0.0):
    elemento = np.zeros([1,dim])
  linhanumerada[index]=elemento

look_back = 5
a, b = create_dataset(linhanumerada,look_back)
a.shape

eval = model.evaluate(x=a, y=b)

"""#Teste do modelo 6 como classe 5"""

cla = 5
#look_back = 5
#dim = 128
#file = '/content/drive/MyDrive/maldrive/arq_todos_opcodes_classe_03_completo.txt'
file = '/content/drive/MyDrive/maldrive/WGRS-2024/sumarios/arq_todos_opcodes_classe_05_sumario.txt'
file_path = file.strip('\n')
lista_opcodes = load_opcodes_classe_txt_token(file_path)

codeSequence = np.array(lista_opcodes)

#tamanho = int(round(len(codeSequence)*80/100,0))
tamanho = int(round(len(codeSequence)*90/100,0))
tamanho = 80000
codeSequence_treino = codeSequence[:tamanho]
codeSequence_teste = codeSequence[(tamanho):88000]

codeSequenceSelection=codeSequence_teste
codeSequenceSelection.shape

linhanumerada = np.zeros([len(codeSequenceSelection),dim])
for index,code in enumerate(codeSequenceSelection):
  elemento = np.zeros([1,dim])
  elemento = vetor[(opcodes==code)]
  if (elemento.sum()==0.0):
    elemento = np.zeros([1,dim])
  linhanumerada[index]=elemento

look_back = 5
a, b = create_dataset(linhanumerada,look_back)
a.shape

eval = model.evaluate(x=a, y=b)

"""#Teste do modelo 6 como classe 7"""

cla = 7
#look_back = 5
#dim = 128
#file = '/content/drive/MyDrive/maldrive/arq_todos_opcodes_classe_03_completo.txt'
file = '/content/drive/MyDrive/maldrive/WGRS-2024/sumarios/arq_todos_opcodes_classe_07_sumario.txt'
file_path = file.strip('\n')
lista_opcodes = load_opcodes_classe_txt_token(file_path)

codeSequence = np.array(lista_opcodes)

#tamanho = int(round(len(codeSequence)*80/100,0))
tamanho = int(round(len(codeSequence)*90/100,0))
tamanho = 80000
codeSequence_treino = codeSequence[:tamanho]
codeSequence_teste = codeSequence[(tamanho):88000]

codeSequenceSelection=codeSequence_teste
codeSequenceSelection.shape

linhanumerada = np.zeros([len(codeSequenceSelection),dim])
for index,code in enumerate(codeSequenceSelection):
  elemento = np.zeros([1,dim])
  elemento = vetor[(opcodes==code)]
  if (elemento.sum()==0.0):
    elemento = np.zeros([1,dim])
  linhanumerada[index]=elemento

look_back = 5
a, b = create_dataset(linhanumerada,look_back)
a.shape

eval = model.evaluate(x=a, y=b)
