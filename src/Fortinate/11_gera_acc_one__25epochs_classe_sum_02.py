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
#from keras.layers import LSTM, MaxPooling1D, Conv1D, Flatten
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
dict_all=load_dict_from_file_gz("/content/drive/MyDrive/maldrive/WGRS-2024/dicionarios/dict_opcodes_one_sum_02.dic.gz")
#opcodes=np.array( tuple(dict_all.keys()) )
opcodes=np.array( tuple(dict_all['opcode'].values()) )

#dict_all['opcode'].keys()

#vetor.shape

#opcodes.shape

#dict_all['opcode'].values()

#dict_all

opcodes

#opcodes_lista[0,64]
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

vetor.shape

#Criação de matriz de dados mapeando X->Y anterior->proximo
def create_dataset(elemento, look_back=1):
    dataX, dataY = [], []
    for i in range(len(elemento)-look_back):
        a = elemento[i:(i+look_back)]
        dataX.append(a)
        dataY.append(elemento[i + look_back])
    return np.array(dataX), np.array(dataY)

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
#Classe 02 - Sumario
"""

#dim=128
dim=64

units = dim # dimensão de vetor
# a ideia de criar memory foi para que a mattriz a não ultrapassasse o tamanho máximo de memória
# 100 -> 100Mb aprox., teste empírico. calculamos tam_memory por formula aproximada.
memory=100*3 # quantidade em Mb que pode ocupar cada partição de a
tam_memory=int(memory*(200/units)*10000/76)

tam_memory

"""# Ler base de teste classe 02"""

label=3 # modelo que será gerado
dd=3
look_back = 5
#look_back = 90
epoch = 100 # problema de tempo de execução para famílias com muitas amostras.
#epoch = 1 # problema de tempo de execução para famílias com muitas amostras.
units = dim # dimensão de vetor
batch = 100

cla = 2
look_back = 5
#dim = 128
dim = 64
#file = '/content/drive/MyDrive/maldrive/arq_todos_opcodes_classe_03_completo.txt'
file = '/content/drive/MyDrive/maldrive/WGRS-2024/sumarios/arq_todos_opcodes_classe_02_sumario.txt'
file_path = file.strip('\n')
lista_opcodes = load_opcodes_classe_txt_token(file_path)

codeSequence = np.array(lista_opcodes)

#tamanho = int(round(len(codeSequence)*80/100,0))
tamanho = int(round(len(codeSequence)*90/100,0))
#tamanho = 80000
#codeSequence_treino = codeSequence[:tamanho]
#codeSequence_teste = codeSequence[(tamanho+100):80100+100]

"""# Ler os modelos"""

ep=25

label=1
model01 = load_model("/content/drive/MyDrive/maldrive/WGRS-2024/modelo100/model_LSTM_" + str(label) +"_"+str(ep)+ ".h5") # ler modelo de cada epoch
label=2
model02 = load_model("/content/drive/MyDrive/maldrive/WGRS-2024/modelo100/model_LSTM_" + str(label) +"_"+str(ep)+ ".h5") # ler modelo de cada epoch
label=3
model03 = load_model("/content/drive/MyDrive/maldrive/WGRS-2024/modelo100/model_LSTM_" + str(label) +"_"+str(ep)+ ".h5") # ler modelo de cada epoch
label=4
model04 = load_model("/content/drive/MyDrive/maldrive/WGRS-2024/modelo100/model_LSTM_" + str(label) +"_"+str(ep)+ ".h5") # ler modelo de cada epoch
label=5
model05 = load_model("/content/drive/MyDrive/maldrive/WGRS-2024/modelo100/model_LSTM_" + str(label) +"_"+str(ep)+ ".h5") # ler modelo de cada epoch
label=6
model06 = load_model("/content/drive/MyDrive/maldrive/WGRS-2024/modelo100/model_LSTM_" + str(label) +"_"+str(ep)+ ".h5") # ler modelo de cada epoch
label=7
model07 = load_model("/content/drive/MyDrive/maldrive/WGRS-2024/modelo100/model_LSTM_" + str(label) +"_"+str(ep)+ ".h5") # ler modelo de cada epoch

#model.summary()

cont = 0
  inicio = 450000+(cont*300)
  final = 450300+(cont*300)
  print(inicio)
  print(final)

cont = 1
inicio = 450000+(cont*300)
final = 450300+(cont*300)
print(inicio)
print(final)

cont = 2
inicio = 450000+(cont*300)
final = 450300+(cont*300)
print(inicio)
print(final)

acertos1=0
acertos2=0
acertos3=0
acertos4=0
acertos5=0
acertos6=0
acertos7=0
total_acur = []
cont = 0
while cont < 1000:
  print("-------------------")
  print(cont)
#  inicio = 450000+(cont*100)
#  final = 450100+(cont*100)
#  inicio = 450000+(cont*200)
#  final = 450200+(cont*200)
#  inicio = 450000+(cont*150)
#  final = 450150+(cont*150)
#  inicio = 450000+(cont*130)
#  final = 450130+(cont*130)
  #inicio = 450000+(cont*300)
  #final = 450300+(cont*300)

  #inicio = 130000+(cont*6)
  #final  = 130006+(cont*6)

  #inicio = 450000+(cont*7)
  #final  = 450007+(cont*7)

  #inicio = 80000+(cont*300)
  #final  = 80300+(cont*300)

#  inicio = 80000+(cont*360)
#  final  = 80360+(cont*360)


  inicio = 80000+(cont*390)
  final  = 80390+(cont*390)


 # inicio = 80000+(cont)
 # final  = inicio+360


#  inicio =    0+(cont*300)
#  final  = 300+(cont*300)

#  inicio =    0+(cont*360)
#  final  = 360+(cont*360)


  #codeSequence_treino = codeSequence[:tamanho]
#  codeSequence_teste = codeSequence[(tamanho+(cont*15)):450015+(cont*15)]
  codeSequence_teste = codeSequence[inicio:final]
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

  acur = []


  #Teste da classe 1 com modelo 1
  eval = model01.evaluate(x=a, y=b)
  acur.append(eval[1])

  #Teste da classe 1 com modelo 2
  eval = model02.evaluate(x=a, y=b)
  acur.append(eval[1])

  #Teste da classe 1 com modelo 3
  eval = model03.evaluate(x=a, y=b)
  acur.append(eval[1])

  #Teste da classe 1 com modelo 4
  eval = model04.evaluate(x=a, y=b)
  acur.append(eval[1])

  #Teste da classe 1 com modelo 5
  eval = model05.evaluate(x=a, y=b)
  acur.append(eval[1])

  #Teste da classe 1 com modelo 6
  eval = model06.evaluate(x=a, y=b)
  acur.append(eval[1])

  #Teste da classe 1 com modelo 7
  eval = model07.evaluate(x=a, y=b)
  acur.append(eval[1])

  acur=array(acur)

  print(argmax(acur))

  if argmax(acur)==1:
    acertos1 = acertos1 + 1

  if argmax(acur)==2:
    acertos2 = acertos2 + 1

  if argmax(acur)==3:
    acertos3 = acertos3 + 1

  if argmax(acur)==4:
    acertos4 = acertos4 + 1

  if argmax(acur)==5:
    acertos5 = acertos5 + 1

  if argmax(acur)==6:
    acertos6 = acertos6 + 1

  if argmax(acur)==7:
    acertos7 = acertos7 + 1

  cont = cont + 1
  print(format(argmax(acur), "b"))
  total_acur.append(acur)


#array_total_acur=array(total_acur)
#array_acc = np.array(array_total_acur)
#np.save('/content/drive/MyDrive/maldrive/experimentos/experimento01/acuracias/acur_classe1_total-360.npy', array_acc)


#60 - 10000 -v6 - continuo

print(acertos1)
print(acertos2)
print(acertos3)
print(acertos4)
print(acertos5)
print(acertos6)
print(acertos7)
