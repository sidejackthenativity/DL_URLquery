import pickle
import DLfunctions as dl
from keras.preprocessing.sequence import pad_sequences

print("What is the name of the hyperparams file?(Local directory):")
params_file_name = input()

a_file = open(params_file_name, "rb")
params = pickle.load(a_file)

w = params["w"]
b = params["b"]

print("What is the questionable URL?")
X_URL = input()
X_URL_list = []

#convert to array
X_URL_list.append([ord(c) for c in X_URL][:-2])
print(X_URL_list)
padded = pad_sequences(X_URL_list,maxlen=3063,padding='post')
print(padded)
X_TEST = padded.T

if(dl.predict(w, b, X_TEST)):
    print("This is probably a bad URL!")
else:
    print("This is probably an okay URL!")
