#convert queries into ascii array

from keras.preprocessing.sequence import pad_sequences
import sys

#grab data from file
file_location = sys.argv[1]
list = []

try:
    file = open(file_location, 'r')
    Lines = file.readlines()

except IOError:
    print('There was an error opening the file!')

    # Strips the newline character
for line in Lines:
    list.append([ord(c) for c in line][:-2]) #remove newline and CR

print(list)


# pad sequence
padded = pad_sequences(list,maxlen=3063,padding='post')
print(padded)
