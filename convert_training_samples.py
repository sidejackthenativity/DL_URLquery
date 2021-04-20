#convert queries into ascii array

from keras.preprocessing.sequence import pad_sequences
import sys
import numpy as np
from numpy import savetxt
from numpy import save

def do_things(file_location, column_value):
    #grab data from file
    list = []

    try:
        file = open(file_location, 'r')
        Lines = file.readlines()

    except IOError:
        print('There was an error opening the file!')

        # Strips the newline character
    for line in Lines:
        list.append([ord(c) for c in line][:-2]) #remove newline and CR


    # pad sequence
    padded = pad_sequences(list,maxlen=3063,padding='post')
    #padded: 3 x 3063
    N = padded.shape[0]
    #now add the value 0 or 1 as declared in argument 2
    if (column_value=='0'):
        new_column = np.zeros((N,1)) #new_column: 3 x 1
    elif (column_value=='1'):
        new_column = np.ones((N,1)) #new_column: 3 x 1
    #print(padded.shape)
    #print(new_column.shape)
    an_array = np.append(padded, new_column, axis=1)

    return an_array



#__main__
if (len(sys.argv)==6):
    file_name = sys.argv[5]
    array1 = do_things(sys.argv[1],sys.argv[2])
    array2 = do_things(sys.argv[3],sys.argv[4])
    an_array = np.vstack((array1,array2))
    #savetxt(file_name, an_array, delimiter=',')
    save(file_name,an_array)
else:
    print("Proper format is: python3 convert_training_samples.py <data_source1> <value as a string in last column> <data_source2> <value as a string in last column> <destination file>")
