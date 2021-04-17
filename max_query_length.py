#find max length of query text
import sys

#read file and measure length of the line and update the buffer if it is larger than the buffer
file_location = sys.argv[1]
max_line_length = 0

try:
    file = open(file_location, 'r')
    Lines = file.readlines()

except IOError:
    print('There was an error opening the file!')

    # Strips the newline character
for line in Lines:
    if (line.length() > max_line_length):
        max_line_length = line.length()

print(max_line_length)
