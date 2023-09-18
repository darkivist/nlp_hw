# Paul Kelly
# DATS 6312 - Homework 2
# 9/17/23

# E1
# Write a python script that reads a string from the user input and print the following
# i. Number of uppercase letters in the string.
# ii. Number of lowercase letters in the string
# iii. Number of digits in the string
# iv. Number of whitespace characters in the string

# source: https://www.geeksforgeeks.org/isupper-islower-lower-upper-python-applications/

text = input("Please enter some text:")

upper = 0
lower = 0
number = 0
whitespace = 0

for t in text:
    if t.isupper():
        upper += 1
    elif t.islower():
        lower += 1
    elif t.isdigit():
        number += 1
    elif t.isspace():
        whitespace += 1

print("Uppercase -", upper)
print("Lowercase -", lower)
print("Digits -", number)
print("Whitespace -", whitespace)

# E2
# Write a python script that accepts a string then create a new string by shifting one position to left.

text_1 = input("Please enter more text:")
text_reordered = str(text_1[1:] + text_1[:1])
print(text_reordered)

# E3
# Write a python script that a user input his name and program display its initials.

fullname = input("Please enter your name in format firstname middlename lastname:")
name_list = fullname.split(sep=' ')

for name in name_list:
    print([name[:1]])

# E4
# Write a python script that accepts a string to setup a passwords.
# E4
# Write a python script that accepts a string to setup a passwords.
# this answer is not correct - revisit

pw = input("Please enter a password (8 characters minimum - 1 uppercase, 1 lowercase, 1 digit:")

upper = 0
lower = 0
digit = 0
length = 0

for p in pw:
    if p.isupper():
        upper += 1
    elif p.islower():
        lower += 1
    elif p.isdigit():
        digit += 1
    elif len(pw) >= 8:
        length += 1

if upper >= 1 and lower >= 1 and digit >= 1 and length >= 1:
    print("Great password!")
else:
    print("Password rejected.")

# E5
# Write a python script that reads a given string character by character and count the repeated
# characters then store it by length of those character(s)

from collections import defaultdict

string = input("Please enter yet more text:")

count = defaultdict(int)
for char in string:
    count[char] += 1
print(count)

# could also be done with Counter:

from collections import Counter

cnt = Counter(string)
print(cnt.most_common())

# E6
# Write a python script to find all lower and upper case combinations of a given string.

import itertools

S1 = "string"

S3 = itertools.permutations(S1)
for each in S3:
    print(each)

# or (source: https://stackoverflow.com/questions/69782189/permutation-lowercase-and-uppercase-in-python)

from itertools import product

string_1 = input("Please enter yet more text:")

cases = zip(*[string_1, string_1.swapcase()])
for permutation in product(*cases):
    print("".join(permutation))

# E7
# Write a python script that:

file = open("test.txt", 'r')
# i. Reads first n lines of a file
print(file.readlines()[0:1])
file.seek(0)
# iii. Counts the number of lines in a text file
print(len(file.readlines()))
file.seek(0)
data = file.read().split()
# ii. Finds the longest words
print(max(data, key=len))
# iv. Counts the frequency of words in a file
count = defaultdict(int)
for word in data:
    count[word] += 1
print(count)