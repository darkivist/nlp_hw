# Paul Kelly
# DATS 6312 - Homework 2

# E1

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

text_1 = input("Please enter more text:")
text_reordered = str(text_1[1:] + text_1[:1])
print(text_reordered)

# E3

fullname = input("Please enter your name in format firstname middlename lastname:")
name_list = fullname.split(sep=' ')

for name in name_list:
    print([name[:1]])

# E4
#this answer is not correct - revisit
pw = input("Please enter a password (8 characters minimum - 1 uppercase, 1 lowercase, 1 digit:")

conditions = 0
for p in pw:
    if p.isupper():
        conditions += 1
    elif p.islower():
        conditions += 1
    elif p.isdigit():
        conditions += 1
    elif len(pw) >= 8:
        conditions += 1

if conditions < 4:
    print("Password rejected.")
else:
    print("Great password!")

#E5

from collections import defaultdict

string = input("Please enter yet more text:")

count = defaultdict(int)
for char in string:
    count[char] += 1
print(count)

#could also be done with Counter:

from collections import Counter

cnt = Counter(string)
print(cnt.most_common())

#E6

#itertools permutations or combinations?