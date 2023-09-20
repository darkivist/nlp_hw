import argparse
import os
import pandas as pd
from collections import defaultdict
# ----------------------------------------------------------------------------------------------------------------------
os.mkdir('Text Feature')
file = open("sample.txt", 'r')
data = file.read().split(sep='.')
count = defaultdict(int)
for sentence in data:
    count[sentence] += 1
print(count)
