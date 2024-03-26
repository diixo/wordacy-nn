
import pathlib
import random


text_file = pathlib.Path("spa-eng/spa.txt")

if text_file.exists():
        fd = open(text_file, 'r', encoding='utf-8')
        lines = fd.read().split("\n")[:-1]

text_pairs = []
for line in lines:
    eng, spa = line.split("\t")
    spa = "[start] " + spa + " [end]"
    text_pairs.append((eng, spa))

for _ in range(5):
    print(random.choice(text_pairs))

#################################################################
