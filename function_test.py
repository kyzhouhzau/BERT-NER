#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author:zhoukaiyin
"""


def _read_data(input_file):
    """Read a BIO data!"""
    rf = open(input_file, 'r')
    lines = []; words = []; labels = []
    for line in rf:
        word = line.strip().split(' ')[0]
        label = line.strip().split(' ')[-1]
        # here we dont do "DOCSTART" check
        if len(line.strip()) == 0 and words[-1] == '.':
            l = ' '.join([label for label in labels if len(label) > 0])
            w = ' '.join([word for word in words if len(word) > 0])
            lines.append((l, w))
            words = []
            labels = []
        words.append(word)
        labels.append(label)
    return lines

def main():
   lines =  _read_data("./data/train.txt")
   print(lines)
main()
