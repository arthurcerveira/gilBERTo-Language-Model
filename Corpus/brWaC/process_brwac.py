import os
from string import punctuation
import re
from datetime import datetime

import chardet
import pandas as pd
from ftfy import fix_text


# regex for question mark between letters
regex = re.compile(r'(?<=[a-zA-Z])\?(?=[a-zA-Z])')


def brwac():
    brwac = open("data/brwac.vert")
    brwac_cleaned = open("data/brWaC.txt", 'w')
    
    new_sentence = str()
    sentences = 0

    status = str()
    skip_space = False

    for line in brwac:
        if sentences % 100_000 == 0:
            if status != f"{sentences} paragraphs processed...":
                print(f"[{datetime.now():%H:%M}] {sentences:,} paragraphs processed...")
                status = f"{sentences} paragraphs processed..."

        if line.startswith("<p>"):
            new_sentence = str()
            continue

        if line.startswith("</p>"):
            sentence_fixed = fix_text(new_sentence)

            sentence_fixed = sentence_fixed.strip()

            while ('  ') in sentence_fixed:
                sentence_fixed = sentence_fixed.replace('  ', ' ')

            if len(sentence_fixed) == 0:
                continue

            if regex.search(sentence_fixed):
                continue

            # verify if every character is punctuation
            if all(char in punctuation for char in sentence_fixed):
                continue

            brwac_cleaned.write(sentence_fixed + '\n')

            sentences += 1
            continue

        if line.startswith("<g/>"):
            skip_space = True
            continue

        if line.startswith("<"):
            continue

        if skip_space:
            new_sentence += line.strip().replace('\n', '')
            skip_space = False
            continue

        new_sentence += ' ' + line.replace('\n', '')

    brwac.close()
    brwac_cleaned.close()


if __name__ == "__main__":
    brwac()
