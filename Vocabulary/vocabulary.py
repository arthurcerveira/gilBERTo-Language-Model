import json
import os

from bs4 import BeautifulSoup
import requests
from tokenizers import BertWordPieceTokenizer

use_valid_words = False

# Text configuration
punctuation_bert = [
    '!', '"', '$', '%', '&', "'", '(', ':', ')', '=', '@', '#', '|', '{', '}', ';',
    '*', '+', ',', '-', '.', '/', '\\', '[', ']', '<', '>', '?', '^', '~', '`', '_'
]
punctuation = "".join(punctuation_bert)
remove_punctuation = str.maketrans(punctuation, ' ' * len(punctuation))

# Files
tested_words_file = "tested_words_priberam.json"
# https://www.ime.usp.br/~pf/dicios/index.html
br_ispell_file = "br.ispell"

# URLs
url_priberam = "https://dicionario.priberam.org/{word}"


def read_text(path):
    with open(path, "r") as file:
        return file.read()


def word_in_priberam(word):
    retries = 0

    while retries < 5:
        try:
            html = requests.get(url_priberam.format(word=word)).text

            soup = BeautifulSoup(html, "html.parser")

            results = soup.find(id="resultados").text

            # print(results)

            if "Palavra não encontrada." in results:
                return False

            return True
        except:
            retries += 1

    print(f"Could not verify if word '{word}' exists in Priberam")
    return True


def save_tested_words(tested_words):
    # Save tested words to a file
    with open(tested_words_file, "w") as f:
        json.dump(tested_words, f, ensure_ascii=False, indent=4)


def load_tested_words():
    # Load tested words from a file
    if os.path.exists(tested_words_file):
        with open(tested_words_file, "r") as f:
            return json.load(f)

    return dict()


def generate_vocabulary_corpus(text, br_ispell):
    # Filter unwanted characters
    text = text.translate(remove_punctuation)

    # Remove numbers from text
    text = ''.join([i for i in text if not i.isdigit()])

    # Apply text cleaning
    text = text.replace("\n", " ")

    while "  " in text:
        text = text.replace("  ", " ")

    if not use_valid_words:
        return text

    # Else use only valid words
    filtered_words = str()

    # Cache tested words to reduce number of requests
    tested_words = load_tested_words()

    # For each word of the document
    for word in text.split():
        # Verify if the word stem exists in Volp
        word_lower = word.lower()

        # Check if word is in the br.ispell dictionary
        if word_lower in br_ispell:
            word_exists = True

        # Verifiy if the word was tested
        elif word_lower in tested_words:
            word_exists = tested_words[word_lower]

        else:
            print(f"Word {word_lower} not found in br.ispell. Verifying in Priberam...")
            # Verify if the word exists in Priberam
            word_exists = word_in_priberam(word_lower)
            tested_words[word] = word_exists

            if not word_exists:
                print("Word not found:", word)

        # If it exists
        if word_exists:
            filtered_words += word + " "

    save_tested_words(tested_words)

    return filtered_words


def generate_vocabulary(files):
    # Instantiate the BERT tokenizer
    tokenizer = BertWordPieceTokenizer(
        clean_text=True,
        handle_chinese_chars=False,
        strip_accents=False,
        lowercase=False,
    )

    # Train the tokenizer on the corpus
    tokenizer.train(files=files,
                    vocab_size=30_000,
                    min_frequency=2,
                    special_tokens=["[PAD]", "[UNK]",
                                    "[CLS]", "[SEP]", "[MASK]"],
                    limit_alphabet=500,
                    wordpieces_prefix="##",
                    show_progress=True,
                    initial_alphabet=punctuation_bert)

    # Save the tokenizer
    tokenizer.save_model(".")


if __name__ == "__main__":
    with open(br_ispell_file, "r") as f:
        br_ispell = set(f.read().splitlines())

    for file in os.listdir("raw-texts"):
        if file.endswith(".txt"):
            text = read_text(os.path.join("raw-texts", file))

            filtered_text = generate_vocabulary_corpus(text, br_ispell)

            with open(os.path.join("texts", file), "w") as f:
                f.write(filtered_text)

    files = list()

    for file in os.listdir("texts"):
        if file.endswith(".txt"):
            files.append(os.path.join("texts", file))

    generate_vocabulary(files)

    # Test the tokenizer with some text
    tokenizer = BertWordPieceTokenizer('./vocab.txt',
                                       strip_accents=False,
                                       lowercase=False)

    sentence = 'Python é uma linguagem de programação'

    print(f"Encoding {sentence}...")

    encoded = tokenizer.encode(sentence)
    print(encoded.tokens)
