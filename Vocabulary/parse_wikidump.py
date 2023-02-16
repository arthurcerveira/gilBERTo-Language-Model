from contextlib import suppress
import re

import lxml.html
import lxml.html.clean
from wiki_dump_reader import Cleaner, iterate
from ftfy import fix_text


file_count = 0
cleaner = Cleaner()
wiki_dump = "ptwiki-20221001-pages-articles-multistream.xml"
texts_dir = "raw-texts"

for title, text in iterate(wiki_dump):
    text = cleaner.clean_text(text)
    cleaned_text, _ = cleaner.build_links(text)

    # Remove titles
    cleaned_text = re.sub(r'\=+[^(\=)]*\=+', '', cleaned_text)
    # Remove text between curly brackets
    cleaned_text = re.sub(r'\{[^}]*\}', '', cleaned_text)
    # Remove words that starts with a backslash
    cleaned_text = ' '.join(
        word for word in cleaned_text.split(' ') 
        if not word.startswith('\\')
    )

    with suppress(Exception):
        # Remove remaining HTML tags and &nbsp;
        doc = lxml.html.fromstring(cleaned_text)
        cleaner_2 = lxml.html.clean.Cleaner(style=True)
        doc = cleaner_2.clean_html(doc)
        cleaned_text = doc.text_content()

    if (cleaned_text.upper().startswith('REDIRECIONAMENTO') or
        cleaned_text.upper().startswith('REDIRECT')):
        continue

    # Remove mojibakes
    cleaned_text = fix_text(cleaned_text)

    # Write to file
    with open(f"{texts_dir}/{file_count}.txt", "w", encoding="utf-8") as file:
        file.write(cleaned_text)

    file_count += 1

    if file_count == 200_000:
        break
