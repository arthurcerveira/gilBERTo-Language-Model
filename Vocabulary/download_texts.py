import glob
from datetime import datetime

from bs4 import BeautifulSoup
import requests


with open("titles.txt", "r", encoding="utf-8") as f:
    titles = set(f.read().splitlines())

texts_dir = "raw-texts"
random_wikipedia = "https://pt.wikipedia.org/wiki/Especial:Aleat%C3%B3ria" 


def save_titles(titles):
    with open("titles.txt", "w", encoding="utf-8") as file:
        for title in titles:
            file.write(title + "\n")


def wikipedia_article(titles):
    retries = 0

    while retries < 5:
        try:
            response = requests.get(random_wikipedia).text
            
            return response
        except:
            retries += 1

    print(f"Could not request from Wikipedia. Exiting...")
    save_titles(titles)

    exit(0)


def count_files(directory):
    return len(glob.glob1(directory, "*.txt"))

def download_texts(n):
    for index in range(n):
        html = wikipedia_article(titles)

        soup = BeautifulSoup(html, "html.parser")

        # Get title of article
        title = soup.find(id="firstHeading").text

        # Guarantee to not repeat articles
        while title in titles:
            html = wikipedia_article(titles)

            soup = BeautifulSoup(html, "html.parser")

            # Get title of article
            title = soup.find(id="firstHeading").text

        content = soup.find(id="mw-content-text").find(class_="mw-parser-output")
        
        paragraphs = content.find_all("p")

        text = str()

        for paragraph in paragraphs:
            text += paragraph.text

        file_name = f"{texts_dir}/{number_of_files + index}.txt"

        with open(file_name, "w", encoding="utf-8") as file:
            file.write(text)

        titles.add(title)

        if (number_of_files + index) % 200 == 0:
            dt_string = datetime.now().strftime("[%H:%M]")

            print(f"{dt_string} Downloaded {number_of_files + index} files...")



if __name__ == "__main__":
    number_of_files = count_files(texts_dir)

    dt_string = datetime.now().strftime("[%H:%M]")
    print(f"{dt_string} Starting on file {number_of_files}...")

    try:
        download_texts(30000)
    except Exception as e:
        print("There was an error while downloading the texts. Exiting...")
        print(e)
        save_titles(titles)

        exit(0)

    number_of_files = count_files(texts_dir)

    dt_string = datetime.now().strftime("[%H:%M]")
    print(f"{dt_string} Ending with {number_of_files} files...")

    save_titles(titles)
