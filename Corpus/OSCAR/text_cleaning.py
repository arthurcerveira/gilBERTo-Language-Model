import argparse
import re
from datetime import datetime

from ftfy import fix_text


clean_tags = re.compile('<.*?>')


def clean_text(file_path, output_path, n_lines=None):
    print(f"[{datetime.now():%H:%M}] Cleaning text file: {file_path}")
    text_file = open(file_path)
    text_file_cleaned = open(output_path, 'w')

    # If lines is None, read all lines
    if n_lines is None:
        lines = text_file.readlines()
    else:
        lines = [text_file.readline() for _ in range(n_lines)]

    for line in lines:
        line_strip = line.strip()
        line_fixed = fix_text(line_strip)
        line_cleaned = re.sub(clean_tags, '', line_fixed)

        text_file_cleaned.write(line_cleaned + '\n')

    text_file.close()
    text_file_cleaned.close()

    print(f"[{datetime.now():%H:%M}] Cleaned text file saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "file_path",
        help="Path to the text file to be cleaned",
        type=str
    )

    parser.add_argument(
        "output_path",
        help="Path to the output file",
        type=str
    )

    parser.add_argument(
        "-l", "--lines",
        help="Number of lines to be read from the text file",
        type=int
    )

    args = parser.parse_args()

    clean_text(args.file_path, args.output_path, args.lines)
