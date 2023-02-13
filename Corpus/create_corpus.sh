#!/bin/bash

source venv/bin/activate

for i in {1..170}; do
    if [[ -f "data/pt_meta_part_$i.txt" ]]; then
        echo "File pt_meta_part_$i.txt already exists. Skipping..."
        continue
    fi

    echo "Processing raw-data/pt_meta_part_$i.jsonl.gz"
    gzip -dk raw-data/pt_meta_part_$i.jsonl.gz
    ./oscar-tools/target/release/oscar-tools v2 extract-tags raw-data/pt_meta_part_$i.jsonl pt_meta_part_$i.jsonl
    ./oscar-tools/target/release/oscar-tools v2 extract-text pt_meta_part_$i.jsonl pt_meta_part_$i.txt
    # ./oscar-tools/target/release/oscar-tools v1 dedup pt_meta_part_$i.txt pt_meta_part_dedup_$i.txt
    # python text_cleaning.py pt_meta_part_dedup_$i.txt data/pt_meta_part_$i.txt
    python text_cleaning.py pt_meta_part_$i.txt data/pt_meta_part_$i.txt

    rm raw-data/pt_meta_part_$i.jsonl
    rm pt_meta_part_$i.jsonl
    rm pt_meta_part_$i.txt
    # rm pt_meta_part_dedup_$i.txt
done

echo "Done!"
