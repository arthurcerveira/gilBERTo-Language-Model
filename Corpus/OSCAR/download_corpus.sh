#!/bin/bash

source venv/bin/activate

USER = "USER"
PASSWORD = "PASSWORD"
PT_META_URL = "URL"

echo "Downloading OSCAR corpus from $PT_META_URL..."

dodc \
 --user $USER \
 --password $PASSWORD \ 
 --base_url $PT_META_URL
 --out .

echo "Done!"
