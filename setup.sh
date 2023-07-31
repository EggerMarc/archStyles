#! /bin/bash

ls;
. ./venv/bin/activate;
download=false
images_folder="/images/"

while getopts "d" option; do
  case "$option" in
    d)
      download=true
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
  esac
done

# Check if the -d option was provided
if $download; then
    cd $images_folder|| exit;
    echo "Running download functionality...";
    gallery-dl --range 1-10 https://www.pinterest.it/historiasdecasa/sala-de-estar-living-room/;
    #gallery-dl --range 1-500 https://www.pinterest.it/casavoguebrasil/sala-de-estar/;
    #gallery-dl --range 1-500 https://www.pinterest.it/luxxu/living-room-decoration/;
    #gallery-dl --range 1-500 https://www.pinterest.it/eggermarc/interior-design/;
    #gallery-dl --range 1-500 https://www.pinterest.it/Designeddecor/living-room-decor/;
    #gallery-dl --range 1-500 https://www.pinterest.it/decoratedlifer/living-room-decor/;
    #gallery-dl --range 1-500 https://www.pinterest.it/susanlori/living-room-decor/;
    #gallery-dl --range https://www.pinterest.it/southernhooch/rustic-living-rooms/;
    find . -mindepth 2 -type f -print -exec mv {} . \;
    find . -type d -empty -delete;
    find . -type f ! -iname "*.jpg" ! -iname "*.png" -delete
    ls -v | cat -n | while read n f; do mv -n "$f" "$n.jpg"; done;
fi