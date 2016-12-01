#!/usr/bin/env bash

myshuf() {
  perl -MList::Util=shuffle -e 'print shuffle(<>);' "$@";
}

normalize_text() {
  tr '[:upper:]' '[:lower:]' | sed -e 's/^/__label__/g' | \
    sed -e "s/'/ ' /g" -e 's/"//g' -e 's/\./ \. /g' -e 's/<br \/>/ /g' \
        -e 's/,/ , /g' -e 's/(/ ( /g' -e 's/)/ ) /g' -e 's/\!/ \! /g' \
        -e 's/\?/ \? /g' -e 's/\;/ /g' -e 's/\:/ /g' | tr -s " " | myshuf
}

DATASET=(
  ag_news
  sogou_news
  dbpedia
  yelp_review_polarity
  yelp_review_full
  yahoo_answers
  amazon_review_full
  amazon_review_polarity
)

# These learning rates were chosen by validation on a subset of the training set.
LR=( 0.25 0.5 0.5 0.1 0.1 0.1 0.05 0.05 )

RESULTDIR=result
DATADIR=data

mkdir -p "${RESULTDIR}"
mkdir -p "${DATADIR}"

for i in {0..7}
do
  echo "Downloading dataset ${DATASET[i]}"
  if [ ! -f "${DATADIR}/${DATASET[i]}.train" ]
  then
    tar -xzvf "${DATADIR}/${DATASET[i]}_csv.tar.gz" -C "${DATADIR}"
    cat "${DATADIR}/${DATASET[i]}_csv/train.csv" | normalize_text > "${DATADIR}/${DATASET[i]}.train"
    cat "${DATADIR}/${DATASET[i]}_csv/test.csv" | normalize_text > "${DATADIR}/${DATASET[i]}.test"
  fi
done

make

for i in {0..7}
do
  echo "Working on dataset ${DATASET[i]}"
  th train.lua -input "${DATADIR}/${DATASET[i]}.train" \
    -output "${RESULTDIR}/${DATASET[i]}" -dim 10 -lr "${LR[i]}" -wordNgrams 2 -epoch 5
  th test.lua -model "${RESULTDIR}/${DATASET[i]}.bin" \
    -test "${DATADIR}/${DATASET[i]}.test"
done
