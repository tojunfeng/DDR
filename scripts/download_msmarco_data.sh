#!/usr/bin/env bash
# copy from https://github.com/microsoft/unilm/blob/master/simlm/scripts/download_msmarco_data.sh

set -x
set -e

DIR="$( cd "$( dirname "$0" )" && cd .. && pwd )"
echo "working directory: ${DIR}"

mkdir -p data/

MSMARCO_BM25="msmarco_bm25_official.zip"
if [ ! -e data/$MSMARCO_BM25 ]; then
  wget -O data/${MSMARCO_BM25} https://huggingface.co/datasets/intfloat/simlm-msmarco/resolve/main/${MSMARCO_BM25}
  unzip data/${MSMARCO_BM25} -d dataset/
fi