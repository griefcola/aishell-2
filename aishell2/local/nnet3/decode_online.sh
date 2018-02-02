#!/bin/bash

# This script simulate online decoding for nnet3

set -e

echo "$0 $@"  # Print the command line for logging

. ./cmd.sh
[ -f path.sh ] && . ./path.sh;
. ./utils/parse_options.sh

if [ $# -ne 3 ]; then
  echo "Usage: $0 <nnet-dir> <graph-dir> <data-dir>"
  echo "e.g.: $0 exp/nnet3/tdnn exp/tri3/graph data/test"
  exit 1;
fi

dir=$1
graphdir=$2
datadir=$3

# prepare online decoding configurations
steps/online/nnet3/prepare_online_decoding.sh \
  --feature-type fbank --fbank-config conf/fbank.16k.conf \
  data/lang "$dir" ${dir}_online || exit 1;

# do decoding
num_jobs=`cat $datadir/utt2spk|cut -d' ' -f2|sort -u|wc -l`
steps/online/nnet3/decode.sh --config conf/decode.config \
  --cmd "$decode_cmd" --nj $num_jobs --per-utt true \
  "$graphdir" "$datadir" \
    ${dir}_online/decode || exit 1;

wait;
exit 0;
