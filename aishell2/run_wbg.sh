#!/bin/bash

# Copyright 2017 Beijing Shell Shell Tech. Co. Ltd. (Authors: Hui Bu)
#           2017 Jiayu Du
#           2017 Xingyu Na
#           2017 Bengu Wu
#           2017 Hao Zheng
# Apache 2.0

# This is a shell script, but it's recommended that you run the commands one by
# one by copying and pasting into the shell.
# Caution: some of the graph creation steps use quite a bit of memory, so you
# should run this on a machine that has sufficient memory.

#data=/home/zhenghao/kaldi/egs/aishell/s5/aishell-data
#data_url=www.openslr.org/resources/33
#

export LD_LIBRARY_PATH=/data/wubengu/kaldi-master/tools/openfst/lib:$LD_LIBRARY_PATH
. ./cmd.sh
. ./path.sh
#
#local/download_and_untar.sh $data $data_url data_aishell || exit 1;
#local/download_and_untar.sh $data $data_url resource_aishell || exit 1;
#
## Lexicon Preparation,
#local/aishell_prepare_dict.sh $data/resource_aishell || exit 1;
#
## Data Preparation,
#local/aishell_data_prep.sh $data/data_aishell/wav $data/data_aishell/transcript || exit 1;
#
## Phone Sets, questions, L compilation
#utils/prepare_lang.sh --position-dependent-phones false data/local/dict \
#    "<SPOKEN_NOISE>" data/local/lang data/lang || exit 1;
#
## LM training
#local/aishell_train_lms.sh || exit 1;
#
## G compilation, check LG composition
#utils/format_lm.sh data/lang data/local/lm/3gram-mincount/lm_unpruned.gz \
#    data/local/dict/lexicon.txt data/lang_test || exit 1;
#exit
# Now make MFCC plus pitch features.
# mfccdir should be some place with a largish disk where you
# want to store MFCC features.
nj=20
mfccdir=mfcc
for x in train dev test; do
  steps/make_mfcc.sh --cmd "$train_cmd" --nj $nj data/$x exp/make_mfcc/$x $mfccdir || exit 1;
  steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x $mfccdir || exit 1;
  utils/fix_data_dir.sh data/$x || exit 1;
done

utils/subset_data_dir.sh data/train 30000 data/train_30k
utils/subset_data_dir.sh data/train 100000 data/train_100k

steps/train_mono.sh --cmd "$train_cmd" --nj $nj \
  data/train_30k data/lang exp/mono || exit 1;

# Monophone decoding
#utils/mkgraph.sh data/lang_test exp/mono exp/mono/graph || exit 1;
#steps/decode.sh --cmd "$decode_cmd" --config conf/decode.config --nj $nj \
#  exp/mono/graph data/dev exp/mono/decode_dev
#steps/decode.sh --cmd "$decode_cmd" --config conf/decode.config --nj $nj \
#  exp/mono/graph data/test exp/mono/decode_test

# Get alignments from monophone system.
steps/align_si.sh --cmd "$train_cmd" --nj $nj \
  data/train_100k data/lang exp/mono exp/mono_ali || exit 1;

# train tri1 [first triphone pass]
steps/train_deltas.sh --cmd "$train_cmd" \
 2500 20000 data/train_100k data/lang exp/mono_ali exp/tri1 || exit 1;

# decode tri1
#utils/mkgraph.sh data/lang_test exp/tri1 exp/tri1/graph || exit 1;
#steps/decode.sh --cmd "$decode_cmd" --config conf/decode.config --nj ${nj} \
#  exp/tri1/graph data/dev exp/tri1/decode_dev
#steps/decode.sh --cmd "$decode_cmd" --config conf/decode.config --nj ${nj} \
#  exp/tri1/graph data/test exp/tri1/decode_test

# align tri1
steps/align_si.sh --cmd "$train_cmd" --nj $nj \
  data/train data/lang exp/tri1 exp/tri1_ali || exit 1;

# train tri2 [delta+delta-deltas]
steps/train_deltas.sh --cmd "$train_cmd" \
 5000 40000 data/train data/lang exp/tri1_ali exp/tri2 || exit 1;

# decode tri2
#utils/mkgraph.sh data/lang_test exp/tri2 exp/tri2/graph
#steps/decode.sh --cmd "$decode_cmd" --config conf/decode.config --nj ${nj} \
#  exp/tri2/graph data/dev exp/tri2/decode_dev
#steps/decode.sh --cmd "$decode_cmd" --config conf/decode.config --nj ${nj} \
#  exp/tri2/graph data/test exp/tri2/decode_test

# train and decode tri2b [LDA+MLLT]
steps/align_si.sh --cmd "$train_cmd" --nj $nj \
  data/train data/lang exp/tri2 exp/tri2_ali || exit 1;

# Train tri3a, which is LDA+MLLT,
steps/train_lda_mllt.sh --cmd "$train_cmd" \
 7500 60000 data/train data/lang exp/tri2_ali exp/tri3 || exit 1;

utils/mkgraph.sh data/lang_test exp/tri3 exp/tri3/graph || exit 1;
steps/decode.sh --cmd "$decode_cmd" --nj ${nj} --config conf/decode.config \
  exp/tri3/graph data/dev exp/tri3/decode_dev
steps/decode.sh --cmd "$decode_cmd" --nj ${nj} --config conf/decode.config \
  exp/tri3/graph data/test exp/tri3/decode_test

# From now, we start building a more serious system (with SAT), and we'll
# do the alignment with fMLLR.

steps/align_fmllr.sh --cmd "$train_cmd" --nj $nj \
  data/train data/lang exp/tri3 exp/tri3_ali || exit 1;

steps/train_sat.sh --cmd "$train_cmd" \
  10000 100000 data/train data/lang exp/tri3_ali exp/tri4 || exit 1;


utils/mkgraph.sh data/lang_test exp/tri4 exp/tri4/graph
steps/decode_fmllr.sh --cmd "$decode_cmd" --nj ${nj} --config conf/decode.config \
  exp/tri4/graph data/dev exp/tri4/decode_dev
steps/decode_fmllr.sh --cmd "$decode_cmd" --nj ${nj} --config conf/decode.config \
  exp/tri4/graph data/test exp/tri4/decode_test
#
#steps/align_fmllr.sh --cmd "$train_cmd" --nj ${nj} \
#  data/train data/lang exp/tri4 exp/tri4_ali

#steps/align_fmllr.sh --cmd "$train_cmd" --nj ${nj} \
#  data/dev data/lang exp/tri4 exp/tri4_ali_cv

exit 0
# nnet3
local/nnet3/run_tdnn.sh

# chain
local/chain/run_tdnn.sh

# getting results (see RESULTS file)
for x in exp/*/decode_test; do [ -d $x ] && grep WER $x/cer_* | utils/best_wer.sh; done 2>/dev/null

exit 0;
