#!/bin/bash

path_inputs="/work/lilu/AttConvLSTM/input/lstm/"
path_outputs="/work/lilu/AttConvLSTM/output/"
epoch=40
cd '/work/lilu/AttConvLSTM/'

python ./src/main.py --path_inputs ${path_inputs} \
                     --path_outputs ${path_outputs} \
                     --num_job 200 \
                     --batch_size 128 \
                     --epoch ${epoch} \
                     --split_ratio 0.2
