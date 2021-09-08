# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import sys
from access.preprocessors import get_preprocessors
from access.resources.prepare import prepare_models
from access.simplifiers import get_fairseq_simplifier, get_preprocessed_simplifier
from access.text import word_tokenize
from access.utils.helpers import yield_lines, write_lines, get_temp_filepath, mute
import os

def batch(iterable, data_size, s, n=1):
        l = data_size
        for ndx in range(s, l, n):
            yield iterable[ndx:min(ndx + n, l)]

if __name__ == '__main__':
    # Usage: python generate.py < my_file.complex
    # Read from stdin
    data = open(sys.argv[1], "r").readlines() # Hoang modfied this to batching.
    count = 0 # to determine the written filename
    batch_size = 50
    data_size = len(data)
    # Hoang added here so that we only need to create temporary files for pred and source once
    pred_filepath = get_temp_filepath()
    source_filepath = get_temp_filepath()
    best_model_dir = prepare_models()
    recommended_preprocessors_kwargs = {
        'LengthRatioPreprocessor': {'target_ratio': 0.95},
        'LevenshteinPreprocessor': {'target_ratio': 0.75},
        'WordRankRatioPreprocessor': {'target_ratio': 0.75},
        'SentencePiecePreprocessor': {'vocab_size': 10000},
    }
    preprocessors = get_preprocessors(recommended_preprocessors_kwargs)
    simplifier = get_fairseq_simplifier(best_model_dir, beam=8)
    simplifier = get_preprocessed_simplifier(simplifier, preprocessors=preprocessors)
    output_file = "/data1/home/vnhh/mnli_simplified/mnli_simplified_sentences/dev_matched_sent2_sim.txt"
    try:
        s = len(open(output_file, "r").readlines())
    except:
        s = 0 
    runs = 0
    with open(output_file, "a") as output:
        if s >= data_size:
            exit(0)
        for data_batch in batch(data, data_size, s, batch_size): # 20 here is the batch size
            count += 1
            write_lines([word_tokenize(line) for line in data_batch], source_filepath)
            # Simplify
            with mute():
                simplifier(source_filepath, pred_filepath)
            # Hoang modified here to write the outputs to file
            for line in yield_lines(pred_filepath):
                output.write(line + "\n")
            # print("Simplified:", min(100, count* batch_size * 100 / data_size), "%")
            runs += 1
            if runs % 50 == 0:
                print("Completed:", runs, "batches")
