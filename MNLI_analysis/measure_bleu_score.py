import sys
import nltk
import numpy as np

def tokenize(sent):
    '''
      each sent is an original/simplified input from MNLI dataset
      @input: string of evaluation text. Ex: "This is a hypothesis!"
      @output: list of strings, each string is a token.
      Ex: ["This", "is", "a", "hypthesis", "!"]
    '''
    return nltk.tokenize.word_tokenize(sent)

def eval_bleu_score(hypothesis, reference):
    '''
       calculate bleu_score between hypothesis and references.
       @input1: hypothesis - list of original tokens returned by def ``tokenize``
       @input2: reference - list of simplified tokens returned by def ``tokenize``
       @output: bleuscore between the original and simplified sentences.
    '''
    return nltk.translate.bleu_score.sentence_bleu([reference], hypothesis)

def main():
    # full path to files with original/simplified sentences, one sentence per line.
    original_data = sys.argv[1]
    simplified_data = sys.argv[2]
    # initialize a list to hold all bleu scores
    bleu_scores = []
    with open(original_data, "r") as original, open(simplified_data, "r") as simplified:
        for (h, r) in zip(original, simplified):
            h_toks = tokenize(h)
            r_toks = tokenize(r)
            score = eval_bleu_score(h_toks, r_toks)
            if score > 1: print("wrong")
            bleu_scores.append(score)
    bleu_scores = np.array(bleu_scores)
    print("BLEU score:", np.mean(bleu_scores), "+-", np.std(bleu_scores))

if __name__ == "__main__":
   main()
