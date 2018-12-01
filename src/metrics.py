import collections
import math

__all__ = ['bleu_one', 'bleu']


def n_gram_counter(sentences, n):
    return collections.Counter(tuple(sentences[i:i + n])
                               for i in range(len(sentences) - n + 1))


def n_gram_modified_precision(candidate, reference, n):
    if len(candidate) < n:
        return 1.0
    candidate_counter = n_gram_counter(candidate, n)
    reference_counter = n_gram_counter(reference, n)
    occurrence = 0
    total = 0
    for n_gram, cc in candidate_counter.items():
        rc = reference_counter.get(n_gram, 0)
        occurrence += min(cc, rc)
        total += cc
    return occurrence / total


def bleu_one(candidate, reference, max_n=4):
    p_list = [n_gram_modified_precision(candidate, reference, n) for n in range(1, max_n + 1)]
    if len(candidate) == 0 or min(p_list) == 0.0:
        return 0.0
    log_b = min(1 - len(reference) / len(candidate), 0) + sum(math.log(p) for p in p_list)
    return math.exp(log_b)


def bleu(candidate_list, reference_list, max_n=4):
    b_list = [bleu_one(candidate, reference, max_n)
              for candidate, reference in zip(candidate_list, reference_list)]
    return sum(b_list) / len(b_list)
