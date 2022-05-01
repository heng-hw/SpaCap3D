#!/usr/bin/env python
#
# File Name : bleu.py
#
# Description : Wrapper for BLEU scorer.
#
# Creation Date : 06-01-2015
# Last Modified : Thu 19 Mar 2015 09:13:28 PM PDT
# Authors : Hao Fang <hfang@uw.edu> and Tsung-Yi Lin <tl483@cornell.edu>

from .bleu_scorer import BleuScorer


class Bleu:
    def __init__(self, n=4):
        # default compute Blue score up to 4
        self._n = n
        self._hypo_for_image = {}
        self.ref_for_image = {}

    def compute_score(self, gts, res):

        assert(gts.keys() == res.keys())
        imgIds = gts.keys()

        bleu_scorer = BleuScorer(n=self._n)
        for id in imgIds:
            hypo = res[id]
            ref = gts[id]

            # Sanity check.
            assert(type(hypo) is list)
            assert(len(hypo) >= 1)
            assert(type(ref) is list)
            assert(len(ref) >= 1)

            bleu_scorer += (hypo[0], ref)

        #score, scores = bleu_scorer.compute_score(option='shortest')
        score, scores = bleu_scorer.compute_score(option='closest', verbose=0)
        #score, scores = bleu_scorer.compute_score(option='average', verbose=1)

        # return (bleu, bleu_info)
        return score, scores

    def method(self):
        return "Bleu"


if __name__=='__main__':
    bleu_scorer = BleuScorer(n=4)
    for id in range(2):
        hypo = ['sos eos']
        ref = ['sos good eos', 'sos jll good good good good eos']

        # Sanity check.
        assert (type(hypo) is list)
        assert (len(hypo) >= 1)
        assert (type(ref) is list)
        assert (len(ref) >= 1)

        bleu_scorer += (hypo[0], ref)

    # score, scores = bleu_scorer.compute_score(option='shortest')
    score, scores = bleu_scorer.compute_score(option='closest', verbose=0)
    print(scores)#for range(1) [[0.6065306591061034], [1.9180183530189284e-08], [6.065306591061037e-08], [1.0785809827805428e-07]]
    print(score) #[0.6065306594093685, 1.3562437847075899e-08, 4.8140370339869904e-08, 9.069748823777882e-08]
    '''
    for two samples range(2)
    [[0.6065306591061034, 0.6065306591061034], 
    [1.9180183530189284e-08, 1.9180183530189284e-08], 
    [6.065306591061037e-08, 6.065306591061037e-08], 
    [1.0785809827805428e-07, 1.0785809827805428e-07]]

    '''