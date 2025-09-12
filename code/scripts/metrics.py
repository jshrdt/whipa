import os 
import pandas as pd

import panphon
import panphon.distance
dist = panphon.distance.Distance()

import numpy as np
from yaml import safe_load

## used for multipa comparison
# import sys
# sys.path.append("../data/multipa")
# from utils import compute_only_fper, compute_all_metrics, retokenize_ipa

##! adapted from ctaguchi/multipa/utils.py  ##
# Spacing Modifier Letters
sml = set()
for i in range(int(0x2b0), int(0x36f)+1):
    sml.add(chr(i))

#! ## added ##
with open(os.path.join(os.path.dirname(panphon.__file__),
                       "data/diacritic_definitions.yml"), "r") as f:
    # map diacritic: info for indexing
    diacritics = {x['marker']: {k: v for k,v in x.items() if k!='marker'}
                  for x in safe_load(f)['diacritics']}

#! complete some missing SMLs, ensure all diacritics are recognised
for dia in diacritics:
    sml.add(dia)
for i in range(741, 746): #! chao tone letters
    sml.add(chr(i))
    
sml.add(chr(0x207f)) #! nasalised ^n diacritic
sml.add(chr(0x325))  #! devoicing . combine below 
sml.add(chr(0x2e4))  #! pharyngeal fricative diacritic
sml.add("ʰ")

def retokenize_ipa(sent: str):
    #! modified
    tie_flag = False
    modified = []
    for i in range(len(sent)):
        if tie_flag:
            tie_flag = False
            continue
        if sent[i] in sml:
            if i == 0:
                # when the space modifier letter comes at the index 0
                modified.append(sent[i])
                continue
            modified[-1] += sent[i]
            if sent[i] == "\u0361":
                # tie bar
                try:
                    modified[-1] += sent[i+1]  #! error here
                except IndexError:
                    continue  #! utterance final erroneous tie flag
                tie_flag = True
        else:
            modified.append(sent[i])
    return modified
## end ##


unk = set()  # collect unknown (sub)phone segments

def w2feats(w):
    # inspired by multipa.utils.combine_features, but draws more heavily on panphon,
    # adds diacritics and other heuristics, re-normalises values, and includes /all/
    # 24 features.
    if not any([ord(c) in range(741, 746) for c in w]) and dist.fm.validate_word(w, normalize=True):
        # all segments knwon to panphone
        word_ft_vec = dist.fm.word_to_vector_list(w, normalize=True,  numeric=True)
    else:
        # iter through tokenised phones
        word_ft_vec = list()
        for seg in retokenize_ipa(w):
            # init "neutral" feature vector for segment
            seg_feats = np.array([0] * 24)
            # try to maximise overlap with known phones
            for p in dist.fm._segs(seg, include_invalid=True):
                # init "neutral" feature vector for subphone
                p_feats = np.array([0] * 24)

                if dist.fm.seg_known(p):
                    # simplest case: segment known to panphon featuretable
                    p_feats = dist.fm.word_to_vector_list(p, normalize=True, numeric=True)[0]

                elif p in diacritics and diacritics[p]["content"]:
                    # add diacritic features from panphon diacritics yaml entry
                    for k, v in diacritics[p]["content"].items():
                        # locate feature name dimension & alter its value directly
                        seg_feats[dist.fm.names.index(k)] = (1 if v=="+" else -1)
                else:
                    # custom heuristics
                    if p in [
                            # no effect on phonetic features acc. to panphon
                            "͡",  # tie flag
                            "̲",  # retracted
                            '̜',  # less round 
                            '̚',  # unreleased stop
                            "̆",  # extra short
                            "˘", # extra short/non-combine
                            "̞",  # 
                            "̈",  # centralised
                            "'゚",
                            # unclear;
                            '̣',  # possibly oddly encoded syllable break?
                            '̄',  # ? mid tone
                            # ? neither known to IPA, panphon, nor phoible; found in voxangeles corpus
                            "˭",
                            "ˢ",
                            'ᵐ',
                            '̱',
                                ]:
                        continue
                    # manually defined vectors with reference to Phoible features
                    # trying to add these is how it became apparent that Taguchi et al.'s code skips the syl feature (confirmed by manually inspecting their returned vectors)
                    elif p=="ʜ": # voicelesss epiglottal fricative
                        # created in reference to https://phoible.org/parameters/DB415281921248FCDAF723AFA0D986E4#4/36.95/19.65
                        p_feats=np.array([-1, -1, -1, 1, 1, -1, -1, 0, -1, +1, -1, 0, -1, 0, -1, 0, 0, 0, 0, -1, 0, -1, 0, 0])
                    elif p=="ʡ": # epiglottal plosive
                        # https://phoible.org/parameters/D17CE01E5AFD7AA148785E96513B07CA#2/46.1/139.2
                        p_feats= np.array([-1, -1, 1, -1, -1, -1, -1, 0, -1, -1, -1, 0, -1, 0, -1, 0, 0, 0, 0, -1, 0, -1, 0, 0])
                    elif p == "̪":
                        # appears in panphon ipa_all.csv but are missing from diacritics file
                        seg_feats[dist.fm.names.index("distr")] = 1
                    elif p == "̊":
                        # devoicing combine above
                        seg_feats[dist.fm.names.index("voi")] = -1
                    elif p == "ʱ":
                        # custom heuristic
                        seg_feats[dist.fm.names.index("sg")] = 1  # same as ʰ
                        seg_feats[dist.fm.names.index("voi")] = 1
                    elif p == "ˑ":
                        # custom heuristic for "half long"
                        seg_feats[dist.fm.names.index("long")] = 0
                    elif p not in unk:
                        # unknown to panphon and no heurtistics defined
                        unk.add(p)
                        print(unk)
                    else:
                        pass

                # add subphone features to segment features
                seg_feats = np.add(seg_feats, p_feats)

                # additive method may result in values outside -1;1 range, so
                # re-fit values to range
                seg_feats = np.array([v if v in [-1, 0, 1] else (-1 if v < -1 else 1)
                                      for v in seg_feats])
            # add segment feature vector to word vector collection
            word_ft_vec.append(seg_feats)
    return word_ft_vec


class STIPA_METRICS():
    def __init__(self, normalize_str: bool = True,
                 rm_supra: bool = False,
                 rm_tones: bool = False):
        self.norm_str = normalize_str
        # panphon handles some of these internally, but enforce for safety
        self.blacklist = ({"g": "ɡ"} if not self.norm_str  # deactivate string normalization for maximal comparability wiht MultIPA
                          else {#rm whitespace/punctuation
                                " ": "",
                                ".": "",
                                "。": "",
                                "-": "",
                                "—": "",
                                "'": "", # primary syllabe stress; missing from most data, not trying to predict stress
                                "ˌ": "",  # secondary stress, same as above
                                ",": "",
                                "?": "",
                                "？": "",
                                "(": "",
                                ")": "",
                                "!": "",
                                ";": "",
                                "、": "",
                               "�": "",  # b'\xef\xbf\xbd'; relic from truncated char
                               "#": "", # noise introduced by multipa greek data
                                # matter of representation, linguistically same phone
                                "ʧ": "t͡ʃ",
                                "ʦ": "t͡s",
                                "ʤ": "d͡ʒ",
                                "ʣ": "d͡z",
                                "ɔʰ": "ɔ̤", # untypical usage of apirated h on vowels
                                # unicode differences
                                "а": "a",
                                "е": "e",
                                "ο": "o",
                                "υ": "ʋ",
                                "ε": "ɛ",
                                "і": "i",
                                "g": "ɡ",
                                "һ": "h",
                                "ä": "ä",
                                "ã": "ã",
                                "õ": "õ",
                                "ă": "ă",
                                "ũ": "ũ",
                                "ç": "ç", 
                                "ς": "ç",
                                "φ": "ɸ",
                                "ĩ": "ĩ",
                                "ӕ": "æ",
                                "ẽ": "ẽ",
                                "ĭ": "ĭ",
                                "ĕ": "ĕ",
                                "ḛ": "ḛ",
                                "ɚ": "ɘ˞",
                                "ç": "ç",
                                "ł": "l̴", 
                                "ˁ": "ˤ",
                                # normalise tone representation
                                "̋": "˥",
                                "́": "˦",
                                "̅": "˧",
                                "̀": "˨",
                                "˵": "˩",
                                "̌": "˩˥",
                                "̂": "˥˩",
                                ## outdated IPA characters
                                "ı": "ɪ",
                                "ɿ": "ɪ",
                                "ᴊ": "ɪ",
                                "ɩ": "ɪ",
                                "ι": "ɪ",
                                })
        if rm_supra:
            self.blacklist.update({
                                "ː": "",
                                "̆": "",
                                "̯": "",
                                "̈": "",
                                "ˑ": "",
                                "̆":"",
                                "˘":""
                                })
        if rm_tones:
            self.blacklist.update({
                chr(i): "" for i in range (741, 746) #˥, ˦, ˧, ˨, ˩,
                })

        self.dist = panphon.distance.Distance()
        self.ft = panphon.FeatureTable()
        self.feature_df = pd.DataFrame({char: seg.data for char, seg in self.ft.seg_dict.items()}).transpose()


    def normalize(self, ipa):
        if self.norm_str:
            # normalise hard-replace characters
            import unicodedata
            ipa = unicodedata.normalize('NFD', ipa).lower()
            return "".join([c.lower() if not c in self.blacklist else self.blacklist[c] for c in ipa])
        else:
            return ipa

    def compute_all(self, pred: str, gold: str, char_based=False) -> dict:
        pred = self.normalize(pred)
        gold = self.normalize(gold)
        if char_based:
            self.lvnshtn = self.do_lvnshtn(pred, gold)
            self.cer = self.lvnshtn / len(gold)  # equals PER
 
        # adapted from panphon.distance.Distance.phoneme_error_rate
        self.ped = self.do_ped(pred, gold)
        # adapted from panphon.distance.Distance.partial_hamming_feature_edit_distance
        # phonetic feature edit distance
        self.pfed = self.do_pfed(pred, gold)

        results= {"ped": self.ped,
                  "per": self.ped /len(retokenize_ipa(gold)) * 100,
                  "per_norm": self.ped/ max([len(retokenize_ipa(pred)),
                                             len(retokenize_ipa(gold))
                                             ]) * 100,
                  "pfed": self.pfed,
                  "pfer": self.pfed /len(retokenize_ipa(gold)) * 100,
                  "pfer_norm": self.pfed / max([len(retokenize_ipa(pred)),len(retokenize_ipa(gold))]) * 100,
                  }
        if char_based:
            results.update({# Implementation via fast_levenshtein_distance in PanPhon (eitdistance module)
                            # range 0 - max(len(pred), len(gold))
                            "lvnshtn": self.lvnshtn,
                            # build on lvnshtn, results equal to HuggingFace evaluate module but faster; cer (here): 0.00013s vs evaluate module 1.0434s
                            # (S+D+I) / (S+D+C) = (S+D+I) / N -> range 0-1 if pred<gold; else range 0 - len(pred)
                            "cer": self.cer,
                            })
        return results
            

    def do_lvnshtn(self, pred, gold):
        return self.dist.fast_levenshtein_distance(source=pred, target=gold)

    def do_cer(self, pred, gold):
        d = self.dist.fast_levenshtein_distance(source=pred, target=gold)
        return d / len(gold)

    def do_ped(self, pred, gold):
        return self.dist.min_edit_distance(lambda v: 1,
                                    lambda v: 1,
                                    lambda x,y: 0 if x == y else 1,
                                    [[]],
                                    source=retokenize_ipa(self.normalize(pred)),
                                    target=retokenize_ipa(self.normalize(gold))
                                    )
    def do_per(self, pred, gold):
        return self.do_ped(pred, gold) / len(retokenize_ipa(self.normalize(gold))) *100

    def do_per_norm(self, pred, gold):
         return self.do_ped(pred, gold) / max([len(retokenize_ipa(self.normalize(pred))),
                                               len(retokenize_ipa(self.normalize(gold)))]) * 100

    def do_pfed(self, pred, gold):
        return self.dist.min_edit_distance(lambda v: 1,
                                            lambda v: 1,
                                            dist.partial_hamming_substitution_cost,
                                            [[]],
                                            source=w2feats(self.normalize(pred)),
                                            target=w2feats(self.normalize(gold)))
    def do_pfer(self, pred, gold):
        return self.do_pfed(pred, gold) / len(retokenize_ipa(self.normalize(gold))) * 100

    def do_pfer_norm(self, pred, gold):
        return self.do_pfed(pred, gold) / max([len(retokenize_ipa(self.normalize(pred))),
                                               len(retokenize_ipa(self.normalize(gold)))]) * 100

    # def do_multipa_pfer(self, pred, gold):
    #     return float(compute_only_fper(pred, gold, self.feature_df))

    # def do_multipa_all(self, pred, gold):
    #     return compute_all_metrics(pred, gold, self.feature_df)


class CORE_STATS:
    # used to flag 'outliers', extract, & recomputing stats without them
    def __init__(self, results,
                 cutoffs = {'per': 400, 'pfer': 200}):
        self.results = results
        self.cutoffs = cutoffs
        self.macro = results['macro']
        self.micro = {lg: {w[0]: w[1] for w in v['micro'].items() if w[0]!='macro'}
                      for lg, v in results.items() if lg!='macro'}
        self.outliers = self.spot_outliers()
        self.n_outliers = {lg: len(x) for lg, x in self.outliers['ids'].items()}
        self.new_results = self.recalculate()

    def spot_outliers(self):
        outliers = {'ids': dict(), 'entries': dict()}
        for lg, entry in self.micro.items():
            outliers['entries'][lg] = {m: list() for m in self.cutoffs}
            for sample in entry.items():
                for m, maxval in self.cutoffs.items():
                    if sample[1][m]>=maxval:
                        outliers['ids'][lg] = outliers['ids'].get(lg, set())
                        outliers['ids'][lg].add(sample[0])
                        outliers['entries'][lg][m].append(sample)
        return outliers

    def recalculate(self):
        new_results = {lg : dict() for lg in self.results}
        for lg, v in (lambda x: {a: b for a,b in x.items() if a!='macro'})(self.results).items():
            new_results[lg]['micro'] = {
                w[0]: w[1] for w in v['micro'].items() if (
                    w[0]!='macro' and w[0] not in self.outliers['ids'].get(lg, []))}
            new_results[lg]['macro'] = {
                m: (sum([sample[m] for sample in new_results[lg]['micro'].values()])
                    /len(new_results[lg]['micro'])) for m in self.macro}
        new_results['macro'] = {m: sum([new_results[lg]['macro'][m]
            for lg in new_results if lg!='macro'])/(len(new_results)-1) for m in self.macro}
        return new_results


class VIS_RESULTS:
    def __init__(self, data=False, fname=False, cutoffs = {'per': 400, 'pfer': 200}):
        assert data or (fname and os.path.isfile(fname)), ValueError("No data or valid path passed.")
        self.fname = fname
        self.data = self.read_file(fname) if self.fname else data
        self.macro = pd.DataFrame(self.data['macro'].values(), index=self.data['macro'].keys(), columns=['score'])
        self.macros = pd.DataFrame({lg: r['macro'] if lg!='macro' else r for lg,r in self.data.items()})
        self.micro = pd.DataFrame({lg: r['micro'] for lg,r in self.data.items() if lg!='macro'})
        self.core = CORE_STATS(self.data, cutoffs=cutoffs)
        self.new_results = self.core.new_results

    def read_file(self, fname):
        with open(fname, 'r') as f:
            res = eval(f.read())
        return res
