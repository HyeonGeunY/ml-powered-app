import os
from pathlib import Path

import spacy
import joblib
from tqdm import tqdm
import pandas as pd
import nltk
from scipy.sparse import vstack, hstack

nltk.download("vader_lexicon")
from nltk.sentiment.vader import SentimentIntensityAnalyzer

POS_NAMES = {
    "ADJ": "adjective",
    "ADP": "adposition",
    "ADV": "adverb",
    "AUX": "auxiliary verb",
    "CONJ": "coordinating conjunction",
    "DET": "determiner",
    "INTJ": "interjection",
    "NOUN": "noun",
    "NUM": "numeral",
    "PART": "particle",
    "PRON": "pronoun",
    "PROPN": "proper noun",
    "PUNCT": "punctuation",
    "SCONJ": "subordinating conjunction",
    "SYM": "symbol",
    "VERB": "verb",
    "X": "other",
}

FEATURE_ARR = [
    "num_questions",
    "num_periods",
    "num_commas",
    "num_exclam",
    "num_quotes",
    "num_colon",
    "num_stops",
    "num_semicolon",
    "num_words",
    "num_chars",
    "num_diff_words",
    "avg_word_len",
    "polarity",
]

FEATURE_ARR.extend(POS_NAMES.keys())

SPACY_MODEL = spacy.load("en_core_web_sm")
tqdm.pandas()

curr_path = Path(os.path.dirname(__file__))

model_path = Path("../models/model_2.pkl")
vectorizer_path = Path("../models/vectorizer_2.pkl")
VECTORIZER = None
MODEL = None

def count_each_pos(df):
    """
    품사의 등장 횟수를 세어 입력 DataFrame에 추가합니다.
    :param df: SPACY_MODEL로 전달된 텍스트를 담고 있는 입력 DataFrame
    :return: 등장 회수가 포함된 DataFrame
    """
    global POS_NAMES
    pos_list = df["spacy_text"].apply(lambda doc: [token.pos_ for token in doc])
    for pos_name in POS_NAMES.keys():
        df[pos_name] = (
            pos_list.apply(
                lambda x: len([match for match in x if match == pos_name])
            )
            / df["num_chars"]
        )
    return df