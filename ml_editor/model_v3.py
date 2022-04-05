import os
from pathlib import Path

import spacy
import joblib
from tqdm import tqdm
import pandas as pd
import nltk

from ml_editor.explanation_generation import (
    FEATURE_ARR,
)

from ml_editor.model_v2 import add_v2_text_features

nltk.download("vader_lexicon")

SPACY_MODEL = spacy.load("en_core_web_sm")
tqdm.pandas()

curr_path = Path(os.path.dirname(__file__))

model_path = Path("../models/model_3.pkl")
MODEL = joblib.load(curr_path / model_path)


def get_features_from_input_text(text_input):
    """
    고유한 텍스트 입력에 대해 특성을 생성한다.

    Args:
        text_input(str): 질문 문자열

    Returns:
        pd.Series: 모델 v3 특성을 담고 있는 1행 시리즈
    """
    arr_features = get_features_from_text_array([text_input])
    return arr_features.iloc[0]


def get_features_from_text_array(input_array):
    """
    입력 텍스트 배열에 대해 특성을 생성한다.

    Args:
        np.array: 입력 질문 배열

    Returns:
        pd.DataFrame: 특성 DataFrame
    """
    text_ser = pd.DataFrame(input_array, columns=["full_text"])
    text_ser = add_v2_text_features(text_ser.copy())
    features = text_ser[FEATURE_ARR].astype(float)
    return features


def get_model_probabilities_for_input_texts(text_array):
    """
    입력 텍스트 배열에 대한 모델 v3의 추정 확률을 반환한다.

    Args:
        text_array(str): 입력 질문 배열

    Returns:
        np.ndarray: 예측 배열
    """
    global MODEL
    features = get_features_from_text_array(text_array)
    return MODEL.predict_proba(features)


def get_question_score_from_input(text):
    """
    고유한 텍스트 입력에 대해 모델 v3의 확률을 반환한다.

    Args:
        text(str): 입력 문자열

    Return:
        높은 점수를 받는 질문의 예측 확률
    """

    preds = get_model_probabilities_for_input_texts([text])
    positive_proba = preds[0][1]
    return positive_proba
