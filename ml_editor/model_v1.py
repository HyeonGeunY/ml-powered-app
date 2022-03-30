import os
from pathlib import Path

import pandas as pd
import joblib
from scipy.sparse import vstack, hstack

from ml_editor.data_processing import add_v1_features

FEATURE_ARR = [
    "action_verb_full",
    "question_mark_full",
    "text_len",
    "language_question",
]

# 현재 파일이 있는 폴더를 기준으로 model 경로 지정하여 로드
curr_path = Path(os.path.dirname(__file__))

model_path = Path("../models/model_1.pkl")
vectorizer_path = Path("../models/vectorizer_1.pkl")
VECTORIZER = joblib.load(curr_path / vectorizer_path)
MODEL = joblib.load(curr_path / model_path)

def get_model_probabilities_for_input_texts(text_array):
    """질문이 높은 점수를 받을 확률을 나타내는 점수의 배열을 반환한다.

    Args:
        text_array (text_array): 점수를 매길 질문의 배열

    Returns:
        _type_: 예측 확률 배열
        format: [[prob_low_score1, prob_high_score_1], ...]
    """
    global FEATURE_ARR, VECTORIZER, MODEL
    
    # 모델을 학습할 때 사용한 vectorizer(중요!)로 text 벡터화
    vectors = VECTORIZER.transform(text_array)
    text_ser = pd.DataFrame(text_array, columns=["full_text"])
    # 문자 정보로부터 특징 추가
    text_ser = add_v1_features(text_ser)
    # 벡터화된 텍스트와 추가할 특징을 이용하여 input 생성
    vec_features = vstack(vectors)
    num_features = text_ser[FEATURE_ARR].astype(float)
    features = hstack([vec_features, num_features])
    return MODEL.predict_proba(features)


def get_model_predictions_for_input_texts(text_array):
    """질문 배열(인풋)에 대한 레이블 배열(클래스) 반환

    Args:
        text_array (text array): 질문 문자열로 구성된 배열

    Returns:
        _type_: 클래스 배열
        format: [False, True, ...]
    """
    probs = get_model_probabilities_for_input_texts(text_array)
    predicted_classes = probs[:, 0] < probs[:, 1]
    return predicted_classes
