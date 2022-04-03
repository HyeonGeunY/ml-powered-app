import sys
import os
if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from pathlib import Path
import pandas as pd
import numpy as np
import pytest

# # pytest를 적절히 임포트하기 위해 필요합니다.
# myPath = os.path.dirname(os.path.abspath(__file__))
# sys.path.insert(0, myPath + "/../")

from ml_editor.data_ingestion import parse_xml_to_csv
from ml_editor.data_processing import (
    get_random_train_test_split,
    get_split_by_author,
    add_text_features_to_df,
    format_raw_df,
)

REQUIRED_FEATURES = [
    "is_question",
    "action_verb_full",
    "language_question",
    "question_mark_full",
    "text_len",
]
CURR_PATH = Path(os.path.dirname(__file__))
XML_PATH = Path("fixtures/MiniPosts.xml")
CSV_PATH = Path("fixtures/MiniPosts.csv")

# csv 파일을 확인
@pytest.fixture(scope="session", autouse=True)
def get_csv():
    parse_xml_to_csv(CURR_PATH / XML_PATH, save_path=CURR_PATH / CSV_PATH)


@pytest.fixture
def df_with_features():
    df = pd.read_csv(CURR_PATH / CSV_PATH)
    df = format_raw_df(df.copy())
    return add_text_features_to_df(df.copy())


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_feature_presence(df_with_features):
    for feat in REQUIRED_FEATURES:
        assert feat in df_with_features.columns


def test_random_split_proportion():
    """
    훈련 세트와 테스트 세트의 비율을 확인한다.
    """
    df = pd.read_csv(CURR_PATH / CSV_PATH)
    train, test = get_random_train_test_split(df, test_size=0.3)
    print(len(train), len(test))
    assert float(len(train) / 0.7) == float(len(test) / 0.3)


def test_author_split_no_leakage():
    """
    author를 기준으로 훈련세트와 테스트세트가 나누어져 데이터 누수가 없는 지 확인한다.
    """
    df = pd.read_csv(CURR_PATH / CSV_PATH)
    train, test = get_split_by_author(df, test_size=0.3)
    train_owners = set(train["OwnerUserId"].values)
    test_owners = set(test["OwnerUserId"].values)
    assert len(train_owners.intersection(test_owners)) == 0


def test_feature_type(df_with_features):
    """
    특징 데이터의 타입을 검사한다.
    """
    assert df_with_features["is_question"].dtype == bool
    assert df_with_features["action_verb_full"].dtype == bool
    assert df_with_features["language_question"].dtype == bool
    assert df_with_features["question_mark_full"].dtype == bool
    assert df_with_features["text_len"].dtype == np.int64


def test_text_length(df_with_features):
    """
    특징 데이터(text_len)의 최소, 최대, 평균 길이를 검사한다.
    """
    text_mean = df_with_features["text_len"].mean()
    text_max = df_with_features["text_len"].max()
    text_min = df_with_features["text_len"].min()
    assert text_mean in pd.Interval(left=200, right=1000)
    assert text_max in pd.Interval(left=0, right=10000)
    assert text_min in pd.Interval(left=0, right=1000)
