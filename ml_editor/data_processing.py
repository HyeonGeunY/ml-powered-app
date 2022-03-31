import numpy as np
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import vstack, hstack


def format_raw_df(df):
    """
    데이터를 정제하고 질문과 대답을 연결한다.

    Parameters
    ------------
    df: pd.DataFrame
    pd.DataFrame

    Reterns
    ----------
    정제된 데이터: pd.DataFrame
    """

    df["PostTypeId"] = df["PostTypeId"].astype(int)
    df["Id"] = df["Id"].astype(int)
    df["AnswerCount"] = df["AnswerCount"].fillna(-1)
    df["AnwserCount"] = df["AnswerCount"].astype(int)
    df["OwnerUserId"].fillna(-1, inplace=True)
    df["OwnerUserId"] = df["OwnerUserId"].astype(int)
    df.set_index("Id", inplace=True, drop=False)

    df["is_question"] = df["PostTypeId"] == 1

    df = df[df["PostTypeId"].isin([1, 2])]
    # ParentId: 답변 데이터의 경우 매칭되는 질문의 Id
    df = df.join(
        df[["Id", "Title", "body_text", "Score", "AcceptedAnswerId"]],
        on="ParentId",
        how="left",
        rsuffix="_question",
    )

    return df


def train_vectorizer(df):
    """
    TfidVectorizer를 이용하여 벡터화 객체를 훈련.
    훈련 데이터와 그 외 데이터를 변환하는데 사용하 벡터화 객체 반환

    Args:
        df (pd.DataFrame): 벡터화 객체를 훈련하는데 사용할 데이터

    Returns:
        훈련된 벡터화 객체
    """
    vectorizer = TfidfVectorizer(strip_accents="ascii", min_df=5, max_df=0.5, max_features=10000)

    vectorizer.fit(df["full_text"].copy())
    return vectorizer


def get_vectorized_series(text_series, vectorizer):
    """
    사전 훈련된 벡터화 객체를 사용해 입력 시리즈를 벡터화

    Args:
        text_series (pd.DataFrame): 텍스트의 판다스 시리즈
        vectorizer (sklearn object): 사전 훈련된 sklearn의 벡터화 객체

    Returns:
        벡터화된 특성 배열
    """
    vectors = vectorizer.transform(text_series)
    vectorized_series = [vectors[i] for i in range(vectors.shape[0])]
    return vectorized_series


def add_text_features_to_df(df):
    """
    DataFrame에 특성 추가

    Args:
        df (pd.DataFrame): DataFrame

    Return:
        특성이 추가된 DataFrame
    """
    # 질문의 제목에 중요한 정보가 포함.
    df["full_text"] = df["Title"].str.cat(df["body_text"], sep=" ", na_rep="")
    # title과 body_text를 합쳐서 fulltext생성, 스페이스로 구분, null값은 공백으로 대체
    df = add_v1_features(df.copy())

    return df


def add_v1_features(df):
    """
    입력 DataFrame에 첫 번째 특성 추가.

    추가한 특성
    - 질문 길이: 매우 짧은 질문은 대답을 받지 못하는 경향이 있는 것으로 보임.
    - 물음표 여부: 물음표가 없으면 답변을 받을 가능성이 낮아 보임.
    - 명확한 질문에 관련된 어휘(동작 동사 등...): 대답을 받지 못한 질문에는 빠져 있는 경우가 많이 보인다.

    Args:
        df (pd.DataFrame): 질문 DataFrame

    Returns:
        특성이 추가된 DataFrame
    """
    # 답변할 질문을 예측하기 좋은 action verb
    df["action_verb_full"] = (
        df["full_text"].str.contains("can", regex=False)
        | df["full_text"].str.contains("What", regex=False)
        | df["full_text"].str.contains("should", regex=False)
    )
    # 영어사용에 관련한 질문은 답변이 잘 오지 않음.
    df["language_question"] = (
        df["full_text"].str.contains("punctuate", regex=False)
        | df["full_text"].str.contains("capitalize", regex=False)
        | df["full_text"].str.contains("abbreviate", regex=False)
    )
    # 물음표가 있는지 확인
    df["question_mark_full"] = df["full_text"].str.contains("?", regex=False)
    # 매우 짧은 질문은 답을 받지 못하는 경향이 있음.
    df["text_len"] = df["full_text"].str.len()

    return df


def get_vectorized_inputs_and_label(df):
    """
    DataFrame 특성과 텍스트 백터를 연결한다.

    Args:
        df (pd.DataFrame): 계산된 특성의 DataFrame

    Returns:
        (pd.DataFrame): 트성과 텍스트로 구성된 벡터
    """
    vectorized_features = np.append(
        np.vstack(df["vectors"]),
        df[
            [
                "action_verb_full",
                "question_mark_full",
                "norm_text_len",
                "language_question",
            ]
        ],
        1,
    )
    label = df["Score"] > df["Score"].median()

    return vectorized_features, label


def get_normalized_series(df, col):
    """DataFrame 열을 정규화 한다.

    Args:
        df (pd.DataFrame): DataFrame
        col (str): 열 이름

    Returns:
        pd.Series: Z-score를 이용하여 정규화 된 시리즈 객체
    """
    return (df[col] - df[col].mean()) / df[col].std()


def get_random_train_test_split(posts, test_size=0.3, random_state=40):
    """DataFrame을 훈련/테스트 세트로 나눈다.
    DataFrame이 질문마다 하나의 행을 가지고 있는다고 가정한다.

    Args:
        posts (pd.DataFrame): 모든 포스트와 레이블
        test_size (float, optional): 테스트로 할당할 비율
        random_state (int, optional): 랜덤 시드 Defaults to 40.
    """
    return train_test_split(posts, test_size=test_size, random_state=random_state)


def get_split_by_author(posts, author_id_column="OwnerUserId", test_size=0.3, random_state=40):
    """
    훈련 세트와 테스트 세트로 나눔.
    작성자가 두 세트 중 하나에만 등장.

    Parameters
    -----------
    posts: 모든 포스트와 레이블(pd.Dataframe)
    author_id_column: author_id 가 들어 있는 열 이름
    test_size: 테스트 세트로 할당할 비율
    param random_state: 랜덤 시드
    """

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)

    splits = splitter.split(posts, groups=posts[author_id_column])
    train_idx, test_idx = next(splits)
    return posts.iloc[train_idx, :], posts.iloc[test_idx, :]


def get_feature_vector_and_label(df, feature_names):
    """벡터 특성과 다른 특성을 사용해 입력과 출력 벡터를 만든다.
    출력 벡터는 질문의 점수가 median을 기준으로 높은지(True) 낮은지(False)이다.

    Args:
        df (pd.DataFrame): 입력 데이터프레임
        feature_names (str): 'vectors'열을 제외한 특성 열 이름

    Returns:
        (np.array): 특성 배열과 레이블 배열
    """
    vec_features = vstack(df["vectors"])
    num_features = df[feature_names].astype(float)
    features = hstack([vec_features, num_features])
    labels = df["Score"] > df["Score"].median()
    return features, labels


