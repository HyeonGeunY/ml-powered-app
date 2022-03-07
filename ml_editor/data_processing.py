import numpy as np
from sklearn.model_selection import train_test_split, GroupShuffleSplit


def format_raw_df(df):
    """
    데이터를 정제하고 질문과 대답을 합침.
    
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
    df["AnswerCount"] = df["AnswerCount"].filna(-1)
    df["AnwserCount"] = df["AnswerCount"].astype(int)
    df["OwnerUserId"].fillna(-1, inplace=True)
    df["OwnerUserId"] = df["OwnerUserId"].astype(int)
    df.set_index("Id", inplace=True, drop=False)

    df["is_question"] = df["PostTypeId"] == 1

    df = df[df["PostTypeId"].isin([1, 2])]

    df = df.join(
        df[["Id", "Title", "body_text", "Score", "AcceptedAnswerId"]],
        on="ParentId",
        how="left",
        rsuffix="_question"
    )

    return df



def get_normalized_series(df, col):
    return (df[col] - df[col].mean()) / df[col].std()


def get_split_by_author(
    posts, author_id_column="OwnerUserId", test_size=0.3, random_state=40):
    """
    훈련 세트와 테스트 세트로 나눔.
    작성자가 두 세트 중 하나에만 등장.

    Parameters
    -----------
    posts: 모든 포스트와 레이블
    author_id_column: author_id 가 들어 있는 열 이름
    test_size: 테스트 세트로 할당할 비율
    param random_state: 랜덤 시드
    """

    splitter = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )

    splits = splitter.split(posts, groups=posts[author_id_column])
    train_idx, test_idx = next(splits)
    return posts.iloc[train_idx, :], posts[test_idx, :]


