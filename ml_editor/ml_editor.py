import argparse
from enum import unique
import logging
from string import punctuation
import sys

from numpy import average

import pyphen
import nltk

pyphen.language_fallback("en_US")

logger = logging.getLogger()
logger.setLevel(logging.INFO)
console_out = logging.StreamHandler(sys.stdout)
console_out.setLevel(logging.DEBUG)
logger.addHandler(console_out)

def parse_arguments():
    """
    return : 수정할 텍스트
    """

    parser = argparse.ArgumentParser(description="Receive text to be edited")
    parser.add_argument("text", metavar="input text", type=str) # metavar: help 명령에서 표시되는 위치 인자의 이름을 바꿈
    args = parser.parse_args()
    return args.text

def clean_input(text):
    """
    텍스트 정제 
    return : ASCII 코드 이외의 문자를 제거한 텍스트
    """

    return str(text.encode().decode("ascii", errors="ignore"))

def preprocess_input(text):
    """
    텍스트 토큰화
    """
    sentences = nltk.sent_tokenize(text)
    tokens = [nltk.word_tokenize(sentence) for sentence in sentences]
    return tokens


def compute_flesch_reading_ease(total_syllables, total_words, total_sentences):
    """
    요약 통계로부터 가독성 점수를 계산.
    
    Parameters
    -----------
    total_syllables : 입력 텍스트에 있는 음절 개수
    total_words : 입력 텍스트에 있는 단어 개수
    total_sentences : 입력 텍스트에 있는 문장 개수

    Return
    ----------
    A readabilty score : 점수가 낮을수록 더 복잡한 텍스트.
    문장 당 단어 수, 단어 당 음절 수 가 증가할수록 값이 낮아진다.

    """
    return (206.85 - 1.015 * (total_words / total_sentences) - 84.6 * (total_syllables / total_words))

def get_reading_level_from_flesch(flesch_score):
    """
    Parameters
    -----------
    flesch_score
    
    Return
    -------
    플레시 점수에 대한 가독성 수준

    """

    if flesch_score < 30:
        return "매우 읽기 어려움"
    elif flesch_score < 50:
        return "읽기 어려움"
    elif flesch_score < 60:
        return "약간 읽기 어려움"
    elif flesch_score < 70:
        return "보통"
    elif flesch_score < 80:
        return "약간 읽기 쉬움"
    elif flesch_score < 90:
        return "읽기 쉬움"
    else:
        return "매우 읽기 쉬움"


def compute_average_word_length(tokens):
    """
    한 문장에 있는 단어의 길이 계산
    
    Parameters
    --------------
    tokens : 단어 리스트

    Return
    ----------
    리스트에 있는 단어의 평균 길이
    """ 
    word_lengths = [len(word) for word in tokens]
    return sum(word_lengths) / len(word_lengths)


def compute_total_average_word_length(sentence_list):
    """
    여러 문장에 대한 단어의 평균 길이를 계산

    Parameters
    ------------
    sentence_list : 단어의 리스트로 구성된 문장 리스트

    Returns
    --------
    문장 리스트에 있는 단어의 평균 길이
    """
    lengths = [compute_average_word_length(tokens) for tokens in sentence_list]
    return sum(lengths) / len(lengths)

def compute_total_unique_words_fraction(sentence_list):
    """
    고유한 단어 비율 계산

    Parameters
    -----------
    sentence_list : 단어의 리스트로 구성된 문장 리스트

    Return
    -----------
    문장에 있는 고유한 단어의 비율
    """
    all_words = [word for word_list in sentence_list for word in word_list]
    unique_words = set(all_words)
    return len(unique_words) / len(all_words)


def count_word_usage(tokens, word_list):
    """
    주어진 단어 리스트의 등장 횟수

    Parameters
    ------------------
    """
    return len([word for word in tokens if word.lower() in word_list])


def count_word_syllables(word):
    """
    단어에 있는 음절 횟수
    :param word: 하나의 단어 문자열
    :return: pyphen으로 구한 음절 개수
    """
    dic = pyphen.Pyphen(lang="en_US")
    # 음절 사이에 하이픈("-")을 추가한 단어를 반환합니다.
    hyphenated = dic.inserted(word)
    return len(hyphenated.split("-"))


def count_sentence_syllables(tokens):
    """
    문장에 있는 음절 개수를 세기.
    param tokens : 단어와 구둣점의 리스트
    return : 문장에 있는 음절 개수
    """

    punctuation = ".,!?/"
    return sum([count_word_syllables(word) for word in tokens if word not in punctuation]) 
    # 구둣점이 아닌 각 단어들에 대해서 음절 단위로 끊은 후 문장 전체의 음절 수 세기


def count_total_syllables(sentence_list):
    """
    문장에 리스트에 있는 음절 세기

    Parameters 
    ------------
    sentence_list : 문장 리스트
    return : 문장에 있는 음절의 개수
    """
    return sum([count_sentence_syllables(sentence) for sentence in sentence_list])


def count_words_per_sentence(sentence_tokens):
    """
    문장에 있는 단어 세기
    Parameters
    ----------------------
    sentence_tokens: 단어와 구둣점의 리스트

    Returns
    --------------------
    return : 문장에 있는 단어의 개수
    """

    punctuation = ".,!?/"
    return len([word for word in sentence_tokens if word not in punctuation])


def count_total_words(sentence_list):
    """
    문장 리스트에 있는 단어 세기

    Parameters
    -------------
    sentence_list : 단어의 리스트로 구성된 문장 리스트
    
    Return
    ------------
    문장에 있는 단어의 개수

    """

    return sum([count_words_per_sentence(sentence) for sentence in sentence_list])


def get_suggestions(sentence_list):
    """
    추천을 포함한 문자열 반환
    """

    told_said_usage = sum((count_word_usage(tokens, ["told", "said"]) for tokens in sentence_list))
    but_and_usage = sum((count_word_usage(tokens, ["but", "and"]) for tokens in sentence_list))

    wh_adverbs_usage = sum(
        (
            count_word_usage(
                tokens,
                [
                    "when",
                    "where",
                    "why",
                    "whence",
                    "whereby",
                    "wherein",
                    "whereupon",
                ],
            ) for tokens in sentence_list
        )
    )   

    result_str = ""
    adverb_usage = f"단어 사용량: {told_said_usage} told/said, {but_and_usage} but/and, {wh_adverbs_usage} wh-접속사"
    result_str += adverb_usage
    average_word_length = compute_total_average_word_length(sentence_list)
    unique_words_fraction = compute_total_average_word_length(sentence_list)

    word_stats = f"평균 단어 길이 {average_word_length:.2f}, 고유한 단어의 비율: {unique_words_fraction:.2f}"
    result_str += "<br/>" # html을 위한 br문
    result_str += word_stats

    number_of_syllables = count_total_syllables(sentence_list)
    number_of_words = count_total_words(sentence_list)
    number_of_sentences = len(sentence_list)

    syllable_counts = f"{number_of_syllables}개의 음절, {number_of_words}개 단어, {number_of_sentences}개 문장"
    result_str += "<br/>"
    result_str += syllable_counts
    
    flesch_score = compute_flesch_reading_ease(number_of_syllables, number_of_words, number_of_sentences)
    flesch = f"플래시 점수 {flesch_score:.2f}: {get_reading_level_from_flesch(flesch_score)}"

    result_str += "<br/>"
    result_str += flesch

    return result_str

def get_recommendations_from_input(text):
    """
    입력 문자열에 대해 정제, 전처리, 규칙 기반 추천 생성
    Parameters
    ----------
    text : 입력 텍스트
    
    Return
    ---------
    주어진 텍스트 입력에 대한 추천
    """
    processed = clean_input(text)
    tokenized_sentences = preprocess_input(processed)
    suggestions = get_suggestions(tokenized_sentences)
    return suggestions

if __name__ == "__main__":
    input_text = parse_arguments()
    print(get_recommendations_from_input(input_text))







