import argparse
import logging
import sys

import pyphen
import nltk

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
    sentence = nltk.sent_tokenize(text)
    tokens = [nltk.word_tokenize(sentence) for sentence in sentences]
    return tokens


