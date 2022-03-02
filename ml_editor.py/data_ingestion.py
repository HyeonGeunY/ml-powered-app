import os
from pathlib import Path

from tqdm import tqdm
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ElT
import pandas as pd

def parse_xml_to_csv(path, save_path=None):
    """
    xml 포스트 덤프를 열어 텍스트 -> csv 변환

    Parameters
    -----------
    path: post가 담긴 xml 문서 경로: str

    Return
    ---------
    처리된 텍스트의 데이터프레임: pd.DataFrame
    """
    # xml 파일 파싱
    doc = ElT.parse(path)
    root = doc.getroot()

    # 각 행은 질문 하나 
    