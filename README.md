# ml-powered-app
from machine learning powered application


# 가상환경 설정
`python -m venv venv or conda`

## 라이브러리 설치
가상환경 활성화 후    
`pip install -r requirements.txt`

## 언어 모델 다운

`python -m spacy download en_core_web_sm`

`python -m spacy download en_core_web_lg`

## `nltk` 패키지   
가상환경 활성화 후

`python`

`import nltk`

`nltk.download('punkt')`

## 데이터 다운

`./data/raw/writers/metadata.toml` 이용

`./ml_editor/data_ingestion.py` 실행하면 다운로드.   
=> ` /data/processed/writers/writers.csv` 에 저장

7z 압축을 풀기 위해 py7zr 라이브러리 사용.
다운로드 도중 연결이 끊기는 것을 대비하여 여러번 재접속 시도 코드 작성. (`./ml_editor/utils._download_url`)

