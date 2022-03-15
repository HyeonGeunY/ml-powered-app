import os
from pathlib import Path
from utils import _download_raw_dataset, _extract_raw_dataset

from tqdm import tqdm
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ElT
import pandas as pd
import toml

#ROOT_DATA_DIRNAME = Path("/home/hyoon/Documents/ml-powered-application/data")
ROOT_DATA_DIRNAME = Path("D:/hyoon/side_project/ml-app/data")
RAW_DATA_DIRNAME = ROOT_DATA_DIRNAME / "raw" / "writers" 
METADATA_FILENAME = RAW_DATA_DIRNAME / "metadata.toml"
DL_DATA_DIRNAME = ROOT_DATA_DIRNAME / "downloaded" / "writers"
EXTRACTED_DATASET_DIRNAME = DL_DATA_DIRNAME / "writersdb"
PROCESSED_DIRNAME = ROOT_DATA_DIRNAME / "processed" / "writers"

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

    # 각 행은 질문 하나의 질문.
    all_rows = [row.attrib for row in root.findall("row")]

    for item in tqdm(all_rows):
        soup = BeautifulSoup(item["Body"], features="html.parser")
        item["body_text"] = soup.get_text()

    df = pd.DataFrame.from_dict(all_rows)
    if save_path:
        if not os.path.exists(save_path.parents[0]):
            os.makedirs(save_path.parents[0])
        df.to_csv(save_path)
    return df

def get_data_from_dump(site_name, load_existing=True):
    """
    xml dump 로드 -> 파싱 -> csv -> 직렬화 -> 반환

    Parameters
    ------------
    load_existing: 기존에 추출한 csv를 로드할지 새로 생성할지 결정. 
    site_name: str
        stackexchange 웹사이트 이름.
    
    Return
    ----------
    parsing된 xml 데이터 : pd.DataFrame
    """

    data_path = Path("/home/hyoon/Documents/ml-powered-application/data/writers_stackexchange/")
    dump_name = f"{site_name}.stackexchange.com/Posts.xml"
    extracted_name = f"{site_name}_from_dump.csv"
    dump_path = data_path / dump_name
    extracted_path = data_path / extracted_name

    if not (load_existing and os.path.isfile(extracted_path)):
        all_data = parse_xml_to_csv(dump_path)
        all_data.to_csv(extracted_path)
    else:
        all_data = pd.DataFrame.from_csv(extracted_path)
    
    return all_data

def download_and_extract():
    metadata = toml.load(METADATA_FILENAME)
    filename = _download_raw_dataset(metadata, DL_DATA_DIRNAME)
    
    xml_dirname = EXTRACTED_DATASET_DIRNAME / "xml"
    if not os.path.exists(xml_dirname):
        os.makedirs(xml_dirname)

    _extract_raw_dataset(filename, xml_dirname)


if __name__ == "__main__":
    download_and_extract()
    parse_xml_to_csv(EXTRACTED_DATASET_DIRNAME / "xml" / "Posts.xml", PROCESSED_DIRNAME / "writers.csv")
    

    

        