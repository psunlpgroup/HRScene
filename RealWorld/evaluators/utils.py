import re
from typing import Tuple


def clean_response(response: str) -> str:
    if not response:
        return ""
    response = response.split('\n')[-1].lower().replace('answer:', "").replace('*', "").replace('.', "").strip()
    
    return response


def realworld_parse(response: str) -> str:
    # here are some possible formats for the response we met during the experiments
    # you could customize the pattern here to fit your needs
    pattern1 = r'<ans>([A-Z])</ans>'
    match1 = re.search(pattern1, response)
    pattern2 = r'\(([A-Z])\)'
    match2 = re.search(pattern2, response)
    
    if match1:
        parsed_response = match1.group(1)
    elif match2:
        parsed_response = match2.group(1)
    else:
        parsed_response = response
    
    return parsed_response
