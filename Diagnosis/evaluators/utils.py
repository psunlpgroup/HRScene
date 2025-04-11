import re
from typing import Tuple


def clean_response(response: str) -> str:
    if not response:
        return ""
    response = response.split('\n')[-1].lower().replace('answer:', "").replace('*', "").replace('.', "").strip()
    
    return response


def complexgrid_parse(response: str) -> Tuple[int, int]:
    response = clean_response(response)
    row_col_pattern = re.compile(r"<row>(\d+)<\/row>.*?<col>(\d+)<\/col>")

    matches = row_col_pattern.findall(response)
    if matches:
        row_parsed, col_parsed = matches[0]
        row_parsed = int(row_parsed) if row_parsed.isdigit() else -10
        col_parsed = int(col_parsed) if col_parsed.isdigit() else -10
    else:
        row_parsed = col_parsed = -10
        for i, ch in enumerate(response):
            if ch.isdigit():
                if i + 1 < len(response) and response[i + 1].isdigit():
                    row_parsed = int(f'{ch}{response[i + 1]}')
                    break
                elif ch.isdigit():
                    row_parsed = int(ch)
                    break
        for i, ch in enumerate(response[::-1]):
            if ch.isdigit():
                if i + 1 < len(response) and response[i + 1].isdigit():
                    col_parsed = int(f'{response[i + 1]}{ch}')
                    break
                elif ch.isdigit():
                    col_parsed = int(ch)
                    break
    
    return row_parsed, col_parsed
