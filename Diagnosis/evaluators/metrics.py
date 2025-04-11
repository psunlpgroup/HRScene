from collections import defaultdict
from typing import List

from .utils import clean_response, complexgrid_parse

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


def default_whitebackground_metrics(responses: List[dict], labels: List[dict]) -> dict:
    eval_results = []

    for response, label in zip(responses, labels):
        prediction = clean_response(response["response"])
        score = min(sum(ans.lower() in prediction.lower() for ans in label["answer"]) / 3, 1) if prediction else 0

        eval_results.append({
            "id_row_col": label["id_row_col"],
            "question": response["metadata"]["question"],
            "answer": label["answer"],
            "response": response["response"],
            "score": score
        })

    return eval_results


def default_complexgrid_metrics(responses: List[dict], labels: List[dict]) -> dict:
    eval_results = []

    for response, label in zip(responses, labels):
        id, row, col = response["metadata"]["id"].split("_")
        row_ans, col_ans = int(row) + 1, int(col) + 1
        row_resp, col_resp = complexgrid_parse(response["response"])
        score = 1 if row_resp == row_ans and col_resp == col_ans else 0

        eval_results.append({
            "id": response["metadata"]["id"],
            "caption": response["metadata"]["caption"],
            "answer": label,
            "response": response["response"],
            "parsed_response": f'row: {row_resp}, col: {col_resp}',
            "score": score
        })

    return eval_results


def draw_heatmap(eval_results: List[dict], grid_size: int, experiment_name: str, save_dir: str) -> None:
    matplotlib.use('Agg')

    result = [[0] * grid_size for _ in range(grid_size)]
    sample_count = 0
    data = defaultdict(list)

    for item in eval_results:
        id, row, col = item['id'].split('_')
        data[id].append({
            'row': row,
            'col': col,
            'score': item['score'],
            'response': item.get('response', '')
        })
    
    for idx in data.keys():
        pred = data[idx]
        for sample in pred:
            row = int(sample['row'])
            col = int(sample['col'])
            if row < grid_size and col < grid_size:
                result[row][col] += sample['score']
        sample_count += 1
    
    normalized_result = [[x/sample_count for x in y] for y in result]
    
    cmap = sns.color_palette("RdYlGn", as_cmap=True)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        normalized_result, cmap=cmap, annot=True, fmt=".2f", 
        linewidths=0.5, square=True, cbar=True,annot_kws={"size": 19, "weight": "bold"}
    )
    
    plt.title(f"{experiment_name}", fontsize=19, fontweight='bold')
    plt.xticks([])
    plt.yticks([])
    
    plt.savefig(f"{save_dir}/{experiment_name}.png", dpi=300, bbox_inches="tight")
    plt.close()
