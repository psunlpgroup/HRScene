from typing import List

from .utils import clean_response, realworld_parse


def default_realworld_metrics(responses: List[dict], labels: List[dict]) -> dict:
    eval_results = []

    for response, label in zip(responses, labels):
        prediction = realworld_parse(response["response"])
        score = int(prediction == label)

        eval_results.append({
            "id": response["metadata"]["id"],
            "question": response["metadata"]["question"],
            "response": response["response"],
            "parsed_response": prediction,
            "answer": label,
            "score": score
        })

    return eval_results
