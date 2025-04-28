from typing import List

from .utils import clean_response, realworld_parse


def default_realworld_metrics(responses: List[dict]) -> dict:
    eval_results = []

    for response in responses:
        prediction = realworld_parse(response["response"])
        score = int(prediction == response["answer"])

        eval_results.append({
            "id": response["id"],
            "question": response["question"],
            "response": response["response"],
            "parsed_response": prediction,
            "answer": response["answer"],
            "score": score
        })

    return eval_results
