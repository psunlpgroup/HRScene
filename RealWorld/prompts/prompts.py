def realworld_prompt() -> str:
    prompt: str = '{question}\n Give answer in exact this format, \
        <ans>ANSWER</ans>, no redundant words. For example: <ans>A</ans>.'

    return prompt
