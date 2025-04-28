def whitebackground_prompt() -> str:
    prompt = '{question}\n Answer as concise as possible, just a few words.'

    return prompt


def complexgrid_prompt() -> str:
    prompt = 'The image is composed of multiple sub-image. \
        The left upper corner is row 1 column 1. \
        We also add the row and column numbers under each image. \
        You need to identify the sub-image that best suit the caption: "{caption}". \
        And return the row and column ids. \
        Give answer in exact this format, <row>ROW_ANSWER</row><col>COL_ANSWER</col>. \
        For example, <row>3</row><col>2</col>.'

    return prompt
