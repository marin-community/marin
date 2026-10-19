import re


def construct_prompt(data: dict, language: str = "python", tokenizer=None, max_token_nums: int = 15800) -> str:
    """
    Construct the prompt for next line prediction.

    :param data: data point from the dataset
    :param language: the language of the code
    :param tokenizer: the tokenizer of the evaluation model
    :param max_token_nums: the maximum number of tokens constraint for the prompt

    :return: the constructed prompt
    """

    # comment symbol for different languages
    comment_symbol = "#" if language == "python" else "//"

    # construct the cross-file prompt and in-file prompt separately
    # cross-file prompt
    cross_file_prompt = f"{comment_symbol} Repo Name: {data['repo_name']}\n"

    for snippet in data["context"]:
        cross_file_prompt += f"{comment_symbol} Path: {snippet['path']}\n{snippet['snippet']}" + "\n\n"

    # in-file prompt
    in_file_prompt = (
        f"{comment_symbol} Path: {data['file_path']}\n{data['import_statement']}\n{data['cropped_code'].rstrip()}\n"
    )

    # if we assign the tokenizer and the max_token_nums, we will truncate the cross-file prompt to meet the constraint
    if tokenizer is not None and max_token_nums is not None:
        cross_file_prompt_token_nums = len(tokenizer.encode(cross_file_prompt))
        in_file_prompt_token_nums = len(tokenizer.encode(in_file_prompt))

        exceed_token_nums = cross_file_prompt_token_nums + in_file_prompt_token_nums - max_token_nums

        if exceed_token_nums > 0:
            # split the cross-file prompt into lines
            cross_file_prompt_lines = cross_file_prompt.split("\n")
            # drop lines from end until the extra token number is less than 0
            for i in range(len(cross_file_prompt_lines) - 1, -1, -1):
                exceed_token_nums -= len(tokenizer.encode(cross_file_prompt_lines[i]))
                if exceed_token_nums < 0:
                    break

            # join the lines back
            cross_file_prompt = "\n".join(cross_file_prompt_lines[:i]) + "\n\n"

    # combine the cross-file prompt and in-file prompt
    prompt = cross_file_prompt + in_file_prompt

    # normalize some empty lines
    prompt = re.sub(r"\n{4,}", "\n\n", prompt)

    return prompt
