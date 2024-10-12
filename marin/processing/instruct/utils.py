from marin.web.convert import convert_page


def process_instruction(source, uuid, message_dict, instruction_format="tulu"):
    """
    Process the instruction for Tulu language.

    Args:
        source (str): The source of the instruction.
        uuid (int): The ID of the instruction.
        message_dict (dict): A dictionary containing the messages.
        instruction_format (str): The format of the instruction. Defaults to "tulu" and that's
        the only option for now until we get more datasets.

    Returns:
        tuple: A tuple containing the Markdown content and the plain text content.
    """
    title = f"Instruction Doc: {id} Source: {source}"
    md_content = f"# {title} \n\n"
    txt_content = f"{title}\n\n"

    for message in message_dict["messages"]:
        role = message.get("role", "")
        text = message.get("content", "")

        out = convert_page(text, url="")
        title = out.get(
            "title",
        )
        md = out["content"]

        if not role:
            raise ValueError("Role cannot be empty.")

        md_content += f" \n\n ## **{role}** \n {md}  \n\n"
        txt_content += f"\n\n{role.upper()}\n{text}\n\n"

    return md_content, txt_content
