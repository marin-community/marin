from enum import Enum


class LMStyle(Enum):
    OpenAIChat = "OpenAIChat"
    Claude = "Claude"  # Claude 1 and Claude 2
    Claude3 = "Claude3"
    Gemini = "Gemini"
    MistralWeb = "MistralWeb"

    GenericBase = "GenericBase"

    DeepSeekCodeInstruct = "DeepSeekCodeInstruct"
    CodeLLaMaInstruct = "CodeLLaMaInstruct"
    StarCoderInstruct = "StarCoderInstruct"

    Phind = "Phind"
    WizardCoder = "WizardCoder"
    MagiCoder = "MagiCoder"
    OC = "OC"

    Qwen1point5 = "Qwen1point5"
    Smaug2 = "Smaug2"

    LLaMa3 = "LLaMa3"
