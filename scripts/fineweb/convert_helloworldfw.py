import json

from tqdm import tqdm
from datasets import load_dataset

from marin.web.convert import convert_page


def main():
    fw_subset = load_dataset("skaramcheti/hello_world_fw")["train"]
    
    with open("hello_world_fw_extracted.jsonl", "w") as f:
        for example in tqdm(fw_subset):
            html = example["text"]

            readability_text = convert_page(html, extract_method="readability")["content"]
            resiliparse_text = convert_page(html, extract_method="resiliparse")["content"]
            trafilatura_text = convert_page(html, extract_method="trafilatura")["content"]

            f.write(json.dumps({
                "id": example["id"],
                "html": html,
                "readability": readability_text,
                "resiliparse": resiliparse_text,
                "trafilatura": trafilatura_text
            }) + "\n")


if __name__=="__main__":
    main()