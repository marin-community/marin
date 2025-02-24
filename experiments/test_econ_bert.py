import ray

from marin.processing.classification.classifier import AutoClassifier

MODEL_PATH = "/opt/gcsfuse_mount/economic-bert-large-8/checkpoint-651"


@ray.remote(resources={"TPU": 1, "TPU-v6e-8-head": 1})
def test_econ_bert():
    # model = AutoModelForSequenceClassification.from_pretrained(
    #     MODEL_PATH, trust_remote_code=True, output_hidden_states=False
    # )
    # tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    texts = [
        "The capital of China is Beijing.",
        "This is the best thing ever.",
        "The economy of the United States is the largest in the world. The GDP of the United States is $20 trillion."
        "China has a GDP of $14 trillion.",
        "What's up with the stock market?",
        "President Trump's pick as labor secretary faced pointed questions from both parties at her Senate confirmation hearing on "  # noqa: E501
        "Wednesday over her past support for pro-union legislation, an issue that could complicate her nomination. "
        "The nominee, Lori Chavez-DeRemer, a former Republican congresswoman, was pressed repeatedly about her stand on the Protecting the Right to Organize Act,"  # noqa: E501
        "known as the PRO Act — a sweeping labor bill that sought to strengthen collective bargaining rights. She was a co-sponsor of the measure, a top Democratic priority "  # noqa: E501
        "that has yet to win passage, and one of few Republicans to back it. "
        "Asked if she continued to support it, Ms. Chavez-DeRemer demurred, saying she was no longer in Congress and would support Mr. Trump's agenda. "  # noqa: E501
        '"I do not believe that the secretary of labor should write the laws," she told the Senate Health, Education, Labor and Pensions '  # noqa: E501
        "Committee, which conducted the hearing. "
        "It will be up to the Congress to write those laws and to work together. What I believe is that the American worker deserves to be paid attention to.",  # noqa: E501
        "Macroeconomics is a branch of economics that deals with the performance, structure, behavior, and decision-making of an economy as a whole.[1]"  # noqa: E501
        "This includes regional, national, and global economies.[2][3] Macroeconomists study topics such as output/GDP (gross domestic product) and national income, "  # noqa: E501
        "unemployment (including unemployment rates), price indices and inflation, consumption, saving, investment, energy, international trade, and international finance. "  # noqa: E501
        "Macroeconomics and microeconomics are the two most general fields in economics.[4] The focus of macroeconomics is often on a country (or larger entities like the whole world) "  # noqa: E501
        "and how its markets interact to produce large-scale phenomena that economists refer to as aggregate variables. In microeconomics the focus of analysis is often a single market, "  # noqa: E501
        'such as whether changes in supply or demand are to blame for price increases in the oil and automotive sectors. From introductory classes in "principles of economics" through doctoral '  # noqa: E501
        "studies, the macro/micro divide is institutionalized in the field of economics. Most economists identify as either macro- or micro-economists.",  # noqa: E501
    ]
    batch = {"text": texts}
    # inputs = tokenizer(texts, return_tensors="pt", truncation=True, max_length=512, padding=True)
    # outputs = model(**inputs)
    # print(outputs.logits)
    classifier = AutoClassifier(MODEL_PATH, "label", "gte", max_length=512)
    print(classifier(batch))


if __name__ == "__main__":
    x = ray.get(test_econ_bert.remote())
