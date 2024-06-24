from marin.processing.fasttext.download_dolma_classifier import download_file

class BaseQualityClassifier:
    def __init__(self, model):
        self.model = model
    
    def predict(self, document):
        raise NotImplementedError

    def __call__(self, batch):
        raise NotImplementedError

class DummyQualityClassifier(BaseQualityClassifier):
    def __init__(self, model):
        self.model = model

    def predict(self, document):
        label, score = "__label__test", 1.0
        return label, score
    
    def __call__(self, row):
        label, score = self.predict(row["text"])
        row.update({
            "label": label,
            "score": score,
        })
        return row


class FasttextQualityClassifier(BaseQualityClassifier):
    def __init__(self, model):
        import fasttext

        # NOTE(chris): 1 try theoretically should work but there are sometimes some SystemExceptions from either FastText loading or Huggingface about metadata
        MAX_RETRIES = 3
        for i in range(MAX_RETRIES):
            try:
                model_path = download_file()
                self.model = fasttext.load_model(model_path)
                break
            except Exception as e:
                print("Failed to load model")            

    def predict(self, document):
        return self.model.predict(document)

    def __call__(self, row):
        if row["text"]:
            text = row["text"].replace("\n", " ")
        else:
            text = ""

        label_arr, score_arr = self.predict(text)
        fasttext_quality_dict = {}
        for label, score in zip(label_arr, score_arr):
            fasttext_quality_dict.update({
                label: score
            })

        row.update({
            "attributes" : {
                "fasttext-quality" : fasttext_quality_dict
            }
        })

        return row