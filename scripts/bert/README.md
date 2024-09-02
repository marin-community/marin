# BERT training

The BERT training pipeline consists of two steps: 1) compute labels for a collection of documents based on 
some function of document attributes and 2) train a BERT model on these labels. See `train_bert.py` for an example 
instantiation of this pipeline. To run `train_bert.py` you will need to specify a set of high quality documents 
and a set of low quality documents. The script will then create the appropriate label attributes, convert these attributes into a BERT dataset and train a model on this dataset. For example, running 

```bash
python ./scripts/bert/train_bert.py \
  --pos_doc_path gs://marin-us-central2/documents/marin_instructv1/v1_olmo_mix/text \
  --neg_doc_path gs://marin-us-central2/documents/hello_world_fw/v1.0/quickstart \
  --pos_sampling_rate 0.1 \
  --neg_sampling_rate 1.0 \
  --output_base_path gs://marin-us-central2 \
  --experiment quickstart \
  --config_path ./scripts/bert/configs/train_bert.yaml
```
will train a BERT classifier using OLMo instruct data as high quality examples and Fineweb as low quality examples and 
the additional training hyperparameters specified in `train_bert.yaml` 
and save the results to `gs://marin-us-central2/classifiers/quickstart`.
Typically, one should choose the sampling rates to balance the number of high quality and low quality examples (and also 
to ensure that the entire dataset fits into worker memory).