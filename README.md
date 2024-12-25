
#### step 1 Download Model and Process Data

##### Download pre-trained model
Download `distilbert-base-uncased-TAS-B`(https://huggingface.co/sentence-transformers/msmarco-distilbert-base-tas-b) and `simlm-msmarco-reranker`(https://huggingface.co/intfloat/simlm-msmarco-reranker) from Hugging Face and place them in the `pretrained_models` directory.

Make sure the following directory exists:
```
pretrained_models/distilbert-base-uncased-TAS-B
pretrained_models/simlm-msmarco-reranker
```


##### We are using the MS Marco dataset released by SimLM.
```
bash scripts/download_msmarco_data.sh
bash scripts/preprocess_data.sh
```

#### step 2 Train Model

Train model
```
bash scripts/runall.sh
```