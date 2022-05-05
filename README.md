# JAX Language Model Training

> Currently this repo is under construction :) 

## todos

- [X] writing corpus tokenizing and featurizing code with Apache Beam (done at 5/6)
- [ ] writing training code with single GPU
- [ ] writing evaluation code with single GPU
- [ ] writing metric tracking code with weight&bias
- [ ] train LM with single GPU and debug!
- [ ] writing parallelism code for multi-GPUs support.
- [ ] train LM with multi-GPUs and debug!
- [ ] training in TPU and TPU-Pod
- [ ] result comparison GPU, GPUs, TPU, TPU-Pod
- [ ] writing similar code with pytorch and compare the training performance

## Pre-Featurize Corpus with Apache Beam

`jax_lm.preprocess` script download the corpus and process to trainable model input. 
I wrote the Apache Beam pipeline to process corpus and export the output with Apache Parquet.

[Apache Beam](https://beam.apache.org/get-started/beam-overview/) is an open source, unified model for defining both batch and streaming data-parallel processing pipelines.
And [Google Dataflow](https://cloud.google.com/dataflow?hl=ko) automatically parallelize Apache Beam Pipeline which could accelerate huge corpus processing time with blazing speed!
(Well, Dataflow processing code is not implemented in this repo yet. I'll do it soon! )

[Apache Parquet](https://parquet.apache.org) is an open source, column-oriented data file format designed for efficient data storage and retrieval. 
It provides efficient data compression and encoding schemes with enhanced performance to handle complex data in bulk.
[huggingface/dataset](https://huggingface.co/docs/datasets/index) can directly load Parquet file!
 So we just need to used huggingface/dataset when we are training :)

The detail of procedures are described as follow.

1. Download the text corpus from huggingface/dataset 
2. Convert the huggingface/dataset to Apache Beam PCollection
3. Tokenize using huggingface/tokenizer with Apache Beam DoFn
4. Chunk tokens into number of input_ids with Apache Beam DoFn
5. Add BOS, EOS, PAD token and make it to model inputs(feature) with Apache Beam DoFn
6. Write the featurized PCollection into Apache Parquet file
7. Now ready to train model!

```shell
python -m jax_lm.preprocess \
    --tokenizer-model "gpt2" \ # huggingface gpt2 fast tokenizer
    --min-sequence-length 128 \ 
    --max-sequence-length 256 \
    --num-special-token-reserved 2 \ # BOS, EOS
    --stride 128 \
    --dataset-name "wikitext" \ 
    --dataset-sub-name "wikitext-103-v1" \ 
    --dataset-split-type "train" \ 
    --output-path "dataset/wikitext.train"
```

```python
from datasets import Dataset

dataset = Dataset.from_parquet("data/wikitext.train*")
```
