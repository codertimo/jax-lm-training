import argparse
from typing import List

import apache_beam as beam
import pyarrow
from apache_beam.options.pipeline_options import PipelineOptions
from datasets import load_dataset
from transformers import AutoTokenizer

# fmt: off
parser = argparse.ArgumentParser()
parser.add_argument("--tokenizer-model", type=str, default="gpt2", help="")
parser.add_argument("--min-sequence-length", type=int, default=128, help="")
parser.add_argument("--max-sequence-length", type=int, default=256, help="")
parser.add_argument("--num-special-token-reserved", type=int, default=2, help="")
parser.add_argument("--stride", type=int, default=128, help="")
parser.add_argument("--dataset-name", type=str, default="wikitext", help="")
parser.add_argument("--dataset-sub-name", type=str, default="wikitext-103-v1", help="")
parser.add_argument("--dataset-split-type", type=str, default="test", help="")
parser.add_argument("--output-path", type=str, default="data/wikitext", help="")
# fmt: on


class TokenizingDoFn(beam.DoFn):
    """Tokenize input corpus text and convert into numerical token using GPT2 tokenizer"""

    def __init__(self, tokenizer_model: str):
        self.tokenizer_model = tokenizer_model

    def setup(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_model, fast=True)

    def process(self, text: str):
        yield self.tokenizer.encode(text)


class SplitingChunksDoFN(beam.DoFn):
    def __init__(
        self,
        min_sequence_length: int = 128,
        max_sequence_length: int = 256,
        num_special_token_reserved: int = 2,
        stride: int = 128,
    ):
        self.min_sequence_length = min_sequence_length - num_special_token_reserved
        self.max_sequence_length = max_sequence_length - num_special_token_reserved
        self.stride = stride

    def process(self, input_ids: List[int]):
        for i in range(0, len(input_ids), self.stride):
            # add one more token for label (if max_sequence_length=256, splitted_input_ids=257)
            splitted_sequence = input_ids[i : i + self.max_sequence_length + 1]
            if len(splitted_sequence) > self.min_sequence_length:
                yield splitted_sequence


class PaddingAndPackagingDoFN(beam.DoFn):
    def __init__(
        self,
        sequence_length: int = 256,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        pad_token_id: int = 0,
        ignore_label: int = -100,
    ):
        self.sequence_length = sequence_length
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.ignore_label = ignore_label

    def process(self, sequence: List[int]):
        # 1. add BOS and EOS token to sequence
        sequence = [self.bos_token_id] + sequence + [self.eos_token_id]
        # 2. add padding tokens to sequence
        sequence = sequence + [self.pad_token_id] * (self.sequence_length - len(sequence) + 1)
        # 3. make input_ids, attention_mask, labels
        input_ids = sequence[: self.sequence_length]
        attention_mask = [1.0 if token > 0 else 0.0 for token in input_ids]
        labels = [
            label if label != self.pad_token_id else self.ignore_label
            for label in sequence[1 : self.sequence_length + 1]
        ]
        assert len(input_ids) == len(attention_mask) == len(labels)
        yield {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def main(args: argparse.Namespace):
    dataset = load_dataset(
        args.dataset_name,
        args.dataset_sub_name,
        split=args.dataset_split_type,
    )
    dataset = list(dataset["text"])

    # TODO: support Google Dataflow for blazing fast processing
    with beam.Pipeline(options=PipelineOptions()) as pipeline:
        split_chunk_do_fn = SplitingChunksDoFN(
            min_sequence_length=args.min_sequence_length,
            max_sequence_length=args.max_sequence_length,
            num_special_token_reserved=args.num_special_token_reserved,
            stride=args.stride,
        )

        write_fn = beam.io.WriteToParquet(
            args.output_path,
            pyarrow.schema(
                [
                    ("input_ids", pyarrow.list_(pyarrow.int64())),
                    ("attention_mask", pyarrow.list_(pyarrow.float32())),
                    ("labels", pyarrow.list_(pyarrow.int64())),
                ]
            ),
        )

        # total preprocess pipeline
        _ = (
            pipeline
            | "Create PCollection from dataset" >> beam.Create(dataset)
            | "Tokenize text" >> beam.ParDo(TokenizingDoFn(args.tokenizer_model))
            | "Split sequence into chunks" >> beam.ParDo(split_chunk_do_fn)
            | "Add padding and make model inputs" >> beam.ParDo(PaddingAndPackagingDoFN())
            | "Write to Parquet format" >> write_fn
        )


if __name__ == "__main__":
    main(parser.parse_args())
