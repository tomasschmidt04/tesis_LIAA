# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning a 🤗 Transformers model for sequence classification on GLUE."""
import argparse
import json
import logging
import math
import os
import sys
import random
from pathlib import Path
from collections import deque
import numpy as np
import pandas as pd
import pickle
from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, Tuple

import datasets
import evaluate
import torch
from accelerate import Accelerator
from datasets import ClassLabel, load_dataset
from huggingface_hub import Repository, create_repo
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from transformers import (
	AutoConfig,
	AutoModelForSequenceClassification,
	AutoTokenizer,
	EvalPrediction,
	DataCollatorWithPadding,
	HfArgumentParser,
	PretrainedConfig,
	Trainer,
	TrainingArguments,
	SchedulerType,
	default_data_collator,
	get_scheduler,
	set_seed,
)
from transformers.utils import check_min_version, get_full_repo_name, send_example_telemetry
from transformers.utils.versions import require_version
try:
	from trainers import OurTrainer
except ImportError:
	OurTrainer = Trainer


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.27.0.dev0")
logger = logging.getLogger(__name__)
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

task_to_keys = {
	"cola": ("sentence", None),
	"mnli": ("premise", "hypothesis"),
	"mrpc": ("sentence1", "sentence2"),
	"qnli": ("question", "sentence"),
	"qqp": ("question1", "question2"),
	"rte": ("sentence1", "sentence2"),
	"sst2": ("sentence", None),
	"stsb": ("sentence1", "sentence2"),
	"wnli": ("sentence1", "sentence2"),
	"trec": ("text", None),
	"ag_news": ("text", None),
	"rioplatense_hate_binary": ("text", None),
}

RIOPLATENSE_RAW_BASE_URL = "https://raw.githubusercontent.com/finiteautomata/rioplatense_hate_speech/refs/heads/main/data"
RIOPLATENSE_TRAIN_FILES = ["test_01.csv", "test_02.csv", "test_03.csv"]
RIOPLATENSE_VALIDATION_FILE = "test_04.csv"
RIOPLATENSE_TEST_FILE = "test_05.csv"
RIOPLATENSE_HATE_CATEGORY_COLUMNS = [
	"CALLS",
	"WOMEN",
	"LGBTI",
	"RACISM",
	"CLASS",
	"POLITICS",
	"DISABLED",
	"APPEARANCE",
	"CRIMINAL",
]

def count_parameters(model):
	return sum(p.numel() for p in model.parameters() if p.requires_grad)

@dataclass
class DataTrainingArguments:
	"""
	Arguments pertaining to what data we are going to input our model for training and eval.

	Using `HfArgumentParser` we can turn this class
	into argparse arguments to be able to specify them on
	the command line.
	"""

	task_name: Optional[str] = field(
		default=None,
		metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
	)
	max_seq_length: int = field(
		default=128,
		metadata={
			"help": (
				"The maximum total input sequence length after tokenization. Sequences longer "
				"than this will be truncated, sequences shorter will be padded."
			)
		},
	)
	overwrite_cache: bool = field(
		default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
	)
	pad_to_max_length: bool = field(
		default=False,
		metadata={
			"help": (
				"Whether to pad all samples to `max_seq_length`. "
				"If False, will pad the samples dynamically when batching to the maximum length in the batch."
			)
		},
	)
	max_train_samples: Optional[int] = field(
		default=None,
		metadata={
			"help": (
				"For debugging purposes or quicker training, truncate the number of training examples to this "
				"value if set."
			)
		},
	)
	max_eval_samples: Optional[int] = field(
		default=None,
		metadata={
			"help": (
				"For debugging purposes or quicker training, truncate the number of evaluation examples to this "
				"value if set."
			)
		},
	)
	max_predict_samples: Optional[int] = field(
		default=None,
		metadata={
			"help": (
				"For debugging purposes or quicker training, truncate the number of prediction examples to this "
				"value if set."
			)
		},
	)
	low_resource_data_seed: Optional[int] = field(
        default=42,
        metadata={
            "help": "seed for selecting subset of the dataset if not using all."
        },
    )
	train_as_val: bool = field(
		default=True,
		metadata={"help": "if True, sample 1k from train as val"},
	)

	label_name: Optional[str] = field(
		default='label',
		metadata={"help": "The name of the label to use"},
	)
	dataset_path: Optional[str] = field(
		default=None,
		metadata={"help": "Optional local path to finiteautomata/rioplatense_hate_speech repo or its data directory."},
	)
	use_context: bool = field(
		default=False,
		metadata={"help": "For rioplatense_hate_binary, concatenate title/context_tweet with text when True."},
	)

	def __post_init__(self):
		if self.task_name is not None:
			self.task_name = self.task_name.lower()
			if self.task_name not in task_to_keys.keys():
				raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))


@dataclass
class ModelArguments:
	"""
	Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
	"""

	model_name_or_path: str = field(
		metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
	)
	config_name: Optional[str] = field(
		default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
	)
	tokenizer_name: Optional[str] = field(
		default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
	)
	cache_dir: Optional[str] = field(
		default=None,
		metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
	)
	use_fast_tokenizer: bool = field(
		default=True,
		metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
	)
	model_revision: str = field(
		default="main",
		metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
	)
	use_auth_token: bool = field(
		default=False,
		metadata={
			"help": "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead."
		},
	)
	# new arguments


def resolve_rioplatense_data_file(dataset_path: Optional[str], file_name: str) -> str:
	if dataset_path is None:
		return f"{RIOPLATENSE_RAW_BASE_URL}/{file_name}"

	root = Path(dataset_path)
	candidates = [root / file_name, root / "data" / file_name]
	for candidate in candidates:
		if candidate.exists():
			return str(candidate)
	raise FileNotFoundError(
		f"Could not find {file_name} under --dataset_path={dataset_path}. "
		f"Tried: {', '.join(str(candidate) for candidate in candidates)}"
	)


def load_rioplatense_hate_binary_dataset(data_args: DataTrainingArguments):
	data_files = {
		"train": [resolve_rioplatense_data_file(data_args.dataset_path, name) for name in RIOPLATENSE_TRAIN_FILES],
		"validation": resolve_rioplatense_data_file(data_args.dataset_path, RIOPLATENSE_VALIDATION_FILE),
		"test": resolve_rioplatense_data_file(data_args.dataset_path, RIOPLATENSE_TEST_FILE),
	}
	raw = load_dataset("csv", data_files=data_files)

	def normalize_example(example):
		text = "" if example.get("text") is None else str(example["text"])
		if data_args.use_context:
			context_parts = []
			for column in ["title", "context_tweet"]:
				value = example.get(column)
				if value is not None and str(value).strip():
					context_parts.append(str(value))
			if context_parts:
				text = " ".join(context_parts + [text])

		hateful_value = int(example["HATEFUL"])
		category_sum = 0
		for column in RIOPLATENSE_HATE_CATEGORY_COLUMNS:
			value = example.get(column)
			if value is not None:
				category_sum += int(value)

		return {
			"text": text,
			"label": hateful_value,
			"hate_category_or_label": int(category_sum > 0),
		}

	normalized = raw.map(normalize_example, load_from_cache_file=not data_args.overwrite_cache)
	keep_columns = ["text", "label", "hate_category_or_label"]
	for split in normalized:
		remove_columns = [column for column in normalized[split].column_names if column not in keep_columns]
		normalized[split] = normalized[split].remove_columns(remove_columns)
		normalized[split] = normalized[split].cast_column("label", ClassLabel(names=["no_hate", "hate"]))
	return normalized


def class_distribution(dataset, label_names: List[str]) -> Dict[str, int]:
	if dataset is None:
		return {}
	labels = dataset["label"]
	return {
		label_names[label_id]: int(sum(1 for value in labels if int(value) == label_id))
		for label_id in range(len(label_names))
	}


def stratified_select_dataset(dataset, sample_size: Optional[int], seed: int):
	if sample_size is None or sample_size <= 0 or len(dataset) <= sample_size:
		return dataset, dataset.select(range(0))

	label_values = [int(value) for value in dataset["label"]]
	indices_by_label: Dict[int, List[int]] = {}
	for index, label in enumerate(label_values):
		indices_by_label.setdefault(label, []).append(index)

	rng = random.Random(seed)
	for indices in indices_by_label.values():
		rng.shuffle(indices)

	total = len(dataset)
	target_counts: Dict[int, int] = {}
	remaining = sample_size
	labels_sorted = sorted(indices_by_label.keys())
	for label in labels_sorted:
		proportion = len(indices_by_label[label]) / total
		count = min(len(indices_by_label[label]), int(math.floor(sample_size * proportion)))
		target_counts[label] = count
		remaining -= count

	while remaining > 0:
		added = False
		for label in sorted(labels_sorted, key=lambda key: len(indices_by_label[key]) - target_counts[key], reverse=True):
			if remaining <= 0:
				break
			if target_counts[label] < len(indices_by_label[label]):
				target_counts[label] += 1
				remaining -= 1
				added = True
		if not added:
			break

	selected_indices = []
	holdout_indices = []
	for label in labels_sorted:
		count = target_counts[label]
		selected_indices.extend(indices_by_label[label][:count])
		holdout_indices.extend(indices_by_label[label][count:])

	rng.shuffle(selected_indices)
	rng.shuffle(holdout_indices)
	return dataset.select(selected_indices), dataset.select(holdout_indices)


def compute_binary_classification_metrics(predictions, labels):
	true_positive = int(((predictions == 1) & (labels == 1)).sum())
	false_positive = int(((predictions == 1) & (labels == 0)).sum())
	false_negative = int(((predictions == 0) & (labels == 1)).sum())
	true_negative = int(((predictions == 0) & (labels == 0)).sum())

	def safe_divide(numerator, denominator):
		return float(numerator / denominator) if denominator else 0.0

	accuracy = safe_divide(true_positive + true_negative, len(labels))
	precision_hate = safe_divide(true_positive, true_positive + false_positive)
	recall_hate = safe_divide(true_positive, true_positive + false_negative)
	f1_hate = safe_divide(2 * precision_hate * recall_hate, precision_hate + recall_hate)

	precision_no_hate = safe_divide(true_negative, true_negative + false_negative)
	recall_no_hate = safe_divide(true_negative, true_negative + false_positive)
	f1_no_hate = safe_divide(2 * precision_no_hate * recall_no_hate, precision_no_hate + recall_no_hate)

	return {
		"accuracy": accuracy,
		"macro_f1": float((f1_hate + f1_no_hate) / 2),
		"precision": precision_hate,
		"recall": recall_hate,
		"f1": f1_hate,
		"f1_hate": f1_hate,
		"f1_no_hate": f1_no_hate,
	}


def write_json(path: str, data: Dict[str, object]):
	def json_default(value):
		if isinstance(value, np.integer):
			return int(value)
		if isinstance(value, np.floating):
			return float(value)
		if isinstance(value, np.ndarray):
			return value.tolist()
		raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

	Path(path).parent.mkdir(parents=True, exist_ok=True)
	with open(path, "w", encoding="utf-8") as handle:
		json.dump(data, handle, indent=2, ensure_ascii=False, default=json_default)



def main():
	# See all possible arguments in src/transformers/training_args.py
	# or by passing the --help flag to this script.
	# We now keep distinct sets of args, for a cleaner separation of concerns.

	parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
	if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
		# If we pass only one argument to the script and it's the path to a json file,
		# let's parse it to get our arguments.
		model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
	else:
		model_args, data_args, training_args = parser.parse_args_into_dataclasses()

	# Setup logging
	logging.basicConfig(
		format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
		datefmt="%m/%d/%Y %H:%M:%S",
		handlers=[logging.StreamHandler(sys.stdout)],
	)

	if training_args.should_log:
		# The default of training_args.log_level is passive, so we set log level at info here to have that default.
		transformers.utils.logging.set_verbosity_info()

	log_level = training_args.get_process_log_level()
	logger.setLevel(log_level)
	datasets.utils.logging.set_verbosity(log_level)
	transformers.utils.logging.set_verbosity(log_level)
	transformers.utils.logging.enable_default_handler()
	transformers.utils.logging.enable_explicit_format()

	# Log on each process the small summary:
	logger.warning(
		f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
		+ f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
	)
	logger.info(f"Training/evaluation parameters {training_args}")

	is_rioplatense_hate_binary = data_args.task_name == "rioplatense_hate_binary"

	# Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
	# or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).
	if data_args.task_name is not None:
		if is_rioplatense_hate_binary:
			raw_datasets = load_rioplatense_hate_binary_dataset(data_args)
		else:
			# download the dataset.
			raw_datasets = load_dataset("glue", data_args.task_name)


	# Labels
	if data_args.task_name is not None:
		is_regression = data_args.task_name == "stsb"
		if not is_regression:
			if is_rioplatense_hate_binary:
				label_list = ["no_hate", "hate"]
			else:
				label_list = raw_datasets["train"].features["label"].names
			num_labels = len(label_list)
		else:
			num_labels = 1


	# Set seed before initializing model.
	set_seed(training_args.seed)

	# Load pretrained model and tokenizer
	# download model & vocab.
	config_kwargs = {
		"num_labels": num_labels,
		"finetuning_task": data_args.task_name,
		"cache_dir": model_args.cache_dir,
		"revision": model_args.model_revision,
	}
	if is_rioplatense_hate_binary:
		config_kwargs["label2id"] = {"no_hate": 0, "hate": 1}
		config_kwargs["id2label"] = {0: "no_hate", 1: "hate"}
	config = AutoConfig.from_pretrained(
		model_args.config_name if model_args.config_name else model_args.model_name_or_path,
		**config_kwargs,
	)
	tokenizer = AutoTokenizer.from_pretrained(
		model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
		cache_dir=model_args.cache_dir,
		use_fast=model_args.use_fast_tokenizer,
		revision=model_args.model_revision,
	)
	model = AutoModelForSequenceClassification.from_pretrained(
		model_args.model_name_or_path,
		from_tf=bool(".ckpt" in model_args.model_name_or_path),
		config=config,
		cache_dir=model_args.cache_dir,
		revision=model_args.model_revision,
		ignore_mismatched_sizes=True,
	)

	# Preprocessing the raw_datasets
	if data_args.task_name is not None:
		sentence1_key, sentence2_key = task_to_keys[data_args.task_name]


	# Some models have set the order of the labels to use, so let's make sure we do use it.
	label_to_id = None
	if is_rioplatense_hate_binary:
		model.config.label2id = {"no_hate": 0, "hate": 1}
		model.config.id2label = {0: "no_hate", 1: "hate"}
	elif (
		model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
		and data_args.task_name is not None
		and not is_regression
	):
		# Some have all caps in their config, some don't.
		label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
		if sorted(label_name_to_id.keys()) == sorted(label_list):
			logger.info(
				f"The configuration of the model provided the following label correspondence: {label_name_to_id}. "
				"Using it!"
			)
			label_to_id = {i: label_name_to_id[label_list[i]] for i in range(num_labels)}
		else:
			logger.info(
				"Your model seems to have been trained with labels, but they don't match the dataset: ",
				f"model labels: {sorted(label_name_to_id.keys())}, dataset labels: {sorted(label_list)}."
				"\nIgnoring the model labels as a result.",
			)
	elif data_args.task_name is None and not is_regression:
		label_to_id = {v: i for i, v in enumerate(label_list)}

	if is_rioplatense_hate_binary:
		pass
	elif label_to_id is not None:
		model.config.label2id = label_to_id
		model.config.id2label = {id: label for label, id in config.label2id.items()}
	elif data_args.task_name is not None and not is_regression:
		model.config.label2id = {l: i for i, l in enumerate(label_list)}
		model.config.id2label = {id: label for label, id in config.label2id.items()}


	def preprocess_function(examples):

		# Tokenize the texts
		texts = (
			(examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
		)
		result = tokenizer(*texts,
							padding="max_length" if data_args.pad_to_max_length else False,
							max_length=data_args.max_seq_length,
							truncation=True,
							)

		# Map labels to IDs (not necessary for GLUE tasks)
		if label_to_id is not None and "label" in examples:
			result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
		return result


	raw_datasets = raw_datasets.map(
		preprocess_function,
		batched=True,
		load_from_cache_file=not data_args.overwrite_cache,
		#remove_columns=raw_datasets["train"].column_names,
		desc="Running tokenizer on dataset",
	)

	if training_args.do_train:
		if "train" not in raw_datasets:
			raise ValueError("--do_train requires a train dataset")
		train_dataset = raw_datasets["train"]
		train_holdout_dataset = train_dataset.select(range(0))
		if data_args.max_train_samples is not None:
			logger.warning(f'shuffling training set w. seed {data_args.low_resource_data_seed}!')
			if is_rioplatense_hate_binary:
				train_dataset, train_holdout_dataset = stratified_select_dataset(
					train_dataset,
					data_args.max_train_samples,
					data_args.low_resource_data_seed,
				)
			else:
				train_dataset_all = train_dataset.shuffle(seed=data_args.low_resource_data_seed)
				train_dataset = train_dataset_all.select(range(data_args.max_train_samples))

	if training_args.do_eval:
		if "validation" not in raw_datasets and "validation_matched" not in raw_datasets:
			raise ValueError("--do_eval requires a validation dataset")
		eval_dataset = raw_datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]
		if data_args.max_eval_samples is not None:
			eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

	if training_args.do_predict:
		if "test" not in raw_datasets and "test_matched" not in raw_datasets:
			raise ValueError("--do_predict requires a test dataset")
		test_dataset = raw_datasets["test_matched" if data_args.task_name == "mnli" else "test"]

	if data_args.train_as_val:
		if is_rioplatense_hate_binary:
			eval_dataset, _ = stratified_select_dataset(
				train_holdout_dataset,
				data_args.max_eval_samples if data_args.max_eval_samples is not None else 1000,
				data_args.low_resource_data_seed + 1,
			)
		else:
			test_dataset = eval_dataset
			eval_dataset = train_dataset_all.select(range(data_args.max_train_samples, data_args.max_train_samples + 1000))

	if is_rioplatense_hate_binary and training_args.local_rank in [-1, 0]:
		os.makedirs(training_args.output_dir, exist_ok=True)
		distribution = {
			"train": class_distribution(train_dataset if training_args.do_train else None, label_list),
			"validation": class_distribution(eval_dataset if training_args.do_eval else None, label_list),
			"test": class_distribution(test_dataset if training_args.do_predict else None, label_list),
		}
		write_json(os.path.join(training_args.output_dir, "class_distribution.json"), distribution)
		write_json(
			os.path.join(training_args.output_dir, "model_init_check.json"),
			{
				"task_name": data_args.task_name,
				"model_name_or_path": model_args.model_name_or_path,
				"output_dir": training_args.output_dir,
				"num_labels": num_labels,
				"label2id": {"no_hate": 0, "hate": 1},
				"id2label": {"0": "no_hate", "1": "hate"},
				"dataset_path": data_args.dataset_path,
				"use_context": data_args.use_context,
				"train_as_val": data_args.train_as_val,
				"max_train_samples": data_args.max_train_samples,
				"low_resource_data_seed": data_args.low_resource_data_seed,
				"split_sizes": {
					"train": len(train_dataset) if training_args.do_train else 0,
					"validation": len(eval_dataset) if training_args.do_eval else 0,
					"test": len(test_dataset) if training_args.do_predict else 0,
				},
				"class_distribution": distribution,
			},
		)
		logger.warning(
			"rioplatense_hate_binary init: "
			f"model_name_or_path={model_args.model_name_or_path}, "
			f"output_dir={training_args.output_dir}, "
			f"num_labels={num_labels}, label2id={{'no_hate': 0, 'hate': 1}}, "
			f"sizes={{'train': {len(train_dataset) if training_args.do_train else 0}, "
			f"'validation': {len(eval_dataset) if training_args.do_eval else 0}, "
			f"'test': {len(test_dataset) if training_args.do_predict else 0}}}"
		)
		logger.info(f"rioplatense_hate_binary class distribution: {distribution}")

	# Get the metric function
	if is_rioplatense_hate_binary:
		metric = None
	elif data_args.task_name is not None:
		metric = evaluate.load("glue", data_args.task_name)
	elif is_regression:
		metric = evaluate.load("mse")
	else:
		metric = evaluate.load("accuracy")


	# You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
	# predictions and label_ids field) and has to return a dictionary string to float.
	def compute_metrics(p: EvalPrediction):
		preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
		preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
		if is_rioplatense_hate_binary:
			result = compute_binary_classification_metrics(preds, p.label_ids)
		else:
			result = metric.compute(predictions=preds, references=p.label_ids)
		if len(result) > 1:
			result["combined_score"] = np.mean(list(result.values())).item()
		return result

	# Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
	# we already did the padding.
	if data_args.pad_to_max_length:
		data_collator = default_data_collator
	elif training_args.fp16:
		data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
	else:
		data_collator = None

	# Initialize our Trainer
	trainer = OurTrainer(
		model=model,
		args=training_args,
		train_dataset=train_dataset if training_args.do_train else None,
		eval_dataset=eval_dataset if training_args.do_eval else None,
		compute_metrics=compute_metrics,
		tokenizer=tokenizer,
		data_collator=data_collator,
	)
	trainer.model_args = model_args
	# Training
	if training_args.do_train:
		train_result = trainer.train()
		metrics = train_result.metrics
		max_train_samples = (
			data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
		)
		metrics["train_samples"] = min(max_train_samples, len(train_dataset))

		#trainer.save_model()  # Saves the tokenizer too for easy upload

		output_train_file = os.path.join(training_args.output_dir, "train_results.txt")
		if trainer.is_world_process_zero():
			with open(output_train_file, "w") as writer:
				logger.info("***** Train results *****")
				for key, value in sorted(train_result.metrics.items()):
					logger.info(f"  {key} = {value}")
					writer.write(f"{key} = {value}\n")

			# Need to save the state, since Trainer.save_model saves only the tokenizer with the model
			trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))

	if is_rioplatense_hate_binary and training_args.do_eval:
		logger.info("*** Evaluation ***")
		eval_metrics = trainer.evaluate(eval_dataset=eval_dataset, metric_key_prefix="eval")
		if trainer.is_world_process_zero():
			write_json(os.path.join(training_args.output_dir, "eval_results.json"), eval_metrics)

	if training_args.do_predict:
		logger.info("*** Test ***")
		# Loop to handle MNLI double evaluation (matched, mis-matched)
		tasks = [data_args.task_name]
		test_datasets = [test_dataset]
		# not evaluating test_mismatched
		# if data_args.task_name == "mnli":
		#     tasks.append("mnli-mm")
		#     test_datasets.append(datasets["validation_mismatched"])

		for test_dataset, task in zip(test_datasets, tasks):
			if is_rioplatense_hate_binary:
				test_metrics = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
				if trainer.is_world_process_zero():
					write_json(os.path.join(training_args.output_dir, "test_results.json"), test_metrics)
				prediction_output = trainer.predict(test_dataset=test_dataset, metric_key_prefix="predict")
				predictions = np.argmax(prediction_output.predictions, axis=1)
				output_test_file = os.path.join(training_args.output_dir, f"test_results_{task}.txt")
				if trainer.is_world_process_zero():
					with open(output_test_file, "w") as writer:
						logger.info(f"***** Test predictions {task} *****")
						writer.write("index\tprediction\n")
						for index, item in enumerate(predictions):
							writer.write(f"{index}\t{label_list[item]}\n")
				continue

			# only do_predict if train_as_val
			# Removing the `label` columns because it contains -1 and Trainer won't like that.
			test_dataset = test_dataset.remove_columns("label")
			predictions = trainer.predict(test_dataset=test_dataset, metric_key_prefix="predict").predictions
			predictions = np.squeeze(predictions) if is_regression else np.argmax(predictions, axis=1)

			output_test_file = os.path.join(training_args.output_dir, f"test_results_{task}.txt")
			if trainer.is_world_process_zero():
				with open(output_test_file, "w") as writer:
					logger.info(f"***** Test results {task} *****")
					writer.write("index\tprediction\n")
					for index, item in enumerate(predictions):
						if is_regression:
							writer.write(f"{index}\t{item:3.3f}\n")
						else:
							item = label_list[item]
							writer.write(f"{index}\t{item}\n")




def _mp_fn(index):
	# For xla_spawn (TPUs)
	main()


if __name__ == "__main__":
	main()
