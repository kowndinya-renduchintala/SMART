####################################################################################################
# Author: <ANONYMOUS>

# This script performs instruction tuning on auto-regressive language models like Mistral, Llama2, Falcon, GPT-2 etc.

# The dataset for instruction tuning(via the 🤗 datasets library) MUST be in the following format:
# {
#     "train": [
#         {
#             "prompt": "prompt text",
#             "response": "response text"
#         },
#         ...
#     ],
#     "validation": [
#         {
#             "prompt": "prompt text",
#             "response": "response text"
#         },
#         ...
#     ],
# }

# TODO: Add support for deepspeed and FSDP later
####################################################################################################
import os
import gc
import threading
import psutil
import json
import copy
import random
from pathlib import Path
import argparse
import logging
import math
from typing import List, Dict
from dataclasses import dataclass
import torch
from torch.utils.data import DataLoader
from torch.nn.utils import rnn
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
import datasets
from datasets import load_dataset, load_from_disk
from huggingface_hub import Repository, create_repo
from peft import AutoPeftModelForCausalLM, LoraConfig, TaskType, get_peft_model
import transformers
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerBase,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    SchedulerType,
    get_scheduler,
)
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizerBase

logger = get_logger(__name__)

IGNORE_INDEX=-100
TORCH_DTYPES={
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "auto": "auto"
}

# Converting Bytes to Megabytes
def b2mb(x):
    return int(x / 2**20)

# This context manager is used to track the peak memory usage of the process
class TorchTracemalloc:
    def __enter__(self):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()  # reset the peak gauge to zero
        self.begin = torch.cuda.memory_allocated()
        self.process = psutil.Process()

        self.cpu_begin = self.cpu_mem_used()
        self.peak_monitoring = True
        peak_monitor_thread = threading.Thread(target=self.peak_monitor_func)
        peak_monitor_thread.daemon = True
        peak_monitor_thread.start()
        return self

    def cpu_mem_used(self):
        """get resident set size memory for the current process"""
        return self.process.memory_info().rss

    def peak_monitor_func(self):
        self.cpu_peak = -1

        while True:
            self.cpu_peak = max(self.cpu_mem_used(), self.cpu_peak)

            if not self.peak_monitoring:
                break

    def __exit__(self, *exc):
        self.peak_monitoring = False

        gc.collect()
        torch.cuda.empty_cache()
        self.end = torch.cuda.memory_allocated()
        self.peak = torch.cuda.max_memory_allocated()
        self.used = b2mb(self.end - self.begin)
        self.peaked = b2mb(self.peak - self.begin)

        self.cpu_end = self.cpu_mem_used()
        self.cpu_used = b2mb(self.cpu_end - self.cpu_begin)
        self.cpu_peaked = b2mb(self.cpu_peak - self.cpu_begin)

@dataclass
class DataCollatorForInstructionTuning:
    """Collate examples for instruction tuning."""

    tokenizer: PreTrainedTokenizerBase
    
    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, attention_mask, labels = tuple([torch.tensor(feature[key]) for feature in features] for key in ["input_ids", "attention_mask", "labels"])
        input_ids=rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask=rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
        labels=rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

def parse_args():
    parser=argparse.ArgumentParser(description="Instruction Tune an auto-regressive language model like Mistral, Llama2, Falcon, GPT-2 etc.")

    # Parameters related to loading data
    parser.add_argument("--load_data_from_disk", action="store_true", help="Whether to load data from disk")
    parser.add_argument("--dataset_name_or_path", default="anonymous/flan2022", help="Dataset name(in 🤗 datasets hub) or path to a local dataset")
    
    # Parameteres related to loading the model(including BitsAndBytes configuration)
    parser.add_argument("--trust_remote_code", action="store_true", help="Whether to trust remote code")
    parser.add_argument("--hf_access_token", type=str, default="", help="HuggingFace access token")
    parser.add_argument("--model_name_or_path", type=str, default="mistralai/Mistral-7B-v0.1", help="Model name or path to local checkpoint")
    parser.add_argument("--sliding_window", type=int, default=4096, help="Sliding window size in case of Mistral")
    parser.add_argument("--torch_dtype", choices=["float32", "float16", "bfloat16", "auto"], default="auto", help="Torch dtype")
    parser.add_argument("--use_flash_attention_2", action="store_true", help="Whether to use FlashAttention2")

    parser.add_argument("--load_in_8bit", action="store_true", help="Whether to enable 8-bit quantization in 8bit")
    parser.add_argument("--load_in_4bit", action="store_true", help="Whether to enable 4-bit quantization in 4bit")
    parser.add_argument("--llm_int8_threshold", type=float, default=6.0, help="Outlier threshold for outlier detection as described in LLM.int8()")
    parser.add_argument("--llm_int8_skip_modules", type=str, default=None, help="Comma separated list of modules to skip in LLM.int8()")
    parser.add_argument("--llm_int8_enable_fp32_cpu_offload", action="store_true", help="Whether to enable fp32 cpu offload in LLM.int8()")
    parser.add_argument("--llm_int8_has_fp16_weight", action="store_true", help="Whether to run LLM.int8() with 16-bit main weights.")
    parser.add_argument("--bnb_4bit_compute_dtype", choices=["float32", "float16", "bfloat16", "auto"], default=None, help="Compute dtype for 4bit quantization")
    parser.add_argument("--bnb_4bit_quant_dtype", choices=["fp4", "nf4"], default="fp4", help="Quant dtype for 4bit quantization")
    parser.add_argument("--bnb_4bit_use_double_quant", action="store_true", help="Whether to use double quantization in 4bit")

    # Parameters related to preprocessing
    parser.add_argument("--max_seq_length", type=int, default=512, help="Input Sequence Length")
    parser.add_argument("--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets")

    parser.add_argument("--learning_rate", default=2.5e-5, type=float, help="Learning rate")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="The per-device train batch size")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4, help="The per-device eval batch size")
    parser.add_argument("--preprocessing_num_workers", type=int, default=12, help="The number of processes to use for the preprocessing.",)

    # Parameters related to training: Reproducibility and resume training
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="If the training should continue from a checkpoint folder.")

    # Parameters related to training: PEFT
    parser.add_argument("--use_peft", action="store_true", help="Whether to use PEFT")
    parser.add_argument("--peft_lora_r", type=int, default=64, help="LoRA R")
    parser.add_argument("--peft_lora_alpha", type=float, default=16, help="LoRA alpha")
    parser.add_argument("--peft_lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--peft_target_modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,lm_head", help="LoRA target modules - comma separated with no spaces")

    # Parameters related to training: Training steps, gradient accumulation, optimization
    parser.add_argument("--num_train_epochs", default=3, type=int, help="Number of training epochs")
    parser.add_argument("--max_train_steps", default=None, help="Max train steps")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16, help="Gradient Accumulation Steps")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Whether to use gradient checkpointing")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="The weight decay")
    parser.add_argument("--adamw_fused", action="store_true", help="Whether to set fused=True in AdamW")
    parser.add_argument("--lr_scheduler_type", type=SchedulerType, default="cosine", help="The learning rate scheduler type")
    parser.add_argument("--lr_warmup_fraction", type=float, default=0.01, help="The number of warmup steps")

    # Parameters related to logging and saving
    parser.add_argument("--with_tracking", action="store_true", help="Whether to enable experiment trackers for logging")
    parser.add_argument("--report_to", type=str, default="all", help='The integration to report the results and logs to. Supported platforms are `"tensorboard"`, `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations. Only applicable when `--with_tracking` is passed.')
    parser.add_argument("--cache_dir", type=str, default=None, help="Cache directory")
    parser.add_argument("--output_dir", default="./results", help="Output directory")
    parser.add_argument("--checkpointing_steps", type=str, default=None, help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.")
    parser.add_argument("--merge_weights", action="store_true", help="Whether to merge weights at the end of training")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`.")
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument("--private_repo", action="store_true", help="Whether the created repo is private or not")
    args=parser.parse_args()

    # Some sanity checks
    if args.load_in_8bit and args.load_in_4bit:
        raise ValueError("Cannot load model in both 8bit and 4bit at the same time")

    if args.push_to_hub:
        if args.output_dir is None:
            raise ValueError("Cannot push to Hub if output_dir is not specified")

    return args

def main():
    args=parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs={}
    if args.with_tracking:
        accelerator_log_kwargs["log_with"]=args.report_to
        accelerator_log_kwargs["project_dir"]=args.output_dir

    accelerator=Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)

    # Make one log on every process with the configuration for debugging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now
    if args.seed is not None:
        set_seed(args.seed)
    
    # Handle the output directory creation
    if accelerator.is_local_main_process:
        if args.push_to_hub:
            # Retrieve or infer repo_name
            repo_name=args.hub_model_id
            if repo_name is None:
                repo_name=Path(args.output_dir).absolute().name
            # Create repo and retrieve repo_id
            is_private=args.private_repo
            repo_id=create_repo(repo_name, exist_ok=True, token=args.hub_token, private=is_private).repo_id
            # Clone repo locally 
            repo=Repository(args.output_dir, clone_from=repo_id, token=args.hub_token)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    if args.load_data_from_disk:
        raw_dataset=load_from_disk(args.dataset_name_or_path)
    else:
        raw_dataset=load_dataset(args.dataset_name_or_path, token=args.hf_access_token)
    
    bnb_config=BitsAndBytesConfig(
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
        llm_int8_threshold=args.llm_int8_threshold,
        llm_int8_skip_modules=args.llm_int8_skip_modules.split(",") if args.llm_int8_skip_modules is not None else None,
        llm_int8_enable_fp32_cpu_offload=args.llm_int8_enable_fp32_cpu_offload,
        llm_int8_has_fp16_weight=args.llm_int8_has_fp16_weight,
        bnb_4bit_compute_dtype=TORCH_DTYPES[args.bnb_4bit_compute_dtype] if args.bnb_4bit_compute_dtype is not None else None,
        bnb_4bit_quant_dtype=args.bnb_4bit_quant_dtype,
        bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
    )
    torch_dtype=TORCH_DTYPES[args.torch_dtype]

    # Load the tokenizer
    tokenizer=AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        token=args.hf_access_token,
    )
    tokenizer.pad_token_id=tokenizer.eos_token_id
    tokenizer.padding_side="right" # Fix weird overflow issue with fp16 training

    if args.load_in_8bit or args.load_in_4bit:
        base_model=AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch_dtype,
            cache_dir=args.cache_dir,
            token=args.hf_access_token,
            use_flash_attention_2=args.use_flash_attention_2,
        )
    else:
        base_model=AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch_dtype,
            cache_dir=args.cache_dir,
            token=args.hf_access_token,
            use_flash_attention_2=args.use_flash_attention_2,
        )
    base_model.config.use_cache=False
    base_model.config.sliding_window=args.sliding_window

    embedding_size=base_model.get_input_embeddings().weight.shape[0]
    if len(tokenizer)>embedding_size:
        base_model.resize_token_embeddings(len(tokenizer))

    if args.gradient_checkpointing:
        if hasattr(base_model, "enable_input_require_grads"):
            base_model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            base_model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        base_model.gradient_checkpointing_enable()
    
    # Preprocessing the datasets
    raw_dataset_column_names=raw_dataset["train"].column_names
    
    if args.max_seq_length is None:
        max_seq_length=tokenizer.model_max_length
        if max_seq_length>1024:
            logger.warning(
                "The chosen tokenizer supports a `model_max_length` that is longer than the default `max_seq_length` value"
                " of 1024. If you would like to use a longer `max_seq_length` up to `tokenizer.model_max_length` you can"
                " override this default with `--max_seq_length xxx`."
            )
        max_seq_length=1024
    else:
        if args.max_seq_length>tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({args.max_seq_length}) is larger than the maximum length for the model"
                f" ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        max_seq_length=min(args.max_seq_length, tokenizer.model_max_length)

    # We preprocess the data in THREE steps:
    # 1. Concatenate prompts and responses
    # 2. Tokenize the concatenated prompt-response pairs
    # 3. Set the labels corresponding to the prompt tokens to IGNORE_INDEX
    def preprocess_function(examples):
        prompts_responses=[p+" "+r for p, r in zip(examples["prompt"], examples["response"])]
        prompts_responses_tokenized=tokenizer(prompts_responses, truncation=True, max_length=max_seq_length)
        prompts_tokenized=tokenizer(examples["prompt"], truncation=True, max_length=max_seq_length)
        all_labels=copy.deepcopy(prompts_responses_tokenized["input_ids"])
        prompts_len=[len(prompt) for prompt in prompts_tokenized["input_ids"]]
        for labels, prompt_len in zip(all_labels, prompts_len):
            labels[:prompt_len]=[IGNORE_INDEX]*prompt_len
        result={k: v for k, v in prompts_responses_tokenized.items()}
        result["labels"]=all_labels
        return result

    with accelerator.main_process_first():
        preprocessed_dataset=raw_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=not args.overwrite_cache,
            remove_columns=raw_dataset_column_names,
            desc="Preprocessing the raw dataset",
        )

    train_dataset=preprocessed_dataset["train"]
    eval_dataset=preprocessed_dataset["validation"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Load LoRA configuration if --use_peft is passed
    if args.use_peft:
        peft_config=LoraConfig(
            r=args.peft_lora_r,
            lora_alpha=args.peft_lora_alpha,
            lora_dropout=args.peft_lora_dropout,
            target_modules=args.peft_target_modules.split(","),
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model=get_peft_model(base_model, peft_config)
    else:
        model=base_model
    
    # Log trainable parameters
    logger.info(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # DataLoaders creation
    data_collator=DataCollatorForInstructionTuning(tokenizer)
    train_dataloader=DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size, pin_memory=True, num_workers=8
    )
    eval_dataloader=DataLoader(
        eval_dataset, shuffle=False, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size, pin_memory=True, num_workers=8
    )

    # If using FSDP, prepare the model before the optimizer is instantiated
    model=accelerator.prepare(model)

    # # Optimizer
    # # Split weights in two groups, one with weight decay and the other not.
    # no_decay = ["bias", "layer_norm.weight"]
    # optimizer_grouped_parameters = [
    #     {
    #         "params": [p for n, p in base_model.named_parameters() if not any(nd in n for nd in no_decay)],
    #         "weight_decay": args.weight_decay,
    #     },
    #     {
    #         "params": [p for n, p in base_model.named_parameters() if any(nd in n for nd in no_decay)],
    #         "weight_decay": 0.0,
    #     },
    # ]
    # optimizer=torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, fused=args.adamw_fused)

    # FSDP currently doesn't support optimizer_grouped_parameters
    optimizer=torch.optim.AdamW(params=model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, fused=args.adamw_fused)

    # Scheduler and math around the number of training steps
    overrode_max_train_steps=False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
    
    lr_scheduler=get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=math.floor(args.lr_warmup_fraction*args.max_train_steps),
        num_training_steps=args.max_train_steps
    )

    # Prepare everything with our `accelerator`.
    optimizer, train_dataloader, eval_dataloader, lr_scheduler=accelerator.prepare(
        optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("instruction_tuner", experiment_config)
    
    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            checkpoint_path = args.resume_from_checkpoint
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
            checkpoint_path = path
            path = os.path.basename(checkpoint_path)

        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.load_state(path)
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader)

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)

    for epoch in range(starting_epoch, args.num_train_epochs):
        with TorchTracemalloc() as tracemalloc:
            model.train()
            if args.with_tracking:
                total_loss=0
            if args.resume_from_checkpoint and epoch==starting_epoch and resume_step is not None:
                # We skip the first `n` batches in the dataloader when resuming from a checkpoint
                active_dataloader=accelerator.skip_first_batches(train_dataloader, resume_step)
            else:
                active_dataloader=train_dataloader
            for step, batch in enumerate(active_dataloader):
                with accelerator.accumulate(model):
                    outputs=model(**batch)
                    loss=outputs.loss
                    # We keep track of loss at each epoch
                    if args.with_tracking:
                        total_loss+=loss.detach().float()
                    accelerator.backward(loss)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    completed_steps += 1
                
                if args.with_tracking:
                    accelerator.log({"instant_loss": loss.item(), "lr": optimizer.param_groups[0]["lr"], "step":completed_steps}, step=completed_steps)
                
                if isinstance(checkpointing_steps, int):
                    if completed_steps%checkpointing_steps==0:
                        output_dir=f"step_{completed_steps}"
                        if args.output_dir is not None:
                            output_dir=os.path.join(args.output_dir, output_dir)
                        accelerator.save_state(output_dir)
                
                if completed_steps>=args.max_train_steps:
                    break
        
        # Printing the GPU memory usage details such as allocated memory, peak memory, and total memory usage
        accelerator.print("GPU Memory before entering the train : {}".format(b2mb(tracemalloc.begin)))
        accelerator.print("GPU Memory consumed at the end of the train (end-begin): {}".format(tracemalloc.used))
        accelerator.print("GPU Peak Memory consumed during the train (max-begin): {}".format(tracemalloc.peaked))
        accelerator.print(
            "GPU Total Peak Memory consumed during the train (max): {}".format(
                tracemalloc.peaked + b2mb(tracemalloc.begin)
            )
        )

        accelerator.print("CPU Memory before entering the train : {}".format(b2mb(tracemalloc.cpu_begin)))
        accelerator.print("CPU Memory consumed at the end of the train (end-begin): {}".format(tracemalloc.cpu_used))
        accelerator.print("CPU Peak Memory consumed during the train (max-begin): {}".format(tracemalloc.cpu_peaked))
        accelerator.print(
            "CPU Total Peak Memory consumed during the train (max): {}".format(
                tracemalloc.cpu_peaked + b2mb(tracemalloc.cpu_begin)
            )
        )
                
        model.eval()
        losses = []
        with TorchTracemalloc() as tracemalloc:
            for step, batch in enumerate(eval_dataloader):
                with torch.no_grad():
                    outputs = model(**batch)

                loss = outputs.loss
                losses.append(accelerator.gather_for_metrics(loss.repeat(args.per_device_eval_batch_size)))

        # Printing the GPU memory usage details such as allocated memory, peak memory, and total memory usage
        accelerator.print("GPU Memory before entering the eval : {}".format(b2mb(tracemalloc.begin)))
        accelerator.print("GPU Memory consumed at the end of the eval (end-begin): {}".format(tracemalloc.used))
        accelerator.print("GPU Peak Memory consumed during the eval (max-begin): {}".format(tracemalloc.peaked))
        accelerator.print(
            "GPU Total Peak Memory consumed during the eval (max): {}".format(
                tracemalloc.peaked + b2mb(tracemalloc.begin)
            )
        )

        accelerator.print("CPU Memory before entering the eval : {}".format(b2mb(tracemalloc.cpu_begin)))
        accelerator.print("CPU Memory consumed at the end of the eval (end-begin): {}".format(tracemalloc.cpu_used))
        accelerator.print("CPU Peak Memory consumed during the eval (max-begin): {}".format(tracemalloc.cpu_peaked))
        accelerator.print(
            "CPU Total Peak Memory consumed during the eval (max): {}".format(
                tracemalloc.cpu_peaked + b2mb(tracemalloc.cpu_begin)
            )
        )

        losses = torch.cat(losses)
        try:
            eval_loss = torch.mean(losses)
            perplexity = math.exp(eval_loss)
        except OverflowError:
            perplexity = float("inf")

        logger.info(f"epoch {epoch}: perplexity: {perplexity} eval_loss: {eval_loss}")

        if args.with_tracking:
            accelerator.log(
                {
                    "perplexity": perplexity,
                    "eval_loss": eval_loss,
                    "train_loss": total_loss.item() / len(train_dataloader),
                    "epoch": epoch,
                    "step": completed_steps,
                },
                step=completed_steps,
            )
        
        if args.push_to_hub and epoch<args.num_train_epochs-1:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
            )
            if accelerator.is_main_process:
                tokenizer.save_pretrained(args.output_dir)
                repo.push_to_hub(
                    commit_message=f"Training in progress epoch {epoch}", blocking=False, auto_lfs_prune=True
                )
        
        if args.checkpointing_steps=="epoch":
            output_dir=f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir=os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)
    
    if args.with_tracking:
        accelerator.end_training()
    
    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)
            if args.push_to_hub:
                repo.push_to_hub(commit_message="End of Training", auto_lfs_prune=True)
            with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
                json.dump({"perplexity": perplexity}, f)
    
    if args.use_peft and args.merge_weights:
        # Free memory for merging weights
        del base_model
        torch.cuda.empty_cache()

        model=AutoPeftModelForCausalLM.from_pretrained(args.output_dir, device_map="auto", torch_dtype=torch_dtype)
        model=model.merge_and_unload()

        output_merged_dir=os.path.join(args.output_dir, "final_merged_checkpoint")
        model.save_pretrained(output_merged_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save)
        if accelerator.is_main_process:
            tokenizer.save_pretrained(output_merged_dir)

if __name__=="__main__":
    main()