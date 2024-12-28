import functools
import os

import torch
from detectron2.config import LazyCall as L

# from detectron2.config import instantiate
import kagglehub
from kaggle.api.kaggle_api_extended import KaggleApi

from fvcore.common.param_scheduler import CompositeParamScheduler, LinearParamScheduler
from transformers import AutoTokenizer
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from human_pref.data_processor.dataset import LMSYSDataset
from human_pref.data_processor.processors import ProcessorPAB, ProcessorPAPB
from human_pref.models.modeling_gemma2_fast import (
    Gemma2DecoderLayer,
    Gemma2ForSequenceClassification,
)
from human_pref.utilis.anyprecision_optimizer import AnyPrecisionAdamW
from human_pref.utilis.logging import get_logger
from user_secrets import UserSecretsClient

secrets_client = UserSecretsClient()

os.environ['KAGGLE_USERNAME'] = secrets_client.get_secret("KAGGLE_USERNAME")
os.environ['KAGGLE_KEY'] = secrets_client.get_secret("KAGGLE_KEY")

api = KaggleApi()
api.authenticate()

model_name_or_path = "google/gemma-2-9b-it" # model path when have acces to dowload weights from HF
path = kagglehub.dataset_download("cdeotte/gemma2-9b-it-fp16")
weights_dir = path 

logger = get_logger("lmsys")

# model config
def build_model():
    logger.debug(os.path.abspath(weights_dir))  # Print absolute path to check the directory structure
    model = Gemma2ForSequenceClassification.from_pretrained(
        weights_dir,
        torch_dtype=torch.bfloat16,
        num_labels=3,
        _attn_implementation="flash_attention_2",
    )
    # re-initilize the head here
    # Gemma2ForSequenceClassification init is weird
    hdim = model.config.hidden_size
    model.score = torch.nn.Sequential(
        torch.nn.Dropout(0.1),
        torch.nn.Linear(hdim, hdim // 2),
        torch.nn.Dropout(0.1),
        torch.nn.GELU(),
        torch.nn.Linear(hdim // 2, 3),
    ).bfloat16()

    return model

# all config reated stuff
model = L(build_model)()
optimizer = L(AnyPrecisionAdamW)(
    lr=0.25e-5,
    use_kahan_summation=True,
    betas=(0.9, 0.99),
    eps=1e-6,
    weight_decay=0.01,
)


# data config
def build_dataset(fold, training, max_length):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.deprecation_warnings[
        "sequence-length-is-longer-than-the-specified-maximum"
    ] = True
    if training:
        processor = [
            ProcessorPAB(
                tokenizer=tokenizer,
                max_length=max_length,
                support_system_role=False,
            ),
            ProcessorPAPB(
                tokenizer=tokenizer,
                max_length=max_length // 2,
                support_system_role=False,
            ),
        ]
        dataset0 = LMSYSDataset(
            csv_file="../output/dtrainval.csv",
            query=f"fold != {fold}",
            processor=processor,
            include_swap=True,
        )
        dataset1 = LMSYSDataset(
            csv_file="../output/lmsys-33k-deduplicated.csv",
            query=None,
            processor=processor,
            include_swap=True,
        )
        dataset = torch.utils.data.ConcatDataset([dataset0, dataset1])
    else:
        processor = ProcessorPAB(
            tokenizer=tokenizer,
            max_length=max_length,
            support_system_role=False,
        )
        dataset = LMSYSDataset(
            csv_file="../output/dtrainval.csv",
            query=f"fold == {fold}",
            processor=processor,
        )
    return dataset


def build_data_loader(dataset, batch_size, num_workers, training=True):
    from human_pref.data_processor.collators import VarlenCollator, ShardedMaxTokensCollator

    max_tokens = 1024 * 16
    return torch.utils.data.DataLoader(
        dataset,
        shuffle=training,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=training,
        collate_fn=ShardedMaxTokensCollator(
            max_tokens=max_tokens,
            base_collator=VarlenCollator(),
            sort_samples=training,
        ),
    )


VAL_FOLD = 0
dataloader = dict(
    train=L(build_data_loader)(
        dataset=L(build_dataset)(fold=VAL_FOLD, training=True, max_length=4096),
        batch_size=80,
        num_workers=4,
        training=True,
    ),
    val=L(build_data_loader)(
        dataset=L(build_dataset)(fold=VAL_FOLD, training=False, max_length=4096),
        batch_size=80,
        num_workers=4,
        training=False,
    ),
)

max_epochs = 1
lr_multiplier = L(CompositeParamScheduler)(
    schedulers=[
        L(LinearParamScheduler)(start_value=0.001, end_value=1),
        L(LinearParamScheduler)(start_value=1, end_value=0.001),
    ],
    lengths=[0.1, 0.9],
    interval_scaling=["rescaled", "rescaled"],
)

train = dict(
    device="cuda",
    max_epochs=max_epochs,
    log_interval=10,
    checkpoint_interval=200,
    eval_interval=1,
    cast_to_bf16=False,
    log_buffer_size=20,
    clip_grad=False,
    seed=3,
)

fsdp = dict(
    auto_wrap_policy=functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={Gemma2DecoderLayer},
    ),
    sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
    mixed_precision=MixedPrecision(
        param_dtype=torch.bfloat16,
        # Gradient communication precision.
        reduce_dtype=torch.bfloat16,
        # Buffer precision. For rotary_emb in this case.
        buffer_dtype=torch.float32,
    ),
)