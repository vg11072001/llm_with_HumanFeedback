# ruff: noqa: E402
"""
* Environment Variables: Sets environment variables to control threading and tokenizer parallelism.
    - OMP_NUM_THREADS: Limits the number of threads used by OpenMP.
    - TOKENIZERS_PARALLELISM: Disables parallelism in tokenizers to avoid potential issues.
    - HF_HOME: (Commented out) Specifies the path for Hugging Face cache.

* Imports: Imports necessary libraries and modules for argument parsing, time management, 
    file operations, PyTorch functionalities, distributed training, and custom utilities.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["HF_HOME"] = "/path/to/fast/storage"

import argparse
import time
import shutil
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    FullStateDictConfig,
    StateDictType,
)

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import apply_activation_checkpointing

from pathlib import Path

from detectron2.config import LazyConfig, instantiate
from detectron2.solver import LRMultiplier
from detectron2.engine.hooks import LRScheduler
from detectron2.utils.env import seed_all_rng

from human_pref.utilis.logging import get_logger
from human_pref.utilis.utils import to_gpu

def parse_args():
    '''
    parse_args: Defines and parses command-line arguments for the script.
        - config: Path to the configuration file.
        --load-from: Path to a checkpoint to load the model from.
        --init-only: Flag to initialize the model and save it without training.
        --eval-only: Flag to run evaluation only.
        --no-log-file: Flag to disable logging to a file.
        --seed: Random seed for reproducibility.
        --output-root: Root directory for output files.
        --opts: Additional configuration options.
        --out: Path to save evaluation results.
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--load-from", default=None, type=str)
    parser.add_argument("--init-only", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--no-log-file", action="store_true")
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--output-root", default="../artifacts")
    parser.add_argument("--opts", help=""" Modify config options at the end of the command, use "path.key=value". """.strip(), default=[], nargs=argparse.ZERO_OR_MORE)
    parser.add_argument("--out", default=None, type=str)
    
    return parser.parse_args()


class LogLossBuffer:
    """
    Circular buffer for storing log loss values
    
    LogLossBuffer: A circular buffer to store and compute the mean of log loss values.
        - __init__: Initializes the buffer with a specified size and device.
        - append: Adds a new value to the buffer.
        - mean: Computes the mean of the stored values.
    
    """

    def __init__(self, size, device="cuda"):
        self.buffer = torch.zeros(size, device=device)
        self.size = size
        self.idx = 0
        self.num = 0

    def append(self, value):
        self.buffer[self.idx] = value
        self.idx = (self.idx + 1) % self.size
        self.num = min(self.num + 1, self.size)

    def mean(self):
        return self.buffer.sum().item() / self.num


@torch.no_grad()
def do_test(cfg, model):
    """ 
    do_test: Evaluates the model on the validation dataset and logs the results.
        
        @torch.no_grad(): Disables gradient computation for evaluation.
        - logger: Initializes a logger.
        - val_loader: Instantiates the validation data loader.
        - model.eval(): Sets the model to evaluation mode.
        - tqdm: Progress bar for tracking evaluation progress.
        - rank and world_size: Get the rank and world size for distributed training.
        - probs: List to store prediction probabilities.
        - for batch in prog_bar: Iterates over batches in the validation loader.
        - for micro_batch in batch: Iterates over micro-batches in each batch.
        - to_gpu: Moves the micro-batch to the GPU.
        - prob: Computes the softmax probabilities for the micro-batch.
        - dist.all_gather: Gathers probabilities from all processes.
        - result: Concatenates and converts probabilities to a numpy array.
        - eval_result: Evaluates the results using the dataset's evaluation method.
    """
    
    logger = get_logger("lmsys")
    logger.info("Evaluation start")

    val_loader = instantiate(cfg.dataloader.val)

    model.eval()
    from tqdm import tqdm

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if rank == 0:
        # Library: tqdm is a Python library that provides a fast, extensible progress bar 
        # for loops and other iterable objects. It is commonly used to display progress in a 
        # visually appealing way.
        # val_loader: This is the validation data loader, which is an instance of torch.utils.data.DataLoader. It provides batches of validation data to the model during evaluation.
        # tqdm(val_loader): Wrapping val_loader with tqdm creates a progress bar that tracks the progress of iterating over the batches in the validation data loader.
        prog_bar = tqdm(val_loader)
    else:
        prog_bar = val_loader

    probs = []
    for batch in prog_bar:
        for micro_batch in batch:
            micro_batch = to_gpu(micro_batch)
            prob = model(micro_batch["input_ids"], micro_batch["cu_seqlens"]).softmax(
                dim=-1
            )
            gather_probs = [torch.zeros_like(prob) for _ in range(world_size)]
            dist.all_gather(gather_probs, prob)
            prob = torch.stack(gather_probs, dim=1).flatten(0, 1)
            probs.append(prob.data.cpu())

    result = torch.cat(probs, dim=0).numpy()
    # the last batch maybe padded to be divisible by world_size
    result = result[: len(val_loader.dataset)]

    logger.info("Evaluation prediction done")
    if not hasattr(val_loader.dataset, "evaluate"):
        eval_result = {"info": f"Not implemented for {type(val_loader.dataset)}"}
    else:
        eval_result = val_loader.dataset.evaluate(result)
    logger.info("Evaluation end")
    return result, eval_result


def save_checkpoint(model, optimizer, work_dir, checkpoint_path):
    """ 
    save_checkpoint: Saves the model's state dictionary to a checkpoint file.
    
        - save_policy: Configuration for saving the state dictionary.
        - FSDP.state_dict_type: Sets the state dictionary type for FSDP.
        - cpu_state: Gets the model's state dictionary.
        - torch.save: Saves the checkpoint to the specified path.
    """
    
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        cpu_state = model.state_dict()
    if dist.get_rank() == 0:
        checkpoint = {
            "model": cpu_state,
            # "optimizer": optimizer.state_dict(),
        }
        torch.save(checkpoint, checkpoint_path)


def do_train(cfg, model):
    """ 
    do_train: Manages the training loop, including logging, checkpointing, and evaluation.
    
        - cfg.optimizer.params: Filters parameters that require gradients.
        - optimizer: Instantiates the optimizer.
        - train_loader: Instantiates the training data loader.
        - max_epochs: Maximum number of training epochs.
        - lr_scheduler: Learning rate scheduler.
        - best_param_group_id: Gets the best parameter group ID for the optimizer.
        - logger: Initializes a logger.
        - loss_history: Initializes a log loss buffer.
        - total_updates: Counter for total updates.
        - rank and fsdp_loss: Get the rank and initialize FSDP loss tensor.
        - clip_grad: Flag to enable gradient clipping.
        - for curr_epoch in range(max_epochs): Iterates over epochs.
        - model.train(): Sets the model to training mode.
        - for curr_iter, batch in enumerate(train_loader): Iterates over batches in the training loader.
        - total_batch_size: Computes the total batch size.
        - fsdp_loss.zero_(): Resets the FSDP loss tensor.
        - for micro_batch in batch: Iterates over micro-batches in each batch.
        - to_gpu: Moves the micro-batch to the GPU.
        - logits: Computes the logits for the micro-batch.
        - loss: Computes the cross-entropy loss.
        - fsdp_loss: Accumulates the loss and batch size.
        - loss.backward(): Backpropagates the loss.
        - dist.all_reduce: Reduces the FSDP loss tensor across all processes.
        - grad_norm: Clips the gradients if enabled.
        - optimizer.step(): Updates the model parameters.
        - optimizer.zero_grad(set_to_none=True): Resets the gradients.
        - loss_history.append: Appends the loss to the log loss buffer.
        - total_updates: Increments the total updates counter.
        - lr_scheduler.step(): Updates the learning rate.
        - logger.info: Logs the training progress.
        - if total_updates % cfg.train.checkpoint_interval == 0: Saves a checkpoint at specified intervals.
        - dist.barrier(): Synchronizes all processes.
        - if (curr_epoch + 1) % cfg.train.get("eval_interval", 1) == 0: Evaluates the model at specified intervals.
    
    """
    
    cfg.optimizer.params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = instantiate(cfg.optimizer)

    train_loader = instantiate(cfg.dataloader.train)
    max_epochs = cfg.train.max_epochs
    lr_scheduler = LRMultiplier(
        optimizer,
        multiplier=instantiate(cfg.lr_multiplier),
        max_iter=max_epochs * len(train_loader),
    )
    best_param_group_id = LRScheduler.get_best_param_group_id(optimizer)

    logger = get_logger("lmsys")
    loss_history = LogLossBuffer(cfg.train.get("log_buffer_size", 100))
    total_updates = 0

    rank = dist.get_rank()
    fsdp_loss = torch.zeros(2).to(rank)

    clip_grad = cfg.train.get("clip_grad", True)
    for curr_epoch in range(max_epochs):
        model.train()
        for curr_iter, batch in enumerate(train_loader):
            total_batch_size = sum(micro_batch["batch_size"] for micro_batch in batch)
            fsdp_loss.zero_()
            for micro_batch in batch:
                micro_batch = to_gpu(micro_batch)
                logits = model(micro_batch["input_ids"], micro_batch["cu_seqlens"])
                loss = F.cross_entropy(logits, micro_batch["label"])
                fsdp_loss[0] += loss.detach() * micro_batch["batch_size"]
                fsdp_loss[1] += micro_batch["batch_size"]
                loss = loss * (micro_batch["batch_size"] / total_batch_size)
                loss.backward()
                # - The line dist.all_reduce(fsdp_loss, op=dist.ReduceOp.SUM) is used to sum the fsdp_loss 
                # across all processes in a distributed training setup. 
                # - It ensures that each process has the same aggregated loss value after 
                # the operation, which is typically done for synchronization in distributed training.
                dist.all_reduce(fsdp_loss, op=dist.ReduceOp.SUM)

            if clip_grad:
                grad_norm = model.clip_grad_norm_(1.0)
                grad_norm = grad_norm.item()
            else:
                grad_norm = 0
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            loss_history.append(fsdp_loss[0] / fsdp_loss[1])
            total_updates += 1
            lr_scheduler.step()
            if total_updates % cfg.train.log_interval == 0:
                lr = optimizer.param_groups[best_param_group_id]["lr"]
                loss_val = loss_history.mean()
                max_mem_mb = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
                logger.info(
                    f"Epoch [{curr_epoch+1}/{max_epochs}] Iter [{curr_iter+1}/{len(train_loader)}]"
                    f" lr: {lr:.4e}, loss: {loss_val:.4f}, grad_norm: {grad_norm:.4f}, max_mem: {max_mem_mb:.0f}M"
                )

            # save every N updates
            if total_updates % cfg.train.checkpoint_interval == 0:
                checkpoint_path = (
                    Path(cfg.train.work_dir) / f"update_{total_updates}.pth"
                )
                logger.info(f"Save checkpoint: {checkpoint_path}")
                save_checkpoint(model, optimizer, cfg.train.work_dir, checkpoint_path)
                logger.info("Save checkpoint done.")
                dist.barrier()

        # end of epoch checkpoint
        checkpoint_path = Path(cfg.train.work_dir) / "update_last.pth"
        logger.info(f"Save checkpoint: {checkpoint_path}")
        save_checkpoint(model, optimizer, cfg.train.work_dir, checkpoint_path)
        logger.info("Save checkpoint done.")

        # - dist.barrier() is a synchronization tool used in distributed training to ensure that 
        # all processes reach a certain point in the program before any process continues. 
        # - It is useful to prevent race conditions, ensure coordinated actions (e.g., model saving, initialization), 
        # and control the flow of execution in distributed systems.
        dist.barrier()

        # evaluate
        if (curr_epoch + 1) % cfg.train.get("eval_interval", 1) == 0:
            result, eval_result = do_test(cfg, model)
            if rank == 0:
                logger.info(f"Epoch {curr_epoch+1} evaluation result: {eval_result}")
                torch.save(
                    result,
                    Path(cfg.train.work_dir) / f"result_epoch_{curr_epoch+1}.pth",
                )


def setup(args):
    """ 
    setup: Initializes the distributed process group, sets up the configuration, logging, and random seed.
        
        - dist.init_process_group("nccl"): Initializes the NCCL process group for distributed training.
        - torch.cuda.set_device(dist.get_rank()): Sets the current CUDA device based on the process rank.
        - cfg: Loads the configuration file.
        - cfg_path: Gets the path to the configuration file.
        - work_dir_root: Sets the root directory for output files.
        - work_dir: Constructs the working directory path.
        - cfg.train.work_dir: Sets the working directory in the configuration.
        - cfg = LazyConfig.apply_overrides(cfg, args.opts): Applies additional configuration options.
        - Path(cfg.train.work_dir).mkdir(parents=True, exist_ok=True): Creates the working directory if it doesn't exist.
        - timestamp: Gets the current timestamp.
        - shutil.copy(args.config, Path(work_dir) / f"{timestamp}.py"): Copies the configuration file to the working directory.
        - log_file: Sets the log file path.
        - logger: Initializes a logger.
        - seed_all_rng(seed): Sets the random seed for reproducibility.
        - logger.info("Start"): Logs the start of the setup process.
    """
    
    dist.init_process_group("nccl")
    torch.cuda.set_device(dist.get_rank())

    cfg = LazyConfig.load(args.config)
    # default work_dir
    cfg_path = Path(args.config)
    work_dir_root = Path(args.output_root)
    # example: work_dir = artifacts/stage1/m0
    work_dir = str(work_dir_root / cfg_path.relative_to("configs/").with_suffix(""))
    cfg.train.work_dir = work_dir
    # override config
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    Path(cfg.train.work_dir).mkdir(parents=True, exist_ok=True)

    # dump config
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    if not args.eval_only and dist.get_rank() == 0:
        # LazyConfig.save(cfg, str(Path(work_dir) / f"{timestamp}.yaml"))
        shutil.copy(args.config, Path(work_dir) / f"{timestamp}.py")

    # logger
    if args.eval_only or args.no_log_file:
        log_file = None
    else:
        log_file = Path(work_dir) / f"{timestamp}.log"
    logger = get_logger("lmsys", log_file=log_file)
    logger.info("Start")

    # seed
    if args.seed >= 0:
        seed = args.seed
    else:
        seed = cfg.train.get("seed", 0)
    seed_all_rng(seed)
    logger.info(f"Set random seed: {seed}")

    return cfg

# def destroy_process_group(group: Optional[ProcessGroup] = None):
#     """
#     Destroy a given process group, and deinitialize the distributed package.

#     Args:
#         group (ProcessGroup, optional): The process group to be destroyed, if
#                                         group.WORLD is given, all process
#                                         groups including the default one will
#                                         be destroyed.
#     """
#     global _world

#     if group == GroupMember.NON_GROUP_MEMBER:
#         return

#     if group is None:
#         pg = GroupMember.WORLD
#     else:
#         pg = group

#     assert pg is not None
#     if _world.pg_map.get(pg, None) is None:
#         raise ValueError("Invalid process group specified")


def clean_up():
    """ 
    clean_up: Destroys the distributed process group.
        - dist.destroy_process_group(): Cleans up the distributed training environment.
    """
    dist.destroy_process_group()

def main():
    """ 
    The main function that orchestrates the entire workflow, including parsing arguments, setting up configurations, 
    initializing the model, handling different modes (initialization, evaluation, training), and cleaning up.
    """
    '''
    Purpose: If the --init-only flag is set, the script will initialize the model and save its state to a file, then exit.
    Steps:
    - Path Setup: Constructs the path where the initialized model will be saved.
    - Save Model State: Saves the model's state dictionary to the specified path.
    - Logging: Logs the path where the initialized model is saved.
    '''
    args = parse_args()
    cfg = setup(args)
    model = instantiate(cfg.model)
    logger = get_logger("lmsys")
    if args.init_only:
        # example: work_dir = artifacts/stage1/m0/initialized.pth
        init_path = Path(cfg.train.work_dir) / "initialized.pth"
        torch.save(model.state_dict(), init_path)
        logger.info(f"Saved initialized model: {init_path}")

    '''
    Purpose: If the configuration specifies cast_to_bf16, the model's parameters are cast to BF16 (bfloat16) precision.
    Steps:
    - Logging: Logs that the model is being cast to BF16.
    - Casting: Iterates over the model's parameters and casts them to BF16.
    '''
    if cfg.train.get("cast_to_bf16", False):
        logger.info("Casting model to BF16")
        # for name, m in model.named_modules():
        #     m.to(torch.bfloat16)
        for p in model.parameters():
            p.data = p.data.to(torch.bfloat16)
    
    '''
    Purpose: Loads a model checkpoint if specified in the configuration or command-line arguments.
    Steps:
    - Determine Checkpoint Path: Checks if a checkpoint path is provided via command-line arguments or configuration.
    - Load Checkpoint: Loads the checkpoint from the specified path.
    - Compatibility Check: Ensures the checkpoint contains the model state dictionary.
    - Load Model State: Loads the model state dictionary from the checkpoint.
    - Logging: Logs the checkpoint path and the result of loading the checkpoint.
    '''
    load_from = cfg.train.get("load_from", None)
    if args.load_from is not None:
        load_from = args.load_from

    if load_from is not None:
        checkpoint = torch.load(load_from, map_location="cpu")
        if "model" not in checkpoint:
            checkpoint = {"model": checkpoint}
        load_result = model.load_state_dict(checkpoint["model"], strict=False)
        logger.info(f"Load checkpoint: {load_from}")
        logger.info(f"Load checkpoint: {load_result}")

    # Purpose: Logs the sharding strategy being used for the model.
    logger.info(f"Use sharding strategy: {cfg.fsdp.sharding_strategy}")

    '''
    Purpose: Wraps the model with Fully Sharded Data Parallel (FSDP) for efficient distributed training.
    Steps:
    - FSDP Wrapping: Wraps the model with FSDP using the specified auto-wrap policy, sharding strategy, device ID, and mixed precision settings.
    - Activation Checkpointing: Applies activation checkpointing to the model to save memory during training.
    '''
    model = FSDP(
        model,
        auto_wrap_policy=cfg.fsdp.auto_wrap_policy,
        sharding_strategy=cfg.fsdp.sharding_strategy,
        device_id=torch.cuda.current_device(),
        mixed_precision=cfg.fsdp.mixed_precision,
    )
    apply_activation_checkpointing(model, auto_wrap_policy=cfg.fsdp.auto_wrap_policy)

    '''
    Purpose: If the --eval-only flag is set, the script will evaluate the model and save the results.
    Steps:
    - Evaluation: Calls the do_test function to evaluate the model.
    - Logging: Logs the evaluation results.
    - Save Results: Saves the evaluation results to the specified output path if provided.
    '''
    if args.eval_only:
        result, eval_result = do_test(cfg, model)
        logger.info(f"Evaluation result: {eval_result}")
        if args.out is not None:
            torch.save(result, args.out)
    else:
        # Purpose: If the --eval-only flag is not set, the script will train the model.
        # Training: Calls the do_train function to train the model.
        do_train(cfg, model)

    # Purpose: Cleans up the distributed process group.
    # Steps:
    # Clean Up: Calls the clean_up function to destroy the distributed process group.
    clean_up()


if __name__ == "__main__":
    main()
