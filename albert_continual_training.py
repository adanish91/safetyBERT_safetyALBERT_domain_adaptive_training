#!/usr/bin/env python3
"""
ALBERT Continual Training Script

A production-ready script for continual training of ALBERT models with masked language modeling (MLM).
Optimized for memory efficiency and mixed precision training on high-end GPUs.

Features:
- Mixed precision training with automatic loss scaling
- Memory-efficient batching with dynamic sizing
- Smart text chunking for long sequences
- Comprehensive monitoring and visualization
- Resumable training with metric restoration
- Multi-dataset support with source balancing

Author: Abid Ali Khan Danish
"""

import os
import warnings
import logging
import argparse
import json
import traceback
from datetime import datetime
from pathlib import Path
from collections import defaultdict

# Environment setup for cleaner output
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PYTHONWARNINGS'] = 'ignore'
warnings.filterwarnings('ignore')

import torch
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from transformers import (
    AlbertTokenizer, 
    AlbertForMaskedLM, 
    AlbertConfig,
    get_linear_schedule_with_warmup
)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class SubsetDatasetWithIndices(Dataset):
    """
    Dataset wrapper that provides access to a subset of data using indices.
    
    Args:
        dataset: The base dataset to subset
        indices: List of indices to include in the subset
    """
    
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
        
    def __getitem__(self, idx):
        if isinstance(idx, list):
            return [self.dataset[self.indices[i]] for i in idx]
        elif isinstance(idx, int):
            return self.dataset[self.indices[idx]]
        else:
            raise TypeError(f"Expected integer index but got {type(idx)} for idx: {idx}")
        
    def __len__(self):
        return len(self.indices)


class SmartMLMDataset(Dataset):
    """
    Intelligent dataset for ALBERT MLM training with automatic text chunking and length optimization.
    
    Features:
    - ALBERT-specific tokenization with proper special tokens
    - Automatic handling of texts longer than max_length via sliding windows
    - Efficient tokenization and caching
    - Source tracking for multi-dataset training
    - Length-based bucketing for efficient batching
    
    Args:
        file_paths: List of CSV file paths
        text_columns: Dict mapping filename to column name(s) containing text
        tokenizer: ALBERT tokenizer instance
        max_length: Maximum sequence length (default: 512)
        stride: Sliding window stride for long texts (default: 256)
    """
    
    def __init__(self, file_paths, text_columns, tokenizer, max_length=512, stride=256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        
        self.texts = []
        self.text_lengths = []
        self.source_ids = []
        
        print("Loading and analyzing texts...")
        self._load_data(file_paths, text_columns)
        self.indices = self._create_indices()
        
    def _load_data(self, file_paths, text_columns):
        """Load texts from CSV files and calculate statistics."""
        for source_id, file_path in enumerate(file_paths):
            file_name = os.path.basename(file_path)
            df = pd.read_csv(file_path)
            column = text_columns[file_name]
            
            # Handle both single columns and lists of columns
            if isinstance(column, list):
                for col in column:
                    texts = df[col].dropna().astype(str).tolist()
                    self.texts.extend(texts)
                    self.source_ids.extend([source_id] * len(texts))
            else:
                texts = df[column].dropna().astype(str).tolist()
                self.texts.extend(texts)
                self.source_ids.extend([source_id] * len(texts))
            
            # Calculate token lengths for optimization
            lengths = [len(self.tokenizer.encode(text)) for text in texts]
            self.text_lengths.extend(lengths)
            
            print(f"Loaded {file_name}: {len(texts)} texts")
            print(f"Length statistics - Mean: {np.mean(lengths):.1f}, "
                  f"Max: {np.max(lengths)}, >512 tokens: {sum(l > 512 for l in lengths)}")
        
    def _create_indices(self):
        """Create access indices with chunking for long sequences."""
        indices = []
        
        for idx, (text, length, source_id) in enumerate(zip(self.texts, self.text_lengths, self.source_ids)):
            if length <= self.max_length - 2:  # Reserve space for [CLS] and [SEP]
                # Short texts: single entry
                indices.append({
                    'text_idx': idx,
                    'start': 0,
                    'end': length,
                    'length': length + 2,  # Add special tokens
                    'source_id': source_id,
                    'is_chunk': False
                })
            else:
                # Long texts: create overlapping chunks
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                chunk_starts = list(range(0, len(tokens), self.stride))
                
                for chunk_idx, start in enumerate(chunk_starts):
                    chunk_end = min(start + self.max_length - 2, len(tokens))  # Reserve space for special tokens
                    chunk_length = chunk_end - start
                    
                    # Skip tiny final chunks unless it's the only chunk
                    if chunk_length < 64 and chunk_idx > 0:
                        continue
                        
                    indices.append({
                        'text_idx': idx,
                        'start': start,
                        'end': chunk_end,
                        'length': chunk_length + 2,  # Add special tokens
                        'source_id': source_id,
                        'is_chunk': True,
                        'chunk_idx': chunk_idx,
                        'total_chunks': len(chunk_starts)
                    })
                    
                    if chunk_end == len(tokens):
                        break
        
        return indices
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        """Get tokenized text sample with ALBERT-specific handling."""
        index_info = self.indices[idx]
        text = self.texts[index_info['text_idx']]
        
        if index_info['is_chunk']:
            # Handle chunked text with ALBERT tokens
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            chunk_tokens = tokens[index_info['start']:index_info['end']]
            
            # Add ALBERT special tokens: [CLS]=2, [SEP]=3
            input_ids = [2] + chunk_tokens + [3]
            attention_mask = [1] * len(input_ids)
            
            # Pad to max_length
            padding_length = self.max_length - len(input_ids)
            if padding_length > 0:
                input_ids = input_ids + [0] * padding_length
                attention_mask = attention_mask + [0] * padding_length
            
            return {
                'input_ids': torch.tensor(input_ids[:self.max_length]),
                'attention_mask': torch.tensor(attention_mask[:self.max_length]),
                'source_id': index_info['source_id'],
                'text_idx': index_info['text_idx'],
                'is_chunk': True,
                'chunk_idx': index_info['chunk_idx'],
                'total_chunks': index_info['total_chunks']
            }
        else:
            # Standard tokenization for short texts
            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                truncation=True,
                padding='max_length',
                return_tensors=None
            )
            
            return {
                'input_ids': torch.tensor(encoding['input_ids']),
                'attention_mask': torch.tensor(encoding['attention_mask']),
                'source_id': index_info['source_id'],
                'text_idx': index_info['text_idx'],
                'is_chunk': False
            }


class MemoryAwareSmartBatchSampler:
    """
    Intelligent batch sampler optimized for ALBERT's memory efficiency.
    
    Features:
    - Length-based bucketing to minimize padding
    - Source balancing across datasets
    - Dynamic batch sizing based on sequence length
    - GPU memory-aware token limits optimized for ALBERT
    
    Args:
        dataset: Dataset to sample from
        base_batch_size: Base batch size for medium-length sequences
        max_tokens_per_batch: Maximum tokens per batch to prevent OOM
        drop_last: Whether to drop incomplete final batches
    """
    
    def __init__(self, dataset, base_batch_size, max_tokens_per_batch=8192, drop_last=False):
        self.dataset = dataset
        self.base_batch_size = base_batch_size
        self.max_tokens_per_batch = max_tokens_per_batch  # Increased for ALBERT efficiency
        self.drop_last = drop_last
        
        # Calculate source distribution for balanced sampling
        self.source_counts = defaultdict(int)
        for idx in range(len(dataset)):
            item = dataset[idx]
            source_id = item['source_id'] if isinstance(item, dict) else dataset.dataset[idx]['source_id']
            self.source_counts[source_id] += 1
            
        total_samples = sum(self.source_counts.values())
        self.source_weights = {
            src: total_samples/count 
            for src, count in self.source_counts.items()
        }
        
        self.buckets = self._create_length_aware_buckets()
        
    def _create_length_aware_buckets(self):
        """Group samples by sequence length for efficient batching."""
        buckets = {
            'very_short': [],   # ≤64 tokens
            'short': [],        # 65-128 tokens
            'medium': [],       # 129-256 tokens
            'long': [],         # 257-384 tokens
            'very_long': [],    # 385-512 tokens
        }
        
        for idx in range(len(self.dataset)):
            item = self.dataset[idx]
            index_info = item if isinstance(item, dict) else self.dataset.dataset[idx]
                
            # Calculate actual sequence length
            if isinstance(index_info['attention_mask'], torch.Tensor):
                length = (index_info['attention_mask'] == 1).sum().item()
            else:
                length = sum(index_info['attention_mask'])
                
            source_id = index_info['source_id']
            
            # Assign to appropriate bucket
            if length <= 64:
                bucket = 'very_short'
            elif length <= 128:
                bucket = 'short'
            elif length <= 256:
                bucket = 'medium'
            elif length <= 384:
                bucket = 'long'
            else:
                bucket = 'very_long'
                
            buckets[bucket].append({
                'idx': idx,
                'length': length,
                'source_id': source_id,
                'weight': self.source_weights[source_id]
            })
            
        return buckets
    
    def __iter__(self):
        """Generate batches with dynamic sizing optimized for ALBERT."""
        # Larger batch sizes due to ALBERT's parameter sharing efficiency
        max_batch_sizes = {
            'very_short': self.base_batch_size * 4,
            'short': self.base_batch_size * 2,
            'medium': self.base_batch_size,
            'long': max(self.base_batch_size // 2, 4),
            'very_long': max(self.base_batch_size // 4, 2)
        }
        
        all_batches = []
        for bucket_name, bucket in self.buckets.items():
            if not bucket:
                continue
                
            max_batch_size = max_batch_sizes[bucket_name]
            current_batch = []
            current_tokens = 0
            
            # Sort by length for efficient packing
            sorted_bucket = sorted(bucket, key=lambda x: x['length'])
            
            for item in sorted_bucket:
                potential_tokens = current_tokens + item['length']
                
                if (len(current_batch) >= max_batch_size or 
                    potential_tokens > self.max_tokens_per_batch):
                    if current_batch:
                        all_batches.append([x['idx'] for x in current_batch])
                    current_batch = [item]
                    current_tokens = item['length']
                else:
                    current_batch.append(item)
                    current_tokens = potential_tokens
            
            # Add final batch if valid
            if current_batch and (not self.drop_last or 
                                len(current_batch) == max_batch_sizes[bucket_name]):
                all_batches.append([x['idx'] for x in current_batch])
        
        # Randomize batch order
        np.random.shuffle(all_batches)
        
        for batch_indices in all_batches:
            yield batch_indices
            
    def __len__(self):
        return sum((len(bucket) + self.base_batch_size - 1) // self.base_batch_size 
                  for bucket in self.buckets.values())


def create_mlm_targets(batch, tokenizer, mlm_probability=0.15):
    """
    Create masked language modeling targets for ALBERT with proper token handling.
    
    ALBERT-specific considerations:
    - Uses different special token IDs than BERT
    - Includes token_type_ids for sentence pair tasks
    - Supports vocabulary-specific masking
    
    Args:
        batch: Batch dictionary containing input_ids and attention_mask
        tokenizer: ALBERT tokenizer instance
        mlm_probability: Probability of masking tokens (default: 0.15)
        
    Returns:
        Tuple of (masked_input_ids, attention_mask, token_type_ids, labels, masking_stats)
    """
    input_ids = batch['input_ids'].clone()
    attention_mask = batch['attention_mask'].clone()
    labels = input_ids.clone()
    
    # Create token_type_ids (all zeros for single sequence)
    token_type_ids = torch.zeros_like(input_ids)
    
    # Create masking probability matrix
    probability_matrix = torch.full(labels.shape, mlm_probability)
    
    # Create mask for special tokens using ALBERT tokenizer
    special_tokens_mask = torch.tensor([
        [1 if token in tokenizer.all_special_ids else 0 for token in val]
        for val in labels.tolist()
    ])
    
    # Don't mask special tokens or padding
    probability_matrix.masked_fill_(special_tokens_mask.bool(), value=0.0)
    probability_matrix.masked_fill_(attention_mask.eq(0), value=0.0)
    
    # Select tokens to mask
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # Only compute loss on masked tokens
    
    # Apply ALBERT's masking strategy (same as BERT)
    # 80% of the time: replace with [MASK] token
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    input_ids[indices_replaced] = tokenizer.mask_token_id
    
    # 10% of the time: replace with random token
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(0, tokenizer.vocab_size, labels.shape, dtype=torch.long)
    input_ids[indices_random] = random_words[indices_random]
    
    # Calculate masking statistics for monitoring
    total_tokens = masked_indices.numel()
    stats = {
        'masked': masked_indices.sum().item() / total_tokens,
        'replaced': indices_replaced.sum().item() / total_tokens,
        'random': indices_random.sum().item() / total_tokens
    }
    
    return input_ids, attention_mask, token_type_ids, labels, stats


def create_memory_efficient_dataloaders(train_dataset, val_dataset, args):
    """
    Create DataLoaders optimized for ALBERT with high-end GPU memory.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        args: Training arguments
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    def collate_fn(batch):
        """Custom collation function optimized for ALBERT."""
        lengths = [sum(item['attention_mask']) for item in batch]
        avg_length = sum(lengths) / len(lengths)

        # Determine bucket type for monitoring
        if avg_length <= 64:
            bucket_type = 'very_short'
        elif avg_length <= 128:
            bucket_type = 'short'
        elif avg_length <= 256:
            bucket_type = 'medium'
        elif avg_length <= 384:
            bucket_type = 'long'
        else:
            bucket_type = 'very_long'
            
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'source_id': [item['source_id'] for item in batch],
            'text_idx': [item['text_idx'] for item in batch],
            'is_chunk': [item['is_chunk'] for item in batch],
            'bucket_type': bucket_type
        }

    # Optimize for A100 40GB memory
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        max_tokens_per_batch = int(min(8192, (gpu_mem * 1536)))
    else:
        max_tokens_per_batch = 4096
    
    # Optimize num_workers for high-core systems
    num_workers = min(args.num_workers, 32)
    
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=MemoryAwareSmartBatchSampler(
            train_dataset, 
            args.batch_size,
            max_tokens_per_batch=max_tokens_per_batch
        ),
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_sampler=MemoryAwareSmartBatchSampler(
            val_dataset, 
            args.batch_size,
            max_tokens_per_batch=max_tokens_per_batch
        ),
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


def train_with_gradient_accumulation(model, train_loader, val_loader, optimizer, 
                                   scheduler, monitor, args, device, tokenizer):
    """
    Train ALBERT model with gradient accumulation and mixed precision.
    
    Features:
    - Automatic mixed precision training with GradScaler
    - Memory-efficient gradient accumulation
    - Comprehensive monitoring and error handling
    - GPU memory optimization
    
    Args:
        model: ALBERT model to train
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        optimizer: Optimizer instance
        scheduler: Learning rate scheduler
        monitor: Training monitor for logging
        args: Training arguments
        device: Training device
        tokenizer: ALBERT tokenizer
        
    Returns:
        Trained model
    """
    best_model_path = os.path.join(args.output_dir, 'best_model')
    scaler = GradScaler()  # For mixed precision training
    
    total_batches = 0
    skipped_batches = 0
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        
        total_batches_per_epoch = len(train_loader)
        progress_bar = tqdm(
            enumerate(train_loader), 
            total=total_batches_per_epoch,
            desc=f"Epoch {epoch+1}/{args.epochs}"
        )
        
        epoch_skipped_batches = 0
        
        for batch_idx, batch in progress_bar:
            try:
                total_batches += 1
                
                # Create MLM targets with ALBERT-specific handling
                input_ids, attention_mask, token_type_ids, labels, mask_stats = create_mlm_targets(
                    batch, 
                    tokenizer,
                    mlm_probability=0.15
                )
                
                # Move to device with non-blocking transfer
                input_ids = input_ids.to(device, non_blocking=True)
                attention_mask = attention_mask.to(device, non_blocking=True)
                token_type_ids = token_type_ids.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                # Forward pass with mixed precision
                with autocast():
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        labels=labels
                    )
                    
                    # Scale loss for gradient accumulation
                    loss = outputs.loss / args.gradient_accumulation_steps
                
                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                
                total_loss += loss.item() * args.gradient_accumulation_steps
                
                # Gradient accumulation and optimization
                if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                    # Unscale gradients for clipping
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    
                    # Optimization step with gradient scaling
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
                
                # Memory cleanup
                del outputs
                torch.cuda.empty_cache()
                
                # Log batch metrics
                gpu_mem_alloc = torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0
                gpu_mem_reserved = torch.cuda.memory_reserved() / (1024**3) if torch.cuda.is_available() else 0
                
                monitor.log_batch({
                    'loss': loss.item() * args.gradient_accumulation_steps,
                    'batch_size': input_ids.size(0),
                    'bucket': batch.get('bucket_type', 'unknown'),
                    'source_id': batch['source_id'],
                    'mask_stats': mask_stats,
                    'gpu_mem_alloc': gpu_mem_alloc,
                    'gpu_mem_reserved': gpu_mem_reserved
                })
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f'{loss.item() * args.gradient_accumulation_steps:.4f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.2e}',
                    'gpu_mem': f'{gpu_mem_alloc:.1f}GB',
                    'progress': f'{batch_idx+1}/{total_batches_per_epoch}',
                    'skipped': f'{epoch_skipped_batches}'
                })
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    skipped_batches += 1
                    epoch_skipped_batches += 1
                    logging.warning(
                        f"OOM error in batch {batch_idx}. "
                        f"Epoch skipped: {epoch_skipped_batches}, "
                        f"Total skipped: {skipped_batches}"
                    )
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    optimizer.zero_grad()  # Clear partial gradients
                    continue
                else:
                    raise e
                    
        # Calculate epoch metrics
        avg_train_loss = total_loss / len(train_loader)
        
        # Evaluation with mixed precision
        with torch.cuda.amp.autocast():
            val_loss, bucket_stats, source_stats = evaluate_model(
                model, val_loader, device, tokenizer
            )
        
        # Log epoch metrics and check for improvement
        improved = monitor.log_epoch(
            avg_train_loss,
            val_loss,
            optimizer.param_groups[0]['lr'],
            bucket_stats,
            source_stats
        )
        
        # Save best model with safetensors
        if improved:
            model.save_pretrained(
                best_model_path,
                save_config=True,
                use_safetensors=True
            )
            logging.info(f"New best model saved with validation loss: {val_loss:.4f}")
            
        # Generate progress plots
        monitor.plot_training_progress(epoch + 1)
        
        # Check early stopping
        if monitor.should_stop(args.patience):
            print("Early stopping triggered!")
            break
            
    return model


def evaluate_model(model, val_loader, device, tokenizer):
    """
    Evaluate ALBERT model on validation set with mixed precision.
    
    Args:
        model: Model to evaluate
        val_loader: Validation DataLoader
        device: Evaluation device
        tokenizer: ALBERT tokenizer
        
    Returns:
        Tuple of (average_loss, bucket_statistics, source_statistics)
    """
    model.eval()
    total_loss = 0
    bucket_losses = defaultdict(list)
    source_losses = defaultdict(list)
    
    with torch.no_grad(), torch.cuda.amp.autocast():
        for batch in tqdm(val_loader, desc="Evaluating"):
            try:
                input_ids, attention_mask, token_type_ids, labels, _ = create_mlm_targets(
                    batch, 
                    tokenizer,
                    mlm_probability=0.15
                )
                
                input_ids = input_ids.to(device, non_blocking=True)
                attention_mask = attention_mask.to(device, non_blocking=True)
                token_type_ids = token_type_ids.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=labels
                )
                
                loss = outputs.loss
                loss_value = loss.item()
                total_loss += loss_value
                
                # Track losses by bucket and source
                bucket_type = batch.get('bucket_type', 'unknown')
                bucket_losses[bucket_type].append(loss_value)
                
                source_ids = batch['source_id']
                for source_id in source_ids:
                    source_losses[source_id].append(loss_value)
                    
            except Exception as e:
                logging.error(f"Error processing batch during evaluation: {str(e)}")
                continue
    
    # Calculate statistics
    num_batches = len(val_loader)
    if num_batches == 0:
        raise ValueError("No valid batches found during evaluation")
        
    avg_loss = total_loss / num_batches
    
    # Calculate detailed bucket statistics
    print("\nBucket Statistics:")
    bucket_stats = {}
    for bucket, losses in bucket_losses.items():
        if losses:
            avg = np.mean(losses)
            std = np.std(losses) if len(losses) > 1 else 0
            print(f"Bucket {bucket}: {len(losses)} samples, avg loss = {avg:.4f} ± {std:.4f}")
            bucket_stats[bucket] = {
                'avg_loss': avg,
                'std_loss': std,
                'num_samples': len(losses)
            }
    
    # Calculate source statistics
    source_stats = {}
    for source_id, losses in source_losses.items():
        if losses:
            avg = np.mean(losses)
            std = np.std(losses) if len(losses) > 1 else 0
            source_stats[source_id] = {
                'avg_loss': avg,
                'std_loss': std,
                'num_samples': len(losses)
            }
    
    return avg_loss, bucket_stats, source_stats


class TrainingMonitor:
    """
    Comprehensive training monitor with logging, visualization, and early stopping.
    
    Features:
    - Real-time metrics tracking with ALBERT-specific optimizations
    - Automated plot generation with professional styling
    - Early stopping with patience
    - Detailed performance analysis by text length and data source
    - GPU memory monitoring for mixed precision training
    
    Args:
        save_dir: Directory to save logs, plots, and metrics
    """
    
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': [],
            'bucket_stats': [],
            'source_stats': [],
            'gpu_metrics': [],
            'batch_metrics': [],
            'mask_stats': [],
            'validation_metrics': []
        }
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.start_time = datetime.now()
        
        # Create output directories
        os.makedirs(os.path.join(save_dir, 'plots'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'metrics'), exist_ok=True)
        
        self._setup_logging()
    
    def _setup_logging(self):
        """Configure logging for training monitoring."""
        self.logger = logging.getLogger('training_monitor')
        self.logger.setLevel(logging.INFO)
        
        # File and console handlers
        file_handler = logging.FileHandler(
            os.path.join(self.save_dir, 'training.log')
        )
        console_handler = logging.StreamHandler()
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def log_batch(self, batch_metrics):
        """Log metrics for individual batch with ALBERT-specific tracking."""
        self.metrics['batch_metrics'].append({
            'loss': batch_metrics['loss'],
            'batch_size': batch_metrics['batch_size'],
            'bucket': batch_metrics['bucket'],
            'source_id': batch_metrics['source_id'],
            'mask_stats': batch_metrics['mask_stats'],
            'gpu_mem_alloc': batch_metrics.get('gpu_mem_alloc', 0),
            'gpu_mem_reserved': batch_metrics.get('gpu_mem_reserved', 0)
        })
    
    def log_epoch(self, train_loss, val_loss, lr, bucket_stats, source_stats):
        """
        Log epoch-level metrics and check for improvement.
        
        Returns:
            bool: True if this epoch achieved the best validation loss
        """
        self.metrics['train_loss'].append(train_loss)
        self.metrics['val_loss'].append(val_loss)
        self.metrics['learning_rates'].append(lr)
        self.metrics['bucket_stats'].append(bucket_stats)
        self.metrics['source_stats'].append(source_stats)
        
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.max_memory_allocated() / 1024**2  # Convert to MB
            self.metrics['gpu_metrics'].append(gpu_mem)
        
        # Check for improvement
        improved = False
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            improved = True
        else:
            self.patience_counter += 1
        
        # Log epoch summary
        self.logger.info(
            f"Epoch Summary - Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, LR: {lr:.2e}, "
            f"Best Val Loss: {self.best_val_loss:.4f}, "
            f"Patience: {self.patience_counter}"
        )
        
        return improved
    
    def should_stop(self, patience):
        """Check if early stopping should be triggered."""
        return self.patience_counter >= patience
    
    def plot_training_progress(self, epoch):
        """Generate comprehensive training progress visualizations."""
        try:
            plots_dir = os.path.join(self.save_dir, 'plots', f'epoch_{epoch}')
            os.makedirs(plots_dir, exist_ok=True)
            
            # Use available matplotlib style
            try:
                plt.style.use('seaborn-v0_8-dark-palette')
            except:
                try:
                    plt.style.use('seaborn')
                except:
                    pass  # Use default style
                
            # Source name mapping for visualization
            source_names = {
                0: 'MSHA',
                1: 'Elsevier',
                2: 'IOGP',
                3: 'iChem',
                4: 'OSHA',
                5: 'NTSB',
                6: 'FRA'
            }
            
            if not self.metrics['train_loss'] or not self.metrics['val_loss']:
                self.logger.warning("No loss data available for plotting")
                return
                
            epochs = range(1, len(self.metrics['train_loss']) + 1)
            
            # 1. Training and Validation Loss
            self._create_loss_plot(epochs, plots_dir)
            
            # 2. Learning Rate Schedule
            if self.metrics['learning_rates']:
                self._create_lr_plot(epochs, plots_dir)
            
            # 3. GPU Memory Usage
            if self.metrics['gpu_metrics']:
                self._create_gpu_plot(epochs, plots_dir)
            
            # 4. Performance by Text Length Buckets
            if self.metrics['bucket_stats']:
                self._create_bucket_plot(plots_dir)
            
            # 5. Performance by Data Source
            if self.metrics['source_stats']:
                self._create_source_plot(plots_dir, source_names)
                
        except Exception as e:
            self.logger.error(f"Error in plot_training_progress: {str(e)}")
            plt.close('all')
    
    def _create_loss_plot(self, epochs, plots_dir):
        """Create training and validation loss plot."""
        try:
            plt.figure(figsize=(12, 8))
            plt.plot(epochs, self.metrics['train_loss'], 'b-', label='Training Loss', linewidth=2)
            plt.plot(epochs, self.metrics['val_loss'], 'r-', label='Validation Loss', linewidth=2)
            plt.title('ALBERT Training and Validation Loss', pad=20, fontsize=16)
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Loss', fontsize=12)
            plt.legend(fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            loss_plot_path = os.path.join(plots_dir, 'loss_plot.png')
            plt.savefig(loss_plot_path, bbox_inches='tight', dpi=300)
            plt.close()
            self.logger.info(f"Saved loss plot to {loss_plot_path}")
        except Exception as e:
            self.logger.error(f"Error creating loss plot: {str(e)}")
            plt.close()
    
    def _create_lr_plot(self, epochs, plots_dir):
        """Create learning rate schedule plot."""
        try:
            plt.figure(figsize=(12, 8))
            plt.plot(epochs, self.metrics['learning_rates'], 'g-', linewidth=2)
            plt.title('Learning Rate Schedule', pad=20, fontsize=16)
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Learning Rate', fontsize=12)
            plt.yscale('log')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            lr_plot_path = os.path.join(plots_dir, 'learning_rate.png')
            plt.savefig(lr_plot_path, bbox_inches='tight', dpi=300)
            plt.close()
            self.logger.info(f"Saved learning rate plot to {lr_plot_path}")
        except Exception as e:
            self.logger.error(f"Error creating learning rate plot: {str(e)}")
            plt.close()
    
    def _create_gpu_plot(self, epochs, plots_dir):
        """Create GPU memory usage plot."""
        try:
            plt.figure(figsize=(12, 8))
            plt.plot(epochs, self.metrics['gpu_metrics'], 'r-', linewidth=2)
            plt.title('GPU Memory Usage (Mixed Precision)', pad=20, fontsize=16)
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Memory (MB)', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            gpu_plot_path = os.path.join(plots_dir, 'gpu_memory.png')
            plt.savefig(gpu_plot_path, bbox_inches='tight', dpi=300)
            plt.close()
            self.logger.info(f"Saved GPU memory plot to {gpu_plot_path}")
        except Exception as e:
            self.logger.error(f"Error creating GPU memory plot: {str(e)}")
            plt.close()
    
    def _create_bucket_plot(self, plots_dir):
        """Create performance by text length bucket plot."""
        try:
            latest_bucket_stats = self.metrics['bucket_stats'][-1]
            bucket_order = ['very_short', 'short', 'medium', 'long', 'very_long']
            
            buckets = []
            avg_losses = []
            std_losses = []
            
            for bucket in bucket_order:
                if bucket in latest_bucket_stats:
                    buckets.append(bucket.replace('_', ' ').title())
                    avg_losses.append(latest_bucket_stats[bucket]['avg_loss'])
                    std_losses.append(latest_bucket_stats[bucket]['std_loss'])
            
            plt.figure(figsize=(12, 8))
            bars = plt.bar(buckets, avg_losses, yerr=std_losses, capsize=5, alpha=0.8)
            
            plt.title('ALBERT Performance by Text Length Category', pad=20, fontsize=16)
            plt.xlabel('Text Length Category', fontsize=12)
            plt.ylabel('Average Loss', fontsize=12)
            plt.grid(True, alpha=0.3, axis='y')
            plt.xticks(rotation=45, ha='right')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}',
                        ha='center', va='bottom', fontsize=10)
            
            plt.tight_layout()
            
            bucket_plot_path = os.path.join(plots_dir, 'bucket_performance.png')
            plt.savefig(bucket_plot_path, bbox_inches='tight', dpi=300)
            plt.close()
            self.logger.info(f"Saved bucket performance plot to {bucket_plot_path}")
        except Exception as e:
            self.logger.error(f"Error creating bucket performance plot: {str(e)}")
            plt.close()
    
    def _create_source_plot(self, plots_dir, source_names):
        """Create performance by data source plot."""
        try:
            latest_source_stats = self.metrics['source_stats'][-1]
            
            sources = []
            avg_losses = []
            std_losses = []
            
            # Sort by average loss for better visualization
            sorted_sources = sorted(latest_source_stats.items(), 
                                 key=lambda x: x[1]['avg_loss'])
            
            for source_id, stats in sorted_sources:
                source_name = source_names.get(int(source_id), f'Source_{source_id}')
                sources.append(source_name)
                avg_losses.append(stats['avg_loss'])
                std_losses.append(stats['std_loss'])
            
            plt.figure(figsize=(14, 8))
            bars = plt.bar(sources, avg_losses, yerr=std_losses, capsize=5, alpha=0.8)
            
            plt.title('ALBERT Performance by Data Source', pad=20, fontsize=16)
            plt.xlabel('Data Source', fontsize=12)
            plt.ylabel('Average Loss', fontsize=12)
            plt.grid(True, alpha=0.3, axis='y')
            plt.xticks(rotation=45, ha='right')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}',
                        ha='center', va='bottom', fontsize=10)
            
            plt.tight_layout()
            
            source_plot_path = os.path.join(plots_dir, 'source_performance.png')
            plt.savefig(source_plot_path, bbox_inches='tight', dpi=300)
            plt.close()
            self.logger.info(f"Saved source performance plot to {source_plot_path}")
        except Exception as e:
            self.logger.error(f"Error creating source performance plot: {str(e)}")
            plt.close()
    
    def save_metrics(self):
        """Save all training metrics to JSON file."""
        metrics_file = os.path.join(self.save_dir, 'metrics', 'training_metrics.json')
        
        def convert_to_serializable(obj):
            """Convert tensors and numpy objects to JSON-serializable types."""
            if torch.is_tensor(obj):
                return obj.item() if obj.numel() == 1 else obj.tolist()
            elif isinstance(obj, dict):
                return {str(k): convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int64, np.int32, np.float64, np.float32)):
                return obj.item()
            return obj
        
        # Create serializable metrics
        serializable_metrics = {
            'train_loss': convert_to_serializable(self.metrics['train_loss']),
            'val_loss': convert_to_serializable(self.metrics['val_loss']),
            'learning_rates': convert_to_serializable(self.metrics['learning_rates']),
            'best_val_loss': float(self.best_val_loss),
            'training_time': str(datetime.now() - self.start_time),
            'bucket_stats': convert_to_serializable(self.metrics['bucket_stats']),
            'source_stats': convert_to_serializable(self.metrics['source_stats']),
            'gpu_metrics': convert_to_serializable(self.metrics['gpu_metrics'])
        }
        
        with open(metrics_file, 'w') as f:
            json.dump(serializable_metrics, f, indent=4)
        
        self.logger.info(f"Training metrics saved to {metrics_file}")


def main(args):
    """
    Main training function optimized for ALBERT with mixed precision.
    
    Handles:
    - ALBERT model initialization with optimal configurations
    - Mixed precision training setup
    - High-performance data loading
    - Comprehensive monitoring and checkpointing
    
    Args:
        args: Parsed command line arguments
    """
    # Setup output directory and logging
    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )
    
    # Save configuration for reproducibility
    config_path = os.path.join(args.output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=4)
    logging.info(f"Configuration saved to {config_path}")
    
    # Initialize Weights & Biases if requested and available
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.init(project="albert-continual-training", config=vars(args))
        logging.info("Weights & Biases logging initialized")
    elif args.use_wandb and not WANDB_AVAILABLE:
        logging.warning("Weights & Biases requested but not available. Install with: pip install wandb")
    
    # Get dataset files
    csv_files = [
        os.path.join(args.data_dir, f) 
        for f in os.listdir(args.data_dir) 
        if f.endswith('.csv')
    ]
    
    if not csv_files:
        raise ValueError(f"No CSV files found in {args.data_dir}")
    
    logging.info(f"Found {len(csv_files)} CSV files: {[os.path.basename(f) for f in csv_files]}")
    
    # Define text column mapping for each dataset
    text_columns = {
        'accident-data-MSHA.csv': 'Narrative',
        'combined_abstracts-elsvier.csv': 'Abstract',
        'final_merged_data-IOGP.csv': ['Narrative', 'What Went Wrong'],
        'final_merged_data-iChem.csv': 'Abstract',
        'merged_abstracts-OSHA.csv': 'Abstract',
        'merged_narratives-NTSB.csv': 'merged_narrative',
        'merged_narratives_FRA_railroad.csv': 'Narrative'
    }
    
    # Initialize ALBERT tokenizer with optimizations
    tokenizer = AlbertTokenizer.from_pretrained(
        'albert-base-v2',
        do_lower_case=True,
        use_fast=True
    )
    logging.info("ALBERT tokenizer initialized")
    
    # Setup device and optimizations
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    if torch.cuda.is_available():
        # Enable CUDA optimizations
        torch.backends.cudnn.benchmark = True
        gpu_info = torch.cuda.get_device_properties(0)
        logging.info(f"GPU: {gpu_info.name}")
        logging.info(f"Available GPU memory: {gpu_info.total_memory / 1024**3:.2f} GB")
        logging.info(f"CUDA Version: {torch.version.cuda}")
        logging.info("Mixed precision training enabled")
    
    # Configure ALBERT model for optimal training
    model_config = AlbertConfig.from_pretrained('albert-base-v2')
    model_config.gradient_checkpointing = True
    model_config.classifier_dropout_prob = args.classifier_dropout_prob
    model_config.hidden_dropout_prob = args.hidden_dropout_prob
    
    # Initialize ALBERT model
    model = AlbertForMaskedLM.from_pretrained(
        'albert-base-v2',
        config=model_config
    )
    model.to(device)
    logging.info("ALBERT model initialized with gradient checkpointing")
    
    # Create datasets
    logging.info("Creating datasets...")
    full_dataset = SmartMLMDataset(
        csv_files, 
        text_columns, 
        tokenizer,
        max_length=args.max_length,
        stride=args.stride
    )
    
    # Split into train/validation sets
    indices = list(range(len(full_dataset)))
    train_indices, val_indices = train_test_split(
        indices, 
        test_size=0.1, 
        random_state=42,
        shuffle=True
    )
    
    train_dataset = SubsetDatasetWithIndices(full_dataset, train_indices)
    val_dataset = SubsetDatasetWithIndices(full_dataset, val_indices)
    
    logging.info(f"Dataset split - Train: {len(train_dataset)}, Validation: {len(val_dataset)}")
    
    # Create data loaders
    train_loader, val_loader = create_memory_efficient_dataloaders(
        train_dataset, 
        val_dataset, 
        args
    )
    
    # Initialize optimizer with ALBERT-specific settings
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Calculate training steps
    total_batches_per_epoch = len(train_loader)
    total_steps = total_batches_per_epoch * args.epochs // args.gradient_accumulation_steps
    warmup_steps = int(total_steps * args.warmup_proportion)
    
    logging.info(f"Training configuration:")
    logging.info(f"  Total batches per epoch: {total_batches_per_epoch}")
    logging.info(f"  Gradient accumulation steps: {args.gradient_accumulation_steps}")
    logging.info(f"  Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    logging.info(f"  Total optimization steps: {total_steps}")
    logging.info(f"  Warmup steps: {warmup_steps}")
    logging.info(f"  Using mixed precision: {args.use_mixed_precision}")
    
    # Create learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Initialize training monitor
    monitor = TrainingMonitor(args.output_dir)
    
    try:
        logging.info("Starting ALBERT training...")
        
        # Main training loop
        model = train_with_gradient_accumulation(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            monitor=monitor,
            args=args,
            device=device,
            tokenizer=tokenizer
        )
        
        # Save final model with safetensors
        final_model_path = os.path.join(args.output_dir, 'final_model')
        model.save_pretrained(
            final_model_path,
            save_config=True,
            use_safetensors=args.use_safetensors
        )
        tokenizer.save_pretrained(final_model_path)
        logging.info(f"Final model and tokenizer saved to {final_model_path}")
        
        # Save training metrics
        monitor.save_metrics()
        
        logging.info("ALBERT training completed successfully!")
        
    except KeyboardInterrupt:
        logging.info("Training interrupted by user")
        # Save current state for potential resuming
        interrupt_model_path = os.path.join(args.output_dir, 'interrupted_model')
        model.save_pretrained(interrupt_model_path, use_safetensors=args.use_safetensors)
        monitor.save_metrics()
        logging.info(f"Model state saved to {interrupt_model_path} for potential resuming")
        
    except Exception as e:
        logging.error(f"Training failed with error: {str(e)}")
        logging.error(f"Traceback: {traceback.format_exc()}")
        raise
        
    finally:
        # Cleanup
        if args.use_wandb and WANDB_AVAILABLE:
            wandb.finish()
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        logging.info("Training script finished")


def parse_arguments():
    """Parse and validate command line arguments for ALBERT training."""
    parser = argparse.ArgumentParser(
        description="ALBERT Continual Training with Smart Data Handling and Mixed Precision",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    data_group = parser.add_argument_group('Data Configuration')
    data_group.add_argument(
        "--data_dir", 
        type=str, 
        required=True,
        help="Directory containing CSV files for training"
    )
    data_group.add_argument(
        "--output_dir", 
        type=str, 
        required=True,
        help="Output directory for models, logs, and plots"
    )
    
    # Model arguments
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument(
        "--max_length", 
        type=int, 
        default=512,
        help="Maximum sequence length for ALBERT input"
    )
    model_group.add_argument(
        "--stride", 
        type=int, 
        default=256,
        help="Stride for sliding window over long texts"
    )
    model_group.add_argument(
        "--use_fast_tokenizer", 
        type=bool, 
        default=True,
        help="Use fast tokenizer implementation"
    )
    
    # Training arguments
    training_group = parser.add_argument_group('Training Configuration')
    training_group.add_argument(
        "--batch_size", 
        type=int, 
        default=16,
        help="Base batch size (optimized for ALBERT efficiency)"
    )
    training_group.add_argument(
        "--gradient_accumulation_steps", 
        type=int, 
        default=4,
        help="Number of gradient accumulation steps"
    )
    training_group.add_argument(
        "--max_tokens_per_batch", 
        type=int, 
        default=8192,
        help="Maximum tokens per batch (increased for ALBERT)"
    )
    training_group.add_argument(
        "--epochs", 
        type=int, 
        default=40,
        help="Number of training epochs"
    )
    training_group.add_argument(
        "--learning_rate", 
        type=float, 
        default=1e-4,
        help="Peak learning rate (optimized for ALBERT)"
    )
    training_group.add_argument(
        "--weight_decay", 
        type=float, 
        default=0.01,
        help="Weight decay for regularization"
    )
    training_group.add_argument(
        "--warmup_proportion", 
        type=float, 
        default=0.1,
        help="Proportion of training steps for warmup"
    )
    training_group.add_argument(
        "--max_grad_norm", 
        type=float, 
        default=2.0,
        help="Maximum gradient norm for clipping (increased for ALBERT)"
    )
    
    # System arguments
    system_group = parser.add_argument_group('System Configuration')
    system_group.add_argument(
        "--num_workers", 
        type=int, 
        default=32,
        help="Number of data loading workers (optimized for high-core systems)"
    )
    system_group.add_argument(
        "--patience", 
        type=int, 
        default=3,
        help="Early stopping patience (epochs without improvement)"
    )
    system_group.add_argument(
        "--use_wandb", 
        action="store_true",
        help="Enable Weights & Biases logging"
    )
    
    # ALBERT-specific arguments
    albert_group = parser.add_argument_group('ALBERT-Specific Configuration')
    albert_group.add_argument(
        "--classifier_dropout_prob", 
        type=float, 
        default=0.1,
        help="Dropout probability for classifier layers"
    )
    albert_group.add_argument(
        "--hidden_dropout_prob", 
        type=float, 
        default=0.1,
        help="Dropout probability for hidden layers"
    )
    albert_group.add_argument(
        "--use_mixed_precision", 
        type=bool, 
        default=True,
        help="Enable mixed precision training with automatic loss scaling"
    )
    albert_group.add_argument(
        "--use_safetensors", 
        type=bool, 
        default=True,
        help="Use safetensors format for model saving"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.data_dir):
        raise ValueError(f"Data directory does not exist: {args.data_dir}")
    
    if args.warmup_proportion < 0 or args.warmup_proportion > 1:
        raise ValueError(f"Warmup proportion must be between 0 and 1, got {args.warmup_proportion}")
    
    if args.classifier_dropout_prob < 0 or args.classifier_dropout_prob > 1:
        raise ValueError(f"Classifier dropout must be between 0 and 1, got {args.classifier_dropout_prob}")
    
    if args.hidden_dropout_prob < 0 or args.hidden_dropout_prob > 1:
        raise ValueError(f"Hidden dropout must be between 0 and 1, got {args.hidden_dropout_prob}")
    
    return args


if __name__ == "__main__":
    try:
        args = parse_arguments()
        main(args)
    except KeyboardInterrupt:
        logging.info("Script interrupted by user")
    except Exception as e:
        logging.error(f"Script failed: {str(e)}")
        logging.error(f"Traceback: {traceback.format_exc()}")
        raise