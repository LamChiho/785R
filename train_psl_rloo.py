"""
Training Script with PSL Regularization and RLOO Baseline
==========================================================

Modifications from train_chunks_fixrev.py:
1. Uses EnhancedGPT2Classifier instead of GPT2Classifier
2. Implements RLOO baseline: computes batch rewards for leave-one-out
3. Logs PSL metrics (transitivity loss, composition consistency)
4. Supports toggling PSL/RLOO via command-line arguments

Usage:
    python train_psl_rloo.py --use_psl --use_rloo --lambda_psl 0.5
"""

import os
import transformers
from loader_chunks_fix import DataLoader
import numpy as np
from time import time
import torch
from torch.optim import AdamW
from tqdm import tqdm
from enhanced_sampling_model import EnhancedGPT2Classifier
import random
import argparse

def run_epoch(model, data_iterator, optimizer, scheduler, phase='train', batch_size=16, epoch=0):
    """
    Training/validation epoch with PSL and RLOO.
    
    Key Changes:
    1. Accumulates batch rewards for RLOO baseline
    2. Logs PSL-specific metrics
    3. Handles batch-level RLOO computation
    """
    if phase == 'train':
        model.train()
    else:
        model.eval()

    t_correct = 0
    t_loss = 0
    n_all = 0
    t0 = time()
    
    # NEW: Metrics tracking
    psl_losses = []
    reinforce_losses = []
    transitivity_losses = []

    for ids, x_batch, m_batch, i_batch, p_batch, y_batch, hypo_len, start, end, hints in tqdm(
        data_iterator.sampled_batch(batch_size=batch_size, phase=phase),
        total=int(len(data_iterator) / batch_size), 
        ascii=True
    ):
        x_batch = torch.tensor(x_batch, dtype=torch.int64).cuda()
        m_batch = torch.tensor(m_batch, dtype=torch.float32).cuda()
        i_batch = torch.tensor(i_batch, dtype=torch.int64).cuda()
        p_batch = torch.tensor(p_batch, dtype=torch.int64).cuda()
        y_batch = torch.tensor(y_batch, dtype=torch.int64).cuda()
        hypo_len = torch.tensor(hypo_len, dtype=torch.int64).cuda()
        start_batch = torch.tensor(start, dtype=torch.int64).cuda()
        end_batch = torch.tensor(end, dtype=torch.int64).cuda()
        
        # ---------------------------------------------------------------
        # NEW: TWO-PASS RLOO IMPLEMENTATION
        # ---------------------------------------------------------------
        if phase == 'train' and model.use_rloo:
            # Pass 1: Compute rewards for all samples in batch (for baseline)
            with torch.no_grad():
                _, _, (history, _), _ = model(
                    input_idxs=x_batch, masks=m_batch, segment_idxs=i_batch,
                    projections=p_batch, hypothesis_len=hypo_len, ys=y_batch,
                    phrase_start=start_batch, phrase_end=end_batch, hints=hints, 
                    train=False  # No sampling variance
                )
                # Compute final rewards
                batch_rewards = []
                for i in range(x_batch.shape[0]):
                    last_state = history[i, hypo_len[i]].item()
                    label_rels = {0: [0, 1], 1: [3, 4], 2: [2, 5, 6]}[y_batch[i].item()]
                    reward = 1.0 if last_state in label_rels else -1.0
                    batch_rewards.append(reward)
                batch_rewards = torch.tensor(batch_rewards, device=x_batch.device)
            
            # Pass 2: Forward with RLOO baseline
            batch_pred, batch_loss, proof, metrics = model(
                input_idxs=x_batch, masks=m_batch, segment_idxs=i_batch,
                projections=p_batch, hypothesis_len=hypo_len, ys=y_batch,
                phrase_start=start_batch, phrase_end=end_batch, hints=hints, 
                train=True, batch_rewards=batch_rewards  # Pass rewards for RLOO
            )
        else:
            # Standard forward pass (no RLOO)
            batch_pred, batch_loss, proof, metrics = model(
                input_idxs=x_batch, masks=m_batch, segment_idxs=i_batch,
                projections=p_batch, hypothesis_len=hypo_len, ys=y_batch,
                phrase_start=start_batch, phrase_end=end_batch, hints=hints, 
                train=(phase == 'train')
            )
        
        batch_loss = batch_loss.mean()
        
        # Update model
        if phase == 'train':
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            scheduler.step()
        
        # Track metrics
        y_batch = torch.clamp(y_batch, min=0, max=2).type(torch.int64)
        n_sample = y_batch.shape[0]
        n_all += n_sample
        t_loss += batch_loss.item() * n_sample
        t_correct += torch.sum(torch.argmax(batch_pred, dim=1) == y_batch).item()
        
        # NEW: Track PSL metrics
        psl_losses.append(metrics['psl_loss'])
        reinforce_losses.append(metrics['reinforce_loss'])
        if 'transitivity_loss' in metrics:
            transitivity_losses.append(metrics['transitivity_loss'])
    
    # Print epoch summary
    accuracy = 100 * t_correct / n_all
    avg_loss = t_loss / n_all
    print(f"{phase} Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%, Time: {time() - t0:.2f}s")
    
    # NEW: Print PSL metrics
    if len(psl_losses) > 0:
        print(f"  PSL Loss: {np.mean(psl_losses):.4f}")
        print(f"  REINFORCE Loss: {np.mean(reinforce_losses):.4f}")
        if len(transitivity_losses) > 0:
            print(f"  Transitivity Loss: {np.mean(transitivity_losses):.4f}")

if __name__ == "__main__":
    # ---------------------------------------------------------------
    # NEW: Command-line arguments for PSL and RLOO
    # ---------------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_psl', action='store_true', help='Enable PSL regularization')
    parser.add_argument('--use_rloo', action='store_true', help='Enable RLOO baseline')
    parser.add_argument('--lambda_psl', type=float, default=0.5, help='PSL loss weight')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--n_epochs', type=int, default=25, help='Number of epochs')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Data paths
    base_dir = './data/snli_gpt2'
    train_file = f'{base_dir}/train_records3.pkl'
    train_rev_file = f'{base_dir}/train_records3_rev.pkl'
    train_rev_id = f'{base_dir}/rev3_train.pkl'
    
    dev_file = f'{base_dir}/dev_records3.pkl'
    dev_rev_file = f'{base_dir}/dev_records3_rev.pkl'
    dev_rev_id = f'{base_dir}/rev3_dev.pkl'
    
    test_file = f'{base_dir}/test_records3.pkl'
    
    # Check which files exist
    print("\n" + "="*70)
    print("Loading Datasets")
    print("="*70)
    print(f"Main files:")
    print(f"  Train: {'?' if os.path.exists(train_file) else '?'} {train_file}")
    print(f"  Dev:   {'?' if os.path.exists(dev_file) else '?'} {dev_file}")
    print(f"  Test:  {'?' if os.path.exists(test_file) else '?'} {test_file}")
    print(f"\nRevision files (optional, will be created if missing):")
    print(f"  Train Rev: {'?' if os.path.exists(train_rev_file) else '?'} {train_rev_file}")
    print(f"  Dev Rev:   {'?' if os.path.exists(dev_rev_file) else '?'} {dev_rev_file}")
    print("="*70 + "\n")
    
    # Load data (skip revision files if missing)
    train_iterator = DataLoader(
        train_file,
        rev_data_file=train_rev_file if os.path.exists(train_rev_file) else None,
        rev_id=train_rev_id if os.path.exists(train_rev_id) else None
    )
    
    dev_iterator = DataLoader(
        dev_file,
        rev_data_file=dev_rev_file if os.path.exists(dev_rev_file) else None,
        rev_id=dev_rev_id if os.path.exists(dev_rev_id) else None
    )
    
    test_iterator = DataLoader(test_file)
    
    # ---------------------------------------------------------------
    # NEW: Initialize Enhanced Model with PSL and RLOO
    # ---------------------------------------------------------------
    model = EnhancedGPT2Classifier(
        n_class=3, 
        lambda_psl=args.lambda_psl, 
        use_psl=args.use_psl,
        use_rloo=args.use_rloo
    ).cuda()
    
    print(f"Model Configuration:")
    print(f"  PSL Regularization: {'ENABLED' if args.use_psl else 'DISABLED'}")
    print(f"  RLOO Baseline: {'ENABLED' if args.use_rloo else 'DISABLED'}")
    print(f"  Lambda PSL: {args.lambda_psl}")
    print(f"  Batch Size: {args.batch_size}")
    
    # Optimizer and scheduler
    num_train_steps = int(len(train_iterator) / args.batch_size * args.n_epochs)
    num_warmup_steps = int(num_train_steps * 0.1)
    optimizer = AdamW(  # FIXED: Don't use transformers.AdamW
        filter(lambda x: x.requires_grad, model.parameters()), 
        lr=args.learning_rate, 
        eps=1e-8
    )
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=num_warmup_steps, 
        num_training_steps=num_train_steps
    )
    
    # Training loop
    print('Start Training...')
    for epoch in range(args.n_epochs):
        print(f'\nEpoch {epoch}...')
        run_epoch(model, train_iterator, optimizer, scheduler, phase='train', batch_size=args.batch_size, epoch=epoch)
        
        # Save checkpoint
        save_path = f'./checkpoints/psl_rloo_epoch{epoch}.pt'
        torch.save(model.state_dict(), save_path)
        
        # Validation
        with torch.no_grad():
            run_epoch(model, dev_iterator, optimizer, scheduler, phase='dev', batch_size=args.batch_size, epoch=epoch)
            run_epoch(model, test_iterator, optimizer, scheduler, phase='test', batch_size=args.batch_size, epoch=epoch)
        print('')