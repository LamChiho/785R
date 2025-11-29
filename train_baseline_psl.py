"""
Training Script with REINFORCE + Learned Baseline + PSL
========================================================

Differences from RLOO:
1. Uses learned value network V(s) instead of empirical baseline
2. Trains both policy and value network
3. Works with any batch size (K=1)
4. Requires value loss in addition to policy loss

Usage:
    python train_baseline_psl.py --use_psl --use_baseline --lambda_psl 0.5
"""

import os
import transformers
from loader_chunks_fix import DataLoader
import numpy as np
from time import time
import torch
from torch.optim import AdamW
from tqdm import tqdm
from baseline_model import BaselineGPT2Classifier  # NEW
import random
import argparse

def run_epoch(model, data_iterator, optimizer, value_optimizer, scheduler, phase='train', batch_size=16, epoch=0):
    """
    Training/validation epoch with learned baseline and PSL.
    
    Key Changes:
    1. Separate optimizer for value network
    2. Update value network with MSE loss
    3. Log value network metrics
    """
    if phase == 'train':
        model.train()
    else:
        model.eval()

    t_correct = 0
    t_loss = 0
    n_all = 0
    t0 = time()
    
    # Metrics tracking
    psl_losses = []
    reinforce_losses = []
    value_losses = []
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
        
        # Forward pass (single pass, unlike RLOO's two passes)
        batch_pred, batch_loss, value_loss, proof, metrics = model(
            input_idxs=x_batch, masks=m_batch, segment_idxs=i_batch,
            projections=p_batch, hypothesis_len=hypo_len, ys=y_batch,
            phrase_start=start_batch, phrase_end=end_batch, hints=hints, 
            train=(phase == 'train')
        )
        
        batch_loss = batch_loss.mean()
        
        # ========================================================
        # FIX: Update model (compute both backward passes first!)
        # ========================================================
        if phase == 'train':
            # Zero gradients for BOTH optimizers
            optimizer.zero_grad()
            value_optimizer.zero_grad()
            
            # Compute BOTH backward passes (accumulate gradients)
            batch_loss.backward(retain_graph=True)  # Policy gradients
            value_loss.backward()  # Value gradients
            
            # Update BOTH networks (after all gradients computed)
            optimizer.step()
            value_optimizer.step()
            scheduler.step()
        
        # Track metrics
        y_batch = torch.clamp(y_batch, min=0, max=2).type(torch.int64)
        n_sample = y_batch.shape[0]
        n_all += n_sample
        t_loss += batch_loss.item() * n_sample
        t_correct += torch.sum(torch.argmax(batch_pred, dim=1) == y_batch).item()
        
        # Track losses
        psl_losses.append(metrics['psl_loss'])
        reinforce_losses.append(metrics['reinforce_loss'])
        value_losses.append(metrics['value_loss'])
        if 'transitivity_loss' in metrics:
            transitivity_losses.append(metrics['transitivity_loss'])
    
    # Print epoch summary
    accuracy = 100 * t_correct / n_all
    avg_loss = t_loss / n_all
    print(f"{phase} Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%, Time: {time() - t0:.2f}s")
    
    # Print metrics
    if len(psl_losses) > 0:
        print(f"  PSL Loss: {np.mean(psl_losses):.4f}")
        print(f"  REINFORCE Loss: {np.mean(reinforce_losses):.4f}")
        print(f"  Value Loss: {np.mean(value_losses):.4f}")
        if len(transitivity_losses) > 0:
            print(f"  Transitivity Loss: {np.mean(transitivity_losses):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_psl', action='store_true', help='Enable PSL regularization')
    parser.add_argument('--use_baseline', action='store_true', help='Enable learned baseline')
    parser.add_argument('--lambda_psl', type=float, default=0.5, help='PSL loss weight')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--value_lr', type=float, default=1e-4, help='Value network learning rate')
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
    
    # Load data
    print("\n" + "="*70)
    print("Loading Datasets")
    print("="*70)
    
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
    
    # Initialize model with learned baseline
    model = BaselineGPT2Classifier(
        n_class=3, 
        lambda_psl=args.lambda_psl, 
        use_psl=args.use_psl,
        use_baseline=args.use_baseline
    ).cuda()
    
    print(f"Model Configuration:")
    print(f"  PSL Regularization: {'ENABLED' if args.use_psl else 'DISABLED'}")
    print(f"  Learned Baseline: {'ENABLED' if args.use_baseline else 'DISABLED'}")
    print(f"  Lambda PSL: {args.lambda_psl}")
    print(f"  Batch Size: {args.batch_size}")
    
    # Optimizer for policy network
    num_train_steps = int(len(train_iterator) / args.batch_size * args.n_epochs)
    num_warmup_steps = int(num_train_steps * 0.1)
    optimizer = AdamW(
        filter(lambda x: x.requires_grad, model.parameters()), 
        lr=args.learning_rate, 
        eps=1e-8
    )
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=num_warmup_steps, 
        num_training_steps=num_train_steps
    )
    
    # Separate optimizer for value network
    value_optimizer = AdamW(
        model.value_network.parameters(), 
        lr=args.value_lr,  # Higher LR for value network
        eps=1e-8
    )
    
    # Training loop
    print('Start Training (REINFORCE + Baseline + PSL)...')
    for epoch in range(args.n_epochs):
        print(f'\nEpoch {epoch}...')
        run_epoch(model, train_iterator, optimizer, value_optimizer, scheduler, phase='train', batch_size=args.batch_size, epoch=epoch)
        
        # Save checkpoint
        save_path = f'./checkpoints/baseline_psl_epoch{epoch}.pt'
        torch.save({
            'model_state_dict': model.state_dict(),
            'value_network_state_dict': model.value_network.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'value_optimizer_state_dict': value_optimizer.state_dict(),
        }, save_path)
        
        # Validation
        with torch.no_grad():
            run_epoch(model, dev_iterator, optimizer, value_optimizer, scheduler, phase='dev', batch_size=args.batch_size, epoch=epoch)
            run_epoch(model, test_iterator, optimizer, value_optimizer, scheduler, phase='test', batch_size=args.batch_size, epoch=epoch)
        print('')