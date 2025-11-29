import os
from torch.optim import AdamW  # FIXED: Use torch.optim instead of transformers
import transformers  # Keep this for scheduler
from loader_chunks_fix import DataLoader
import numpy as np
from time import time
import datetime
import torch
from tqdm import tqdm
from sampling_model_fix3 import GPT2Classifier
from torch.nn.parallel import DataParallel
import random
import json
import pickle
def run_epoch(model, data_iterator, optimizer, scheduler, phase='train', batch_size=16):

    if phase == 'train':
        model.train()
    else:
        model.eval()

    rev_label = []
    t_correct = 0
    t_loss = 0
    n_all = 0
    t0 = time()
    #count = 0

    for ids, x_batch, m_batch, i_batch, p_batch, y_batch, hypo_len, start, end, hints in tqdm(data_iterator.sampled_batch(batch_size=batch_size, phase=phase),
                                                   total=int(len(data_iterator) / batch_size), ascii=True):
        #if count == 28:
        #    print('#')
        #count += 1
        x_batch = torch.tensor(x_batch, dtype=torch.int64).cuda()
        m_batch = torch.tensor(m_batch, dtype=torch.float32).cuda()
        i_batch = torch.tensor(i_batch, dtype=torch.int64).cuda()
        p_batch = torch.tensor(p_batch, dtype=torch.int64).cuda()
        y_batch = torch.tensor(y_batch, dtype=torch.int64).cuda()
        hypo_len = torch.tensor(hypo_len, dtype=torch.int64).cuda()
        start_batch = torch.tensor(start, dtype=torch.int64).cuda()
        end_batch = torch.tensor(end, dtype=torch.int64).cuda()



        # forward
        #print(x_batch)
        batch_pred, batch_loss, proof = model(input_idxs=x_batch, masks=m_batch, segment_idxs=i_batch,
                                       projections=p_batch, hypothesis_len=hypo_len, ys=y_batch,
                                       phrase_start=start_batch, phrase_end=end_batch, hints=hints, train=phase=='train')
        batch_loss = batch_loss.mean()
        # update model params
        if phase == 'train':
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            scheduler.step()


        y_batch = torch.clamp(y_batch, min=0, max=2).type(torch.int64)
        n_sample = y_batch.shape[0]
        n_all += n_sample
        t_loss += batch_loss.item() * n_sample
        t_correct += torch.sum(torch.argmax(batch_pred, dim=1) == y_batch).item()


    print("{} Loss: {:.4f},  Accuarcy: {:.2f}%, {:.2f} Seconds Used:".
          format(phase, t_loss / n_all, 100 * t_correct / n_all, time() - t0))




if __name__ == "__main__":
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    
    # Configuration
    max_word = 128
    batch_size = 16
    learning_rate = 5e-5
    label_size = 3
    n_epochs = 20
    warmup_proportion = 0.1
    
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
    print("Loading Datasets (Baseline NS-NLI)")
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
    print("="*70 + "\n")
    
    # Model
    bert_model = GPT2Classifier(n_class=3).cuda()
    
    # Optimizer (FIXED)
    num_train_steps = int(len(train_iterator) / batch_size * n_epochs)
    num_warmup_steps = int(num_train_steps * warmup_proportion)
    
    optimizer = AdamW(  # FIXED: Use torch.optim.AdamW
        filter(lambda x: x.requires_grad, bert_model.parameters()), 
        lr=learning_rate, 
        eps=1e-8
    )
    
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=num_warmup_steps, 
        num_training_steps=num_train_steps
    )
    
    # Training loop
    print('Start Training (Baseline NS-NLI)...')
    for i in range(n_epochs):
        print(f'\nEpoch {i}...')
        run_epoch(bert_model, train_iterator, optimizer, scheduler, phase='train', batch_size=batch_size)
        torch.save(bert_model.state_dict(), f'./checkpoints/saved_model_chunk_gpt2_fixrev_{i}.pt')
        
        with torch.no_grad():
            run_epoch(bert_model, dev_iterator, optimizer, scheduler, phase='dev', batch_size=batch_size)
            run_epoch(bert_model, test_iterator, optimizer, scheduler, phase='test', batch_size=batch_size)
        print('')