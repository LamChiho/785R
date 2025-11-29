"""
Stage 1: Convert raw SNLI to pickle format with monotonicity projections
"""

import json
import pickle
import os
from tqdm import tqdm

def read_snli_jsonl(file_path):
    """Read SNLI JSONL file."""
    samples = []
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            if data['gold_label'] == '-':  # Skip unlabeled
                continue
            samples.append(data)
    return samples

def add_monotonicity_projections(samples):
    """
    Add basic monotonicity projections to each sample.
    
    Simplified version - assigns default projections.
    Full version would use a monotonicity calculator.
    """
    default_projection = [0, 1, 2, 3, 4, 5, 6]  # Identity projection
    
    processed_samples = []
    for sample in tqdm(samples, desc="Adding projections"):
        # Extract premise and hypothesis
        premise = sample['sentence1']
        hypothesis = sample['sentence2']
        label = sample['gold_label']
        
        # Tokenize (simple split for now)
        premise_tokens = premise.split()
        hypothesis_tokens = hypothesis.split()
        
        # Create processed sample
        processed = {
            'sent_1_tokens': hypothesis_tokens,  # Note: hypothesis is sent_1 in NS-NLI
            'sent_2_tokens': premise_tokens,     # premise is sent_2
            'sent_1_projection': [default_projection] * len(hypothesis_tokens),
            'sent_2_projection': [default_projection] * len(premise_tokens),
            'y': label,
            'pairID': sample['pairID']
        }
        
        processed_samples.append(processed)
    
    return processed_samples

def main():
    # Paths
    snli_dir = '/mmfs1/project/mx6/sp3463/NS-NSLI/Modified/snli_1.0'
    output_dir = '/mmfs1/project/mx6/sp3463/NS-NSLI/Modified/data/snli'
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each split
    for split in ['train', 'dev', 'test']:
        print(f"\n{'='*60}")
        print(f"Processing {split} split")
        print('='*60)
        
        # Read raw SNLI
        input_file = f'{snli_dir}/snli_1.0_{split}.jsonl'
        print(f"Reading {input_file}...")
        samples = read_snli_jsonl(input_file)
        print(f"Loaded {len(samples)} samples")
        
        # Add projections
        print("Adding monotonicity projections...")
        processed = add_monotonicity_projections(samples)
        
        # Save
        output_file = f'{output_dir}/{split}_records.pkl'
        print(f"Saving to {output_file}...")
        with open(output_file, 'wb') as f:
            pickle.dump(processed, f)
        
        print(f"? Saved {len(processed)} samples")
    
    print("\n" + "="*60)
    print("Stage 1 preprocessing complete!")
    print("="*60)

if __name__ == '__main__':
    main()