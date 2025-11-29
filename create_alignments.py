"""
Create dummy alignment files for SNLI
(Real alignments would use word alignment tools)
"""

import json
import pickle
import os

def create_dummy_alignment(num_samples):
    """
    Create dummy word alignments.
    Real version would use tools like fast_align or GIZA++.
    """
    alignments = []
    for i in range(num_samples):
        # Empty alignment (will be filled by chunking heuristics)
        alignments.append({
            'sureAlign': '',  # e.g., "0-0 1-2 3-3"
            'possibleAlign': ''
        })
    return alignments

def main():
    snli_dir = '/mmfs1/project/mx6/sp3463/NS-NSLI/Modified/data/snli'
    
    for split in ['train', 'dev', 'test']:
        print(f"Creating alignment for {split}...")
        
        # Load preprocessed samples to get count
        with open(f'{snli_dir}/{split}_records.pkl', 'rb') as f:
            samples = pickle.load(f)
        
        # Create alignments
        alignments = create_dummy_alignment(len(samples))
        
        # Save
        output_file = f'{snli_dir}/aligned_snli_{split}.json'
        with open(output_file, 'w') as f:
            json.dump(alignments, f)
        
        print(f"? Created {output_file} with {len(alignments)} alignments")

if __name__ == '__main__':
    main()