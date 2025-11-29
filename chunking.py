"""
Phrase Chunking for Natural Logic using spaCy
==============================================

This module extracts noun phrases and verb phrases from sentences
using spaCy's dependency parsing.

Based on NS-NLI's preprocessing pipeline.
"""

import spacy

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model...")
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")


def chunking_sent(sentence):
    """
    Chunk sentence into phrases using spaCy.
    
    Extracts:
    - Noun phrases (subjects, objects)
    - Verb phrases
    - Prepositional phrases
    
    Args:
        sentence: str - Input sentence
    
    Returns:
        list of str - Phrase chunks
    
    Example:
        >>> chunking_sent("A dog is running in the park")
        ['A dog', 'is running', 'in the park']
    """
    doc = nlp(sentence.strip())
    
    chunks = []
    
    # Method 1: Use spaCy's noun_chunks (most common in NLI)
    # This extracts noun phrases like "A dog", "the park"
    noun_chunks = list(doc.noun_chunks)
    
    if len(noun_chunks) > 0:
        # Collect noun chunks
        noun_chunk_spans = [(nc.start, nc.end) for nc in noun_chunks]
        
        # Build chunks by combining noun phrases and filling gaps
        current_pos = 0
        for start, end in sorted(noun_chunk_spans):
            # Add tokens between previous chunk and this noun chunk (usually verbs/prepositions)
            if current_pos < start:
                between_text = doc[current_pos:start].text.strip()
                if between_text:
                    chunks.append(between_text)
            
            # Add the noun chunk
            chunks.append(doc[start:end].text.strip())
            current_pos = end
        
        # Add any remaining tokens after last noun chunk
        if current_pos < len(doc):
            remaining_text = doc[current_pos:].text.strip()
            if remaining_text:
                chunks.append(remaining_text)
    
    else:
        # Fallback: If no noun chunks, split by major syntactic boundaries
        current_chunk = []
        for token in doc:
            current_chunk.append(token.text)
            
            # Break on:
            # - Main verbs (ROOT)
            # - Prepositions (prep)
            # - Conjunctions (cc)
            if token.dep_ in ['ROOT', 'prep', 'cc'] or token.pos_ in ['VERB', 'AUX']:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
        
        # Add remaining tokens
        if current_chunk:
            chunks.append(' '.join(current_chunk))
    
    # Clean up: remove empty chunks and strip whitespace
    chunks = [c.strip() for c in chunks if c.strip()]
    
    # If no chunks extracted, return whole sentence
    if not chunks:
        chunks = [sentence.strip()]
    
    return chunks


def chunking_sent_simple(sentence):
    """
    Simplified chunking - just use spaCy's built-in noun_chunks
    (faster but less accurate for complex sentences)
    """
    doc = nlp(sentence.strip())
    chunks = [chunk.text for chunk in doc.noun_chunks]
    
    if not chunks:
        chunks = [sentence.strip()]
    
    return chunks


# For testing
if __name__ == '__main__':
    test_sentences = [
        "A dog is running in the park",
        "The cat sleeps on the mat",
        "People are walking down the street quickly",
        "No animal is moving outside"
    ]
    
    print("=" * 60)
    print("Chunking Examples")
    print("=" * 60)
    
    for sent in test_sentences:
        chunks = chunking_sent(sent)
        print(f"\nInput:  {sent}")
        print(f"Output: {chunks}")
        print(f"        {' | '.join(chunks)}")