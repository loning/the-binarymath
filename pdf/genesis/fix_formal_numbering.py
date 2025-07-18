#!/usr/bin/env python3
"""Fix formal numbering in LaTeX files to match chapter numbers."""

import re
import os

# Mapping of chapter numbers
chapter_map = {
    'ch01_introduction.tex': 1,
    'ch02_axiom.tex': 2,
    'ch03_derivation.tex': 3,
    'ch04_encoding.tex': 4,
    'ch05_quantum.tex': 5,
    'ch06_riemann.tex': 6,
    'ch07_applications.tex': 7,
    'ch08_conclusion.tex': 8,
    'ch09_defense.tex': 9,
    'ch10_completeness.tex': 10,
    'ch11_philosophy.tex': 11,
}

def fix_formal_numbering(content, chapter_num):
    """Fix theorem, definition, lemma numbering to match chapter."""
    
    lines = content.split('\n')
    
    # Track counters for each type
    counters = {
        'theorem': 0,
        'definition': 0,
        'lemma': 0,
        'corollary': 0,
        'proposition': 0,
        'observation': 0,
        'axiom': 0,
    }
    
    # Patterns to match formal statements
    patterns = [
        # Pattern 1: \textbf{Theorem X.Y ...}
        (r'\\textbf\{(Theorem|Definition|Lemma|Corollary|Proposition|Observation|Axiom)\s+(\d+(?:\.\d+)?)[^}]*\}', 'textbf'),
        # Pattern 2: **Theorem X.Y ...**
        (r'\*\*(Theorem|Definition|Lemma|Corollary|Proposition|Observation|Axiom)\s+(\d+(?:\.\d+)?)[^*]*\*\*', 'stars'),
        # Pattern 3: Theorem X.Y:
        (r'^(\s*)(Theorem|Definition|Lemma|Corollary|Proposition|Observation|Axiom)\s+(\d+(?:\.\d+)?)\s*:', 'plain'),
    ]
    
    modified = False
    
    for i, line in enumerate(lines):
        original_line = line
        
        for pattern, style in patterns:
            matches = list(re.finditer(pattern, line))
            
            for match in reversed(matches):  # Process from right to left to preserve positions
                stmt_type = match.group(1).lower() if style == 'textbf' or style == 'stars' else match.group(2).lower()
                old_num = match.group(2) if style == 'textbf' or style == 'stars' else match.group(3)
                
                # For items that already have chapter prefix correct, skip
                if '.' in old_num:
                    prefix = old_num.split('.')[0]
                    if prefix == str(chapter_num):
                        continue
                
                # Increment counter
                counters[stmt_type] += 1
                new_num = f"{chapter_num}.{counters[stmt_type]}"
                
                # Reconstruct the match with new number
                if style == 'textbf':
                    # Extract the full content including parenthetical description
                    full_match = match.group(0)
                    # Find where the number ends
                    num_end = match.start(2) - match.start(0) + len(old_num)
                    before_num = full_match[:match.start(2) - match.start(0)]
                    after_num = full_match[num_end:]
                    replacement = f"{before_num}{new_num}{after_num}"
                elif style == 'stars':
                    full_match = match.group(0)
                    num_end = match.start(2) - match.start(0) + len(old_num)
                    before_num = full_match[:match.start(2) - match.start(0)]
                    after_num = full_match[num_end:]
                    replacement = f"{before_num}{new_num}{after_num}"
                else:  # plain
                    indent = match.group(1)
                    stmt_type_cap = match.group(2)
                    replacement = f"{indent}{stmt_type_cap} {new_num}:"
                
                # Replace in line
                line = line[:match.start()] + replacement + line[match.end():]
        
        # Special handling for references to old numbering
        # Fix references like "Definition 1.5" that should be "Definition 3.X"
        # This is context-specific and needs careful handling
        
        # Fix specific known references
        if chapter_num == 4:  # ch04_encoding.tex
            line = re.sub(r'Definition 1\.5', 'Definition 3.5', line)
            line = re.sub(r'Lemma 1\.4\.1', 'Lemma 3.4', line)
            line = re.sub(r'Lemma 1\.3', 'Lemma 3.3', line)
        elif chapter_num == 8:  # ch08_conclusion.tex
            line = re.sub(r'Lemma 1\.3', 'Lemma 3.3', line)
        elif chapter_num == 11:  # ch11_philosophy.tex
            line = re.sub(r'Definition 1\.1', 'Definition 2.1', line)
        
        if line != original_line:
            lines[i] = line
            modified = True
    
    return '\n'.join(lines), modified, counters

def process_file(filepath):
    """Process a single LaTeX file."""
    filename = os.path.basename(filepath)
    if filename not in chapter_map:
        print(f"Skipping {filename} - not in chapter map")
        return
    
    chapter_num = chapter_map[filename]
    print(f"\nProcessing {filename} (Chapter {chapter_num})...")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    new_content, modified, counters = fix_formal_numbering(content, chapter_num)
    
    if modified:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print(f"  Fixed formal numbering:")
        for stmt_type, count in counters.items():
            if count > 0:
                print(f"    - {count} {stmt_type}(s)")
    else:
        print(f"  No changes needed")

def main():
    """Main function."""
    genesis_dir = '/Users/cookie/the-binarymath/pdf/genesis'
    os.chdir(genesis_dir)
    
    # Process all chapter files
    for filename in sorted(chapter_map.keys()):
        if os.path.exists(filename):
            process_file(filename)
        else:
            print(f"Warning: {filename} not found")
    
    print("\nFormal numbering fix complete!")
    print("\nNote: Some cross-references may need manual review.")

if __name__ == '__main__':
    main()