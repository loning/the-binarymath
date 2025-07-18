#!/usr/bin/env python3
"""Fix philosophical propositions numbering and other special cases."""

import re
import os

def fix_philosophical_numbering(content, chapter_num):
    """Fix philosophical propositions to match chapter number."""
    
    lines = content.split('\n')
    prop_counter = 0
    modified = False
    
    for i, line in enumerate(lines):
        original_line = line
        
        # Fix Philosophical Propositions
        prop_match = re.search(r'\\textbf\{Philosophical Proposition (\d+\.\d+)', line)
        if prop_match:
            old_num = prop_match.group(1)
            prop_counter += 1
            new_num = f"{chapter_num}.{prop_counter}"
            line = line.replace(f"Philosophical Proposition {old_num}", f"Philosophical Proposition {new_num}")
        
        # Fix "Observation X.Y" that should use chapter number
        obs_match = re.search(r'\\textbf\{Observation (\d+\.\d+)', line)
        if obs_match and chapter_num == 9:  # ch09_defense.tex
            old_num = obs_match.group(1)
            # Extract the Y part
            parts = old_num.split('.')
            if parts[0] != str(chapter_num):
                counter = int(parts[1]) if len(parts) > 1 else 1
                new_num = f"{chapter_num}.{counter}"
                line = line.replace(f"Observation {old_num}", f"Observation {new_num}")
        
        if line != original_line:
            lines[i] = line
            modified = True
    
    return '\n'.join(lines), modified

def process_file(filepath, chapter_num):
    """Process a single file."""
    print(f"Processing {filepath}...")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    new_content, modified = fix_philosophical_numbering(content, chapter_num)
    
    if modified:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"  - Fixed philosophical propositions/observations")
    else:
        print(f"  - No changes needed")

def main():
    """Main function."""
    genesis_dir = '/Users/cookie/the-binarymath/pdf/genesis'
    os.chdir(genesis_dir)
    
    # Process ch11_philosophy.tex
    if os.path.exists('ch11_philosophy.tex'):
        process_file('ch11_philosophy.tex', 11)
    
    # Process ch09_defense.tex for observations
    if os.path.exists('ch09_defense.tex'):
        process_file('ch09_defense.tex', 9)
    
    print("\nPhilosophical propositions numbering fix complete!")

if __name__ == '__main__':
    main()