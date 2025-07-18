#!/usr/bin/env python3
"""Fix chapter and section numbering in LaTeX files."""

import re
import os

# Define the correct chapter mapping
# Key: filename, Value: (actual_chapter_number, old_section_prefix)
chapter_mapping = {
    'ch01_introduction.tex': (1, None),  # No sections in this file
    'ch02_axiom.tex': (2, '1.1'),      # Sections are 1.1.x, should be 2.x
    'ch03_derivation.tex': (3, '1'),    # Sections are 1.x, should be 3.x
    'ch04_encoding.tex': (4, '2'),      # Sections are 2.x, should be 4.x
    'ch05_quantum.tex': (5, '3'),       # Sections are 3.x, should be 5.x
    'ch06_riemann.tex': (6, '4'),       # Sections are 4.x, should be 6.x
    'ch07_applications.tex': (7, '5'),  # Sections are 5.x, should be 7.x
    'ch08_conclusion.tex': (8, '6'),    # Sections are 6.x, should be 8.x
    'ch09_defense.tex': (9, '7'),       # Sections are 7.x, should be 9.x
    'ch10_completeness.tex': (10, '8'), # Sections are 8.x, should be 10.x
    'ch11_philosophy.tex': (11, '9'),   # Sections are 9.x, should be 11.x
}

def fix_section_numbering(content, chapter_num, old_prefix):
    """Fix section numbering in content."""
    if old_prefix is None:
        return content
    
    # Handle special case for ch02_axiom.tex with three-level numbering
    if old_prefix == '1.1':
        # First, add numbering to the unnumbered section
        content = re.sub(r'\\section\{Complete Formal Statement of the Axiom\}',
                        r'\\section{2.1 Complete Formal Statement of the Axiom}',
                        content)
        
        # Then fix the 1.1.x numbering to 2.x
        for i in range(1, 9):  # 1.1.1 through 1.1.8
            old_pattern = rf'\\section\{{1\.1\.{i} '
            new_pattern = rf'\\section{{{chapter_num}.{i+1} '  # 2.2 through 2.9
            content = re.sub(old_pattern, new_pattern, content)
    else:
        # Standard case: replace old_prefix.x with chapter_num.x
        pattern = rf'\\section\{{{old_prefix}\.(\d+)'
        
        # Special handling for ch04_encoding.tex duplicate 2.6
        if chapter_num == 4 and old_prefix == '2':
            # First occurrence of 2.6 stays as 4.6
            # Second occurrence of 2.6 becomes 4.7
            # And subsequent sections get incremented
            lines = content.split('\n')
            section_26_count = 0
            for i, line in enumerate(lines):
                if '\\section{2.6 ' in line:
                    section_26_count += 1
                    if section_26_count == 1:
                        lines[i] = line.replace('\\section{2.6 ', '\\section{4.6 ')
                    else:
                        lines[i] = line.replace('\\section{2.6 ', '\\section{4.7 ')
                elif '\\section{2.7' in line:
                    lines[i] = line.replace('\\section{2.7', '\\section{4.8')
                elif '\\section{2.8' in line:
                    lines[i] = line.replace('\\section{2.8', '\\section{4.9')
                elif re.match(rf'\\s*\\section\{{2\.(\d+)', line):
                    # Handle other sections normally
                    lines[i] = re.sub(rf'\\section\{{2\.(\d+)', 
                                     lambda m: f'\\section{{{chapter_num}.{m.group(1)}',
                                     line)
            content = '\n'.join(lines)
        else:
            # Normal replacement
            def replace_func(match):
                section_num = match.group(1)
                return f'\\section{{{chapter_num}.{section_num}'
            
            content = re.sub(pattern, replace_func, content)
    
    return content

def process_file(filepath, chapter_num, old_prefix):
    """Process a single LaTeX file."""
    print(f"Processing {filepath}...")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix section numbering
    new_content = fix_section_numbering(content, chapter_num, old_prefix)
    
    # Write back if changes were made
    if new_content != content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"  - Fixed section numbering for chapter {chapter_num}")
    else:
        print(f"  - No changes needed")

def main():
    """Main function."""
    # Change to the genesis directory
    genesis_dir = '/Users/cookie/the-binarymath/pdf/genesis'
    os.chdir(genesis_dir)
    
    # Process each chapter file
    for filename, (chapter_num, old_prefix) in chapter_mapping.items():
        if os.path.exists(filename):
            process_file(filename, chapter_num, old_prefix)
        else:
            print(f"Warning: {filename} not found")
    
    print("\nSection numbering fix complete!")

if __name__ == '__main__':
    main()