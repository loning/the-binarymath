#!/usr/bin/env python3
"""Remove manual numbering from section titles - LaTeX will auto-number them."""

import re
import os

def remove_section_numbers(filename):
    """Remove manual numbers from section titles."""
    
    print(f"Processing {filename}...")
    
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.split('\n')
    modified = False
    
    for i, line in enumerate(lines):
        # Match any \section{ command
        section_match = re.match(r'^(\s*)\\section\{([^}]+)\}', line)
        if section_match:
            indent = section_match.group(1)
            section_title = section_match.group(2)
            
            # Remove number prefix if present (e.g., "2.1 Title" -> "Title")
            # Match patterns like "2.1 ", "2.5.1 ", etc.
            number_match = re.match(r'^\d+(?:\.\d+)*\s+(.+)$', section_title)
            
            if number_match:
                title_text = number_match.group(1)
                new_line = f"{indent}\\section{{{title_text}}}"
                lines[i] = new_line
                modified = True
                print(f"  - Removed numbering from: {section_title}")
    
    # Write back if modified
    if modified:
        new_content = '\n'.join(lines)
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"  - File updated successfully")
    else:
        print(f"  - No manual numbering found")

def main():
    """Main function."""
    # All chapter files
    chapter_files = [
        'ch01_introduction.tex',
        'ch02_axiom.tex', 
        'ch03_derivation.tex',
        'ch04_encoding.tex',
        'ch05_quantum.tex',
        'ch06_riemann.tex',
        'ch07_applications.tex',
        'ch08_conclusion.tex',
        'ch09_defense.tex',
        'ch10_completeness.tex',
        'ch11_philosophy.tex',
    ]
    
    # Change to the genesis directory
    genesis_dir = '/Users/cookie/the-binarymath/pdf/genesis'
    os.chdir(genesis_dir)
    
    # Process each file
    for filename in chapter_files:
        if os.path.exists(filename):
            remove_section_numbers(filename)
        else:
            print(f"Warning: {filename} not found")
    
    print("\nAll manual section numbering removed!")
    print("LaTeX will now auto-number all sections based on chapter numbers.")

if __name__ == '__main__':
    main()