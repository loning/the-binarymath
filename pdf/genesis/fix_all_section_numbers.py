#!/usr/bin/env python3
"""Comprehensively fix all section numbering in LaTeX files."""

import re
import os

def fix_chapter_sections(filename, chapter_num):
    """Fix all section numbers in a chapter file to match the chapter number."""
    
    print(f"Processing {filename} for chapter {chapter_num}...")
    
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.split('\n')
    modified = False
    
    # Track section numbers for proper sequencing
    next_section = 1
    subsection_stack = {}
    
    for i, line in enumerate(lines):
        # Match any \section{ command
        section_match = re.match(r'^(\s*)\\section\{([^}]+)\}', line)
        if section_match:
            indent = section_match.group(1)
            section_title = section_match.group(2)
            
            # Extract existing number if present
            number_match = re.match(r'^(\d+(?:\.\d+)*)\s+(.+)$', section_title)
            
            if number_match:
                old_number = number_match.group(1)
                title_text = number_match.group(2)
                
                # Determine the new section number
                parts = old_number.split('.')
                
                # Handle subsections (e.g., 2.5.1 -> 4.5.1)
                if len(parts) > 2:
                    # This is a subsection
                    main_section = int(parts[1])
                    subsection = '.'.join(parts[2:])
                    new_number = f"{chapter_num}.{main_section}.{subsection}"
                elif len(parts) == 2:
                    # This is a regular section
                    section_num = int(parts[1])
                    new_number = f"{chapter_num}.{section_num}"
                else:
                    # Single number, shouldn't happen but handle it
                    new_number = f"{chapter_num}.{parts[0]}"
                
                # Create the new line
                new_line = f"{indent}\\section{{{new_number} {title_text}}}"
                if new_line != line:
                    lines[i] = new_line
                    modified = True
                    print(f"  - Changed: {old_number} -> {new_number}")
            else:
                # No number in section title, add one if needed
                if chapter_num == 2 and section_title == "Complete Formal Statement of the Axiom":
                    # Special case for ch02
                    new_line = f"{indent}\\section{{2.1 {section_title}}}"
                    lines[i] = new_line
                    modified = True
                    print(f"  - Added numbering: 2.1 {section_title}")
    
    # Write back if modified
    if modified:
        new_content = '\n'.join(lines)
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"  - File updated successfully")
    else:
        print(f"  - No changes needed")

def main():
    """Main function."""
    # Chapter mapping
    chapters = [
        ('ch01_introduction.tex', 1),
        ('ch02_axiom.tex', 2),
        ('ch03_derivation.tex', 3),
        ('ch04_encoding.tex', 4),
        ('ch05_quantum.tex', 5),
        ('ch06_riemann.tex', 6),
        ('ch07_applications.tex', 7),
        ('ch08_conclusion.tex', 8),
        ('ch09_defense.tex', 9),
        ('ch10_completeness.tex', 10),
        ('ch11_philosophy.tex', 11),
    ]
    
    # Change to the genesis directory
    genesis_dir = '/Users/cookie/the-binarymath/pdf/genesis'
    os.chdir(genesis_dir)
    
    # Process each file
    for filename, chapter_num in chapters:
        if os.path.exists(filename):
            fix_chapter_sections(filename, chapter_num)
        else:
            print(f"Warning: {filename} not found")
    
    print("\nAll section numbering fixed!")

if __name__ == '__main__':
    main()