#!/usr/bin/env python3
"""Fix chapter and section references in LaTeX files."""

import re
import os

# Mapping of old chapter numbers to new chapter numbers based on actual content
# Old numbering -> New numbering
chapter_mapping = {
    # Based on the original document structure
    '1': '1-3',    # Chapter 1 in text may refer to Introduction, Axiom, or Derivation
    '2': '4',      # Encoding chapter
    '3': '5',      # Quantum chapter  
    '4': '6',      # Riemann chapter
    '5': '7',      # Applications chapter
    '6': '8',      # Conclusion chapter
    '7': '9',      # Defense chapter
    '8': '10',     # Completeness chapter
    '9': '11',     # Philosophy chapter
}

# Section mapping for specific known references
section_mapping = {
    '9.6': '11.6',  # Section 9.6 -> Section 11.6
    '9.2': '11.2',  # Section 9.2 -> Section 11.2
    '9.8': '11.8',  # Section 9.8 -> Section 11.8
}

def fix_references(content):
    """Fix chapter and section references in content."""
    
    # First, let's identify which file this is to understand context
    lines = content.split('\n')
    modified = False
    
    for i, line in enumerate(lines):
        original_line = line
        
        # Fix section references (e.g., "Section 9.6" -> "Section 11.6")
        for old_sec, new_sec in section_mapping.items():
            patterns = [
                f'Section {old_sec}',
                f'section {old_sec}',
                f'§{old_sec}'
            ]
            for pattern in patterns:
                if pattern in line:
                    new_pattern = pattern.replace(old_sec, new_sec)
                    line = line.replace(pattern, new_pattern)
        
        # Fix specific known chapter references based on context
        # For ch04_encoding.tex references
        if 'defined in Chapter 1' in line:
            # This likely refers to basic definitions, which are in Chapter 2
            line = line.replace('defined in Chapter 1', 'defined in Chapter 2')
        elif 'Chapter 1 has proven' in line:
            # This refers to the axiom/derivation content
            line = line.replace('Chapter 1 has proven', 'Chapters 1-3 have proven')
        elif "Chapter 1's definition" in line:
            # Refers to axiom definitions
            line = line.replace("Chapter 1's definition", "Chapter 2's definition")
        elif "Chapter 1's axiom" in line:
            # Refers to the axiom
            line = line.replace("Chapter 1's axiom", "Chapter 2's axiom")
        elif 'defined in Chapter 3' in line:
            # Observer concept is in quantum chapter
            line = line.replace('defined in Chapter 3', 'defined in Chapter 5')
        
        # Fix chapter verification references in completeness check
        if 'Chapter 1 verification' in line:
            line = 'Chapters 1-3 verification'
        elif 'Chapter 2 verification' in line:
            line = 'Chapter 4 verification'
        elif 'Chapter 3 verification' in line:
            line = 'Chapter 5 verification'
        elif 'Chapter 4 verification' in line:
            line = 'Chapter 6 verification'
        elif 'Chapter 5 verification' in line:
            line = 'Chapter 7 verification'
        
        # Fix references to Chapter 4 as Riemann/analogy
        if 'Chapter 4' in line and ('analogy' in line or 'Riemann' in line):
            line = line.replace('Chapter 4', 'Chapter 6')
        
        # Fix references to Chapter 5 as applications
        if 'Chapter 5' in line and ('application' in line or 'prediction' in line):
            line = line.replace('Chapter 5', 'Chapter 7')
        
        # Fix philosophy chapter references
        if 'Chapter 9' in line and ('philosophy' in line or 'self-examination' in line):
            line = line.replace('Chapter 9', 'Chapter 11')
        
        # Update Chapter enumeration in philosophy chapter
        if '- Chapter 1:' in line and 'core concepts' in line:
            line = line.replace('- Chapter 1:', '- Chapters 1-3:')
        elif '- Chapter 2:' in line and 'theorems' in line:
            line = line.replace('- Chapter 2:', '- Chapter 4:')
        elif '- Chapter 3:' in line and 'quantum' in line:
            line = line.replace('- Chapter 3:', '- Chapter 5:')
        
        # Fix "Chapter 9's structure"
        if "Chapter 9's structure" in line:
            line = line.replace("Chapter 9's structure", "Chapter 11's structure")
        
        # General chapter range fixes
        if 'Chapters 1-3' in line and 'rigorous derivations' in line:
            # This is likely correct as-is
            pass
        
        if line != original_line:
            lines[i] = line
            modified = True
    
    return '\n'.join(lines), modified

def process_file(filepath):
    """Process a single LaTeX file."""
    print(f"Processing {filepath}...")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    new_content, modified = fix_references(content)
    
    if modified:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"  - Fixed references in {filepath}")
    else:
        print(f"  - No references to fix in {filepath}")

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
            process_file(filename)
        else:
            print(f"Warning: {filename} not found")
    
    print("\nReference fixing complete!")
    print("\nNote: Some references may need manual review, especially:")
    print("- References to 'Chapter 1' could mean Introduction, Axiom, or Derivation")
    print("- Complex cross-references between chapters")
    print("- References to specific theorems or definitions")

if __name__ == '__main__':
    main()