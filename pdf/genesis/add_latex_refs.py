#!/usr/bin/env python3
"""Add LaTeX labels and convert text references to \ref commands."""

import re
import os

def add_labels_and_refs(content, filename):
    """Add labels to sections, theorems, etc. and convert references to \ref."""
    
    lines = content.split('\n')
    modified = False
    
    # Extract chapter name from filename (e.g., ch04_encoding.tex -> ch04)
    chapter_prefix = filename.replace('.tex', '')
    
    for i, line in enumerate(lines):
        original_line = line
        
        # Add labels to sections
        section_match = re.match(r'^(\\section\{)([^}]+)(\})$', line)
        if section_match and i + 1 < len(lines) and not lines[i + 1].startswith('\\label{'):
            section_title = section_match.group(2)
            # Create a label based on section title
            label_name = re.sub(r'[^a-zA-Z0-9]+', '-', section_title.lower()).strip('-')
            label_name = f"sec:{chapter_prefix}:{label_name}"
            # Add the label on the next line
            lines.insert(i + 1, f"\\label{{{label_name}}}")
            modified = True
            continue
        
        # Add labels to theorems, definitions, etc.
        theorem_patterns = [
            (r'\\textbf\{(Theorem|Definition|Lemma|Corollary|Proposition|Observation) (\d+\.\d+)[^}]*\}', 'thm'),
            (r'\\textbf\{(Axiom) (\d+\.\d+)[^}]*\}', 'axiom'),
        ]
        
        for pattern, prefix in theorem_patterns:
            theorem_match = re.search(pattern, line)
            if theorem_match:
                theorem_type = theorem_match.group(1).lower()
                theorem_num = theorem_match.group(2)
                label_name = f"{prefix}:{theorem_num}"
                # Add label at the end of the line if not already there
                if f"\\label{{{label_name}}}" not in line and i + 1 < len(lines) and not lines[i + 1].startswith('\\label{'):
                    lines.insert(i + 1, f"\\label{{{label_name}}}")
                    modified = True
        
        # Convert text references to \ref
        # Pattern: "Chapter X" -> "Chapter~\ref{ch:X}"
        chapter_ref_pattern = r'Chapter (\d+)'
        def replace_chapter_ref(match):
            chapter_num = match.group(1)
            # Map chapter numbers to labels
            chapter_labels = {
                '1': 'ch:introduction',
                '2': 'ch:axiom',
                '3': 'ch:derivation',
                '4': 'ch:encoding',
                '5': 'ch:quantum',
                '6': 'ch:riemann',
                '7': 'ch:applications',
                '8': 'ch:conclusion',
                '9': 'ch:defense',
                '10': 'ch:completeness',
                '11': 'ch:philosophy'
            }
            if chapter_num in chapter_labels:
                return f"Chapter~\\ref{{{chapter_labels[chapter_num]}}}"
            return match.group(0)
        
        line = re.sub(chapter_ref_pattern, replace_chapter_ref, line)
        
        # Pattern: "Chapters X-Y" -> "Chapters~\ref{ch:X}--\ref{ch:Y}"
        chapters_range_pattern = r'Chapters (\d+)-(\d+)'
        def replace_chapters_range(match):
            start_num = match.group(1)
            end_num = match.group(2)
            chapter_labels = {
                '1': 'ch:introduction',
                '2': 'ch:axiom',
                '3': 'ch:derivation',
                '4': 'ch:encoding',
                '5': 'ch:quantum',
                '6': 'ch:riemann',
                '7': 'ch:applications',
                '8': 'ch:conclusion',
                '9': 'ch:defense',
                '10': 'ch:completeness',
                '11': 'ch:philosophy'
            }
            if start_num in chapter_labels and end_num in chapter_labels:
                return f"Chapters~\\ref{{{chapter_labels[start_num]}}}--\\ref{{{chapter_labels[end_num]}}}"
            return match.group(0)
        
        line = re.sub(chapters_range_pattern, replace_chapters_range, line)
        
        # Pattern: "Theorem X.Y" -> "Theorem~\ref{thm:X.Y}"
        theorem_ref_pattern = r'Theorem (\d+\.\d+)'
        def replace_theorem_ref(match):
            theorem_num = match.group(1)
            return f"Theorem~\\ref{{thm:{theorem_num}}}"
        
        line = re.sub(theorem_ref_pattern, replace_theorem_ref, line)
        
        # Pattern: "Section X.Y" -> "Section~\ref{sec:...}"
        # This is trickier as we need to map section numbers to actual section labels
        # For now, we'll leave these as-is since section labels depend on section titles
        
        if line != original_line:
            lines[i] = line
            modified = True
    
    return '\n'.join(lines), modified

def process_file(filepath):
    """Process a single LaTeX file."""
    filename = os.path.basename(filepath)
    print(f"Processing {filename}...")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    new_content, modified = add_labels_and_refs(content, filename)
    
    if modified:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"  - Added labels and references to {filename}")
    else:
        print(f"  - No changes needed in {filename}")

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
    
    print("\nLaTeX reference system implementation complete!")
    print("\nNotes:")
    print("- Chapter references now use \\ref{ch:name}")
    print("- Theorem references now use \\ref{thm:X.Y}")
    print("- Section references still need manual adjustment based on section titles")
    print("- Make sure to compile twice for references to resolve")

if __name__ == '__main__':
    main()