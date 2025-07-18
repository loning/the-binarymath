#!/usr/bin/env python3
"""
Fix LaTeX escape issues in tex files
"""
import os
import re
import sys

def fix_tex_file(filepath):
    """Fix escaped LaTeX commands in a tex file"""
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Fix \textbackslash{} issues
    content = content.replace('\\textbackslash{}', '\\')
    
    # Fix escaped braces
    content = content.replace('\\{', '{')
    content = content.replace('\\}', '}')
    
    # Fix escaped underscores in math mode
    content = re.sub(r'\\_', '_', content)
    
    # Fix double escapes
    content = content.replace('\\\\\\\\', '\\\\')
    
    # Fix specific patterns
    replacements = [
        ('\\section{', '\\section{'),
        ('\\begin{', '\\begin{'),
        ('\\end{', '\\end{'),
        ('\\text{', '\\text{'),
        ('\\equiv', '\\equiv'),
        ('\\exists', '\\exists'),
        ('\\forall', '\\forall'),
        ('\\to', '\\to'),
        ('\\in', '\\in'),
        ('\\mathcal{', '\\mathcal{'),
        ('\\mathbb{', '\\mathbb{'),
        ('\\Rightarrow', '\\Rightarrow'),
        ('\\cup', '\\cup'),
        ('\\Phi', '\\Phi'),
        ('\\log', '\\log'),
        ('\\item', '\\item'),
        ('\\subseteq', '\\subseteq'),
        ('\\neq', '\\neq'),
        ('\\land', '\\land'),
    ]
    
    # Apply replacements
    for old, new in replacements:
        if old != new:  # Only if they're different
            content = content.replace(old, new)
    
    # Write back if changed
    if content != original_content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Fixed: {filepath}")
        return True
    else:
        print(f"No changes needed: {filepath}")
        return False

def main():
    # Directory containing tex files
    tex_dir = '/Users/cookie/the-binarymath/pdf/genesis'
    
    # Process all .tex files
    fixed_count = 0
    for filename in os.listdir(tex_dir):
        if filename.endswith('.tex'):
            filepath = os.path.join(tex_dir, filename)
            if fix_tex_file(filepath):
                fixed_count += 1
    
    print(f"\nTotal files fixed: {fixed_count}")

if __name__ == "__main__":
    main()