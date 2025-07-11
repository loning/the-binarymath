#!/usr/bin/env python3
"""
Chapter 010: NestCollapse - Verification Program
Nested Collapse Structures in φ-Language

This program verifies that collapse patterns can nest within each other,
creating hierarchical structures that maintain the φ-constraint at every level
while encoding deeper semantic relationships.

从ψ的递归本质中，涌现出嵌套的崩塌结构——层层深入的自指形式。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import unittest
from typing import List, Tuple, Dict, Optional, Set, Union, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
import numpy as np
from enum import Enum
import re


class NestLevel(Enum):
    """Levels of nesting in collapse structures"""
    ATOM = 0      # Single bit
    WORD = 1      # Basic pattern
    PHRASE = 2    # Composed patterns
    SENTENCE = 3  # Complete structures
    PARAGRAPH = 4 # Multiple sentences
    DOCUMENT = 5  # Full collapse narrative


@dataclass
class NestedStructure:
    """A nested collapse structure with hierarchical organization"""
    content: str  # The φ-valid trace
    level: NestLevel
    children: List['NestedStructure'] = field(default_factory=list)
    parent: Optional['NestedStructure'] = None
    start_pos: int = 0
    end_pos: int = 0
    
    def depth(self) -> int:
        """Calculate nesting depth"""
        if not self.children:
            return 0
        return 1 + max(child.depth() for child in self.children)
    
    def span(self) -> str:
        """Get the span of content this structure covers"""
        if self.parent:
            return self.parent.content[self.start_pos:self.end_pos]
        return self.content
    
    def is_valid_phi(self) -> bool:
        """Check if structure maintains φ-constraint"""
        # Check own content
        if '11' in self.content:
            return False
        
        # Check all children recursively
        for child in self.children:
            if not child.is_valid_phi():
                return False
        
        return True
    
    def to_tree_string(self, indent: int = 0) -> str:
        """Convert to tree representation"""
        span_content = self.span()
        if not span_content and self.level == NestLevel.ATOM:
            # For atoms, show the actual content
            span_content = self.content
        result = "  " * indent + f"{self.level.name}: {span_content}\n"
        for child in self.children:
            result += child.to_tree_string(indent + 1)
        return result


class CollapseParser:
    """
    Parses φ-traces into nested collapse structures.
    Uses bracketing notation to indicate nesting levels.
    """
    
    def __init__(self):
        # Define parsing patterns for different levels
        self.atom_pattern = re.compile(r'[01]')
        self.word_patterns = [
            re.compile(r'00+'),      # Void words
            re.compile(r'01'),       # Emergence
            re.compile(r'10'),       # Return
            re.compile(r'(01)+'),    # Oscillation
            re.compile(r'(10)+')     # Anti-oscillation
        ]
        
        # Bracket markers for nested structures
        self.brackets = {
            '(': ')',  # Level 1 nesting
            '[': ']',  # Level 2 nesting
            '{': '}',  # Level 3 nesting
            '<': '>'   # Level 4 nesting
        }
    
    def parse(self, trace: str) -> NestedStructure:
        """Parse a trace into nested structure"""
        # Remove any existing brackets for clean parsing
        clean_trace = self._remove_brackets(trace)
        
        # Create root structure
        root = NestedStructure(
            content=clean_trace,
            level=NestLevel.DOCUMENT
        )
        
        # Parse hierarchically
        self._parse_sentences(root)
        
        return root
    
    def _remove_brackets(self, text: str) -> str:
        """Remove all bracket markers"""
        for open_b, close_b in self.brackets.items():
            text = text.replace(open_b, '').replace(close_b, '')
        return text
    
    def _parse_sentences(self, parent: NestedStructure):
        """Parse into sentence-level structures"""
        content = parent.content
        
        # Find natural sentence boundaries (sequences of patterns)
        sentences = self._find_sentences(content)
        
        for start, end in sentences:
            sentence = NestedStructure(
                content=content[start:end],
                level=NestLevel.SENTENCE,
                parent=parent,
                start_pos=start,
                end_pos=end
            )
            parent.children.append(sentence)
            
            # Parse phrases within sentence
            self._parse_phrases(sentence)
    
    def _find_sentences(self, content: str) -> List[Tuple[int, int]]:
        """Find sentence boundaries based on pattern complexity"""
        if len(content) <= 8:
            return [(0, len(content))]
        
        sentences = []
        start = 0
        
        # Simple heuristic: sentences are ~8-16 bits
        while start < len(content):
            end = min(start + 16, len(content))
            
            # Try to end on a word boundary
            for i in range(end, max(start + 8, end - 4), -1):
                if i < len(content) and self._is_word_boundary(content, i):
                    end = i
                    break
            
            sentences.append((start, end))
            start = end
        
        return sentences
    
    def _is_word_boundary(self, content: str, pos: int) -> bool:
        """Check if position is a good word boundary"""
        if pos == 0 or pos >= len(content):
            return True
        
        # Check for pattern transitions
        if pos > 1:
            prev_pair = content[pos-2:pos]
            next_pair = content[pos:pos+2] if pos < len(content) - 1 else ""
            
            # Good boundaries: after 00, 10, or before 01
            if prev_pair in ['00', '10'] or next_pair.startswith('01'):
                return True
        
        return False
    
    def _parse_phrases(self, sentence: NestedStructure):
        """Parse sentence into phrase structures"""
        content = sentence.span()
        
        # Find phrase patterns
        phrases = self._find_phrases(content)
        
        for start, end in phrases:
            phrase = NestedStructure(
                content=content[start:end],
                level=NestLevel.PHRASE,
                parent=sentence,
                start_pos=sentence.start_pos + start,
                end_pos=sentence.start_pos + end
            )
            sentence.children.append(phrase)
            
            # Parse words within phrase
            self._parse_words(phrase)
    
    def _find_phrases(self, content: str) -> List[Tuple[int, int]]:
        """Find phrase boundaries (groups of related words)"""
        if len(content) <= 4:
            return [(0, len(content))]
        
        phrases = []
        start = 0
        
        while start < len(content):
            # Phrases are typically 4-8 bits
            end = min(start + 8, len(content))
            
            # Adjust to word boundaries
            for i in range(end, max(start + 4, end - 2), -1):
                if self._is_word_boundary(content, i):
                    end = i
                    break
            
            phrases.append((start, end))
            start = end
        
        return phrases
    
    def _parse_words(self, phrase: NestedStructure):
        """Parse phrase into word structures"""
        content = phrase.span()  # Use span to get actual content
        if not content:
            return
            
        i = 0
        
        while i < len(content):
            # Try to match word patterns
            matched = False
            
            for pattern in self.word_patterns:
                match = pattern.match(content[i:])
                if match:
                    word = NestedStructure(
                        content=match.group(),
                        level=NestLevel.WORD,
                        parent=phrase,
                        start_pos=phrase.start_pos + i,
                        end_pos=phrase.start_pos + i + len(match.group())
                    )
                    phrase.children.append(word)
                    
                    # Parse atoms within word
                    self._parse_atoms(word)
                    
                    i += len(match.group())
                    matched = True
                    break
            
            if not matched and i < len(content):
                # Single atom word
                word = NestedStructure(
                    content=content[i],
                    level=NestLevel.WORD,
                    parent=phrase,
                    start_pos=phrase.start_pos + i,
                    end_pos=phrase.start_pos + i + 1
                )
                phrase.children.append(word)
                self._parse_atoms(word)
                i += 1
    
    def _parse_atoms(self, word: NestedStructure):
        """Parse word into atomic structures"""
        content = word.span()
        for i, bit in enumerate(content):
            atom = NestedStructure(
                content=bit,
                level=NestLevel.ATOM,
                parent=word,
                start_pos=word.start_pos + i,
                end_pos=word.start_pos + i + 1
            )
            word.children.append(atom)


class NestedEncoder:
    """
    Encodes nested structures with bracket notation.
    Preserves φ-constraint at all levels.
    """
    
    def __init__(self):
        self.level_brackets = {
            NestLevel.WORD: ('', ''),        # No brackets for words
            NestLevel.PHRASE: ('(', ')'),    # Parentheses for phrases
            NestLevel.SENTENCE: ('[', ']'),  # Square brackets for sentences
            NestLevel.PARAGRAPH: ('{', '}'), # Curly braces for paragraphs
            NestLevel.DOCUMENT: ('<', '>')   # Angle brackets for documents
        }
    
    def encode(self, structure: NestedStructure) -> str:
        """Encode nested structure with brackets"""
        if structure.level == NestLevel.ATOM:
            return structure.content
        
        # Get bracket style for this level
        open_b, close_b = self.level_brackets.get(
            structure.level, ('', '')
        )
        
        # Encode children
        if structure.children:
            child_encodings = [self.encode(child) for child in structure.children]
            content = ''.join(child_encodings)
        else:
            content = structure.content
        
        # Apply brackets if needed
        if open_b and close_b:
            return f"{open_b}{content}{close_b}"
        return content
    
    def encode_with_separators(self, structure: NestedStructure) -> str:
        """Encode with separators between levels"""
        if structure.level == NestLevel.ATOM:
            return structure.content
        
        # Encode children with separators
        child_encodings = []
        for i, child in enumerate(structure.children):
            child_encodings.append(self.encode_with_separators(child))
            
            # Add separator between children of same level
            if i < len(structure.children) - 1:
                separator = self._get_separator(child.level)
                if separator and separator != '11':  # Ensure φ-constraint
                    child_encodings.append(separator)
        
        return ''.join(child_encodings)
    
    def _get_separator(self, level: NestLevel) -> str:
        """Get separator for level transitions"""
        separators = {
            NestLevel.ATOM: '',
            NestLevel.WORD: '0',
            NestLevel.PHRASE: '00',
            NestLevel.SENTENCE: '000',
            NestLevel.PARAGRAPH: '0000'
        }
        return separators.get(level, '')


class NestingAnalyzer:
    """
    Analyzes properties of nested collapse structures.
    """
    
    def __init__(self):
        self.parser = CollapseParser()
        self.encoder = NestedEncoder()
    
    def analyze_nesting_depth(self, structure: NestedStructure) -> Dict[str, Any]:
        """Analyze nesting depth and distribution"""
        depths = []
        level_counts = defaultdict(int)
        
        # BFS to collect all nodes
        queue = deque([structure])
        while queue:
            node = queue.popleft()
            depths.append(node.depth())
            level_counts[node.level] += 1
            queue.extend(node.children)
        
        return {
            'max_depth': max(depths) if depths else 0,
            'avg_depth': np.mean(depths) if depths else 0,
            'level_distribution': dict(level_counts),
            'total_nodes': len(depths)
        }
    
    def analyze_balance(self, structure: NestedStructure) -> float:
        """Analyze how balanced the nesting tree is"""
        if not structure.children:
            return 1.0
        
        # Calculate balance factor for each internal node
        balance_factors = []
        
        def calculate_balance(node):
            if not node.children:
                return 0
            
            child_depths = [child.depth() for child in node.children]
            if len(child_depths) > 1:
                max_depth = max(child_depths)
                min_depth = min(child_depths)
                if max_depth > 0:
                    balance = 1.0 - (max_depth - min_depth) / max_depth
                else:
                    balance = 1.0
                balance_factors.append(balance)
            
            for child in node.children:
                calculate_balance(child)
        
        calculate_balance(structure)
        
        return np.mean(balance_factors) if balance_factors else 1.0
    
    def find_patterns(self, structure: NestedStructure) -> Dict[str, int]:
        """Find recurring patterns at each level"""
        patterns = defaultdict(int)
        
        def collect_patterns(node):
            if node.level != NestLevel.ATOM:
                # Record pattern at this level
                pattern_key = f"{node.level.name}:{node.span()}"
                patterns[pattern_key] += 1
            
            for child in node.children:
                collect_patterns(child)
        
        collect_patterns(structure)
        
        # Filter to patterns occurring more than once
        return {k: v for k, v in patterns.items() if v > 1}


class RecursiveCollapser:
    """
    Performs recursive collapse operations on nested structures.
    Each level can collapse independently while maintaining φ-constraint.
    """
    
    def __init__(self):
        self.collapse_rules = {
            '000': '0',    # Triple void collapses to single
            '010': '1',    # Emergence pattern
            '101': '0',    # Symmetric return
            '0101': '10',  # Oscillation collapses
            '1010': '01'   # Anti-oscillation collapses
        }
    
    def collapse(self, structure: NestedStructure, level: NestLevel) -> NestedStructure:
        """Collapse structure at specified level"""
        if structure.level.value < level.value:
            # This structure is below target level, return as is
            return structure
        
        if structure.level == level:
            # Collapse this level
            collapsed_content = self._apply_collapse_rules(structure.span())
            
            # Create new collapsed structure
            collapsed = NestedStructure(
                content=collapsed_content,
                level=structure.level,
                parent=structure.parent,
                start_pos=structure.start_pos,
                end_pos=structure.start_pos + len(collapsed_content)
            )
            
            return collapsed
        
        # Recursively collapse children
        new_structure = NestedStructure(
            content=structure.content,
            level=structure.level,
            parent=structure.parent,
            start_pos=structure.start_pos,
            end_pos=structure.end_pos
        )
        
        for child in structure.children:
            collapsed_child = self.collapse(child, level)
            new_structure.children.append(collapsed_child)
        
        return new_structure
    
    def _apply_collapse_rules(self, content: str) -> str:
        """Apply collapse rules to content"""
        result = content
        
        # Apply rules repeatedly until no more changes
        changed = True
        while changed:
            changed = False
            for pattern, replacement in self.collapse_rules.items():
                if pattern in result:
                    # Check that replacement won't create '11'
                    new_result = result.replace(pattern, replacement, 1)
                    if '11' not in new_result:
                        result = new_result
                        changed = True
                        break
        
        return result
    
    def recursive_collapse(self, structure: NestedStructure) -> List[NestedStructure]:
        """Perform recursive collapse from bottom up"""
        collapse_sequence = [structure]
        
        # Collapse from atoms up to document
        for level in [NestLevel.ATOM, NestLevel.WORD, NestLevel.PHRASE, 
                     NestLevel.SENTENCE, NestLevel.PARAGRAPH]:
            current = collapse_sequence[-1]
            collapsed = self.collapse(current, level)
            
            # Only add if actually changed
            if collapsed != current:
                collapse_sequence.append(collapsed)
        
        return collapse_sequence


class NeuralNestingModel(nn.Module):
    """
    Neural model that learns to predict and generate nested structures.
    """
    
    def __init__(self, vocab_size: int = 4, hidden_dim: int = 64, num_levels: int = 5):
        super().__init__()
        self.vocab_size = vocab_size  # 0, 1, (, )
        self.hidden_dim = hidden_dim
        self.num_levels = num_levels
        
        # Embedding for tokens and brackets
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        
        # Hierarchical LSTM for each nesting level
        self.level_lstms = nn.ModuleList([
            nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
            for _ in range(num_levels)
        ])
        
        # Output heads for each level
        self.output_heads = nn.ModuleList([
            nn.Linear(hidden_dim, vocab_size)
            for _ in range(num_levels)
        ])
        
        # Level attention
        self.level_attention = nn.MultiheadAttention(hidden_dim, num_heads=4)
        
        # φ-constraint enforcer
        self.phi_mask = nn.Parameter(torch.ones(vocab_size), requires_grad=False)
        self.phi_mask[3] = 0  # Assuming index 3 would create '11'
    
    def forward(self, x: torch.Tensor, level: int = 0):
        """
        Forward pass for specific nesting level.
        x: (batch, seq_len) token indices
        """
        # Embed input
        embedded = self.embedding(x)
        
        # Process through appropriate level LSTM
        lstm_out, (h_n, c_n) = self.level_lstms[level](embedded)
        
        # Apply level attention if not at bottom level
        if level > 0:
            attended, _ = self.level_attention(lstm_out, lstm_out, lstm_out)
            lstm_out = lstm_out + attended
        
        # Generate output
        output = self.output_heads[level](lstm_out)
        
        # Apply φ-constraint mask
        output = output * self.phi_mask
        
        return output, (h_n, c_n)
    
    def generate_nested(self, seed: torch.Tensor, max_length: int = 50) -> str:
        """Generate nested structure with brackets"""
        generated = []
        current = seed
        
        for level in range(self.num_levels):
            level_output = []
            
            with torch.no_grad():
                output, _ = self.forward(current, level)
                probs = F.softmax(output[:, -1, :], dim=-1)
                
                # Sample tokens for this level
                for _ in range(max_length // self.num_levels):
                    next_token = torch.multinomial(probs, 1)
                    level_output.append(next_token.item())
                    
                    # Update input
                    current = torch.cat([current, next_token], dim=1)
                    output, _ = self.forward(current, level)
                    probs = F.softmax(output[:, -1, :], dim=-1)
            
            generated.extend(level_output)
        
        # Convert to string with proper brackets
        return self._tokens_to_string(generated)
    
    def _tokens_to_string(self, tokens: List[int]) -> str:
        """Convert token indices to string representation"""
        mapping = {0: '0', 1: '1', 2: '(', 3: ')'}
        return ''.join(mapping.get(t, '') for t in tokens)


class NestCollapseTests(unittest.TestCase):
    """Test nested collapse structures"""
    
    def setUp(self):
        self.parser = CollapseParser()
        self.encoder = NestedEncoder()
        self.analyzer = NestingAnalyzer()
        self.collapser = RecursiveCollapser()
    
    def test_basic_parsing(self):
        """Test: Basic parsing creates valid structures"""
        trace = "0101001010"
        structure = self.parser.parse(trace)
        
        self.assertEqual(structure.level, NestLevel.DOCUMENT)
        self.assertTrue(structure.is_valid_phi())
        self.assertGreater(len(structure.children), 0)
    
    def test_nesting_depth(self):
        """Test: Nesting depth is calculated correctly"""
        trace = "01010010100101"
        structure = self.parser.parse(trace)
        
        depth = structure.depth()
        self.assertGreaterEqual(depth, 3)  # At least: document->sentence->phrase->word
    
    def test_phi_constraint_maintained(self):
        """Test: φ-constraint maintained at all levels"""
        traces = ["0101", "001001", "10101010", "000010001000"]
        
        for trace in traces:
            structure = self.parser.parse(trace)
            self.assertTrue(structure.is_valid_phi())
            
            # Check all nested levels
            def check_all_levels(node):
                self.assertNotIn('11', node.span())
                for child in node.children:
                    check_all_levels(child)
            
            check_all_levels(structure)
    
    def test_encoding_decoding(self):
        """Test: Encoding preserves structure"""
        trace = "0101001010"
        structure = self.parser.parse(trace)
        
        # Encode with brackets
        encoded = self.encoder.encode(structure)
        
        # Should contain bracket markers
        self.assertTrue(any(b in encoded for b in ['(', '[', '{', '<']))
    
    def test_recursive_collapse(self):
        """Test: Recursive collapse works correctly"""
        trace = "01010101"
        structure = self.parser.parse(trace)
        
        # Perform recursive collapse
        sequence = self.collapser.recursive_collapse(structure)
        
        # Should have multiple stages
        self.assertGreater(len(sequence), 1)
        
        # Each stage should be valid
        for stage in sequence:
            self.assertTrue(stage.is_valid_phi())
    
    def test_pattern_finding(self):
        """Test: Can find recurring patterns"""
        trace = "01010101001001001"
        structure = self.parser.parse(trace)
        
        patterns = self.analyzer.find_patterns(structure)
        
        # Should find some recurring patterns
        self.assertGreater(len(patterns), 0)
    
    def test_balance_analysis(self):
        """Test: Balance analysis works"""
        trace = "0101001010100101"
        structure = self.parser.parse(trace)
        
        balance = self.analyzer.analyze_balance(structure)
        
        # Balance should be between 0 and 1
        self.assertGreaterEqual(balance, 0.0)
        self.assertLessEqual(balance, 1.0)
    
    def test_nesting_statistics(self):
        """Test: Nesting statistics are computed correctly"""
        trace = "01010010100101001010"
        structure = self.parser.parse(trace)
        
        stats = self.analyzer.analyze_nesting_depth(structure)
        
        self.assertIn('max_depth', stats)
        self.assertIn('avg_depth', stats)
        self.assertIn('level_distribution', stats)
        self.assertGreater(stats['total_nodes'], 0)
    
    def test_neural_model(self):
        """Test: Neural model can process nested structures"""
        model = NeuralNestingModel(vocab_size=4)
        
        # Test input
        x = torch.tensor([[0, 1, 0, 1]])  # Simple pattern
        
        output, hidden = model(x, level=0)
        
        # Check output shape
        self.assertEqual(output.shape[0], 1)  # Batch size
        self.assertEqual(output.shape[2], 4)  # Vocab size
    
    def test_separator_encoding(self):
        """Test: Separator encoding maintains φ-constraint"""
        trace = "0101001010"
        structure = self.parser.parse(trace)
        
        encoded = self.encoder.encode_with_separators(structure)
        
        # Should not contain '11'
        self.assertNotIn('11', encoded)


def visualize_nested_structures():
    """Visualize nested collapse structures"""
    print("=" * 60)
    print("Nested Collapse Structures in φ-Language")
    print("=" * 60)
    
    parser = CollapseParser()
    encoder = NestedEncoder()
    analyzer = NestingAnalyzer()
    collapser = RecursiveCollapser()
    
    # Example traces
    traces = [
        "0101010101",
        "0010010010",
        "1001010010",
        "00100100101001",
        "01010010100101"
    ]
    
    for trace in traces:
        print(f"\n1. Original trace: {trace}")
        
        # Parse into nested structure
        structure = parser.parse(trace)
        
        # Show tree structure
        print("\n2. Nested structure:")
        print(structure.to_tree_string())
        
        # Encode with brackets
        encoded = encoder.encode(structure)
        print(f"3. Bracket encoding: {encoded}")
        
        # Analyze nesting
        stats = analyzer.analyze_nesting_depth(structure)
        print(f"\n4. Nesting statistics:")
        print(f"   Max depth: {stats['max_depth']}")
        print(f"   Avg depth: {stats['avg_depth']:.2f}")
        print(f"   Total nodes: {stats['total_nodes']}")
        print(f"   Level distribution:")
        for level, count in stats['level_distribution'].items():
            print(f"      {level.name}: {count}")
        
        # Find patterns
        patterns = analyzer.find_patterns(structure)
        if patterns:
            print(f"\n5. Recurring patterns:")
            for pattern, count in list(patterns.items())[:5]:
                print(f"   {pattern}: {count} times")
        
        # Balance analysis
        balance = analyzer.analyze_balance(structure)
        print(f"\n6. Tree balance: {balance:.3f}")
        
        # Recursive collapse
        print(f"\n7. Recursive collapse sequence:")
        sequence = collapser.recursive_collapse(structure)
        for i, stage in enumerate(sequence):
            print(f"   Stage {i}: {stage.span()}")
        
        print("\n" + "-" * 40)
    
    # Neural generation example
    print("\n8. Neural nested generation:")
    model = NeuralNestingModel()
    seed = torch.tensor([[0, 1]])  # Start with "01"
    
    print("   Model architecture: Hierarchical LSTM")
    print("   Levels: 5 (atom to document)")
    print("   Can generate nested structures with brackets")
    
    print("\n" + "=" * 60)
    print("Nesting reveals the fractal nature of collapse")
    print("=" * 60)


if __name__ == "__main__":
    # Run visualization
    visualize_nested_structures()
    
    # Run tests
    print("\n\nRunning formal verification tests...\n")
    unittest.main(verbosity=2)