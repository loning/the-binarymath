#!/usr/bin/env python3
"""
Chapter 009: TraceLexicon - Verification Program
Lexicon of Meaningful Trace Words

This program verifies the emergence of a lexicon—a dictionary of meaningful
"words" in the language of φ-traces, where each word carries semantic content
derived from its collapse pattern.

从ψ的轨迹模式中，涌现出具有语义的词汇——崩塌语言的字典。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import unittest
from typing import List, Tuple, Dict, Optional, Set, Union
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import numpy as np
from enum import Enum
import math


class SemanticCategory(Enum):
    """Semantic categories for trace words"""
    VOID = "void"  # Pure 0 patterns
    EMERGENCE = "emergence"  # 0→1 transitions
    RETURN = "return"  # 1→0 transitions
    OSCILLATION = "oscillation"  # Alternating patterns
    FIBONACCI = "fibonacci"  # Fibonacci-related patterns
    PRIME = "prime"  # Prime position patterns
    SYMMETRIC = "symmetric"  # Palindromic patterns
    COMPLEX = "complex"  # High entropy patterns


@dataclass
class TraceWord:
    """A word in the trace lexicon with semantic properties"""
    pattern: str
    category: SemanticCategory
    frequency: float
    entropy: float
    fibonacci_value: int  # Zeckendorf interpretation
    semantic_vector: torch.Tensor  # Learned semantic embedding
    related_words: Set[str] = field(default_factory=set)
    antonym: Optional[str] = None
    
    def __hash__(self):
        return hash(self.pattern)
    
    def semantic_distance(self, other: 'TraceWord') -> float:
        """Compute semantic distance to another word"""
        if self.semantic_vector is None or other.semantic_vector is None:
            return float('inf')
        return torch.dist(self.semantic_vector, other.semantic_vector).item()


class TraceLexicon:
    """
    The complete lexicon of trace words with semantic relationships.
    Words emerge from collapse patterns and form a structured vocabulary.
    """
    
    def __init__(self):
        self.words: Dict[str, TraceWord] = {}
        self.categories: Dict[SemanticCategory, List[TraceWord]] = defaultdict(list)
        self.semantic_space = SemanticSpace(embedding_dim=32)
        
        # Initialize fundamental words
        self._init_fundamental_words()
    
    def _init_fundamental_words(self):
        """Initialize the fundamental vocabulary"""
        # Void words
        self.add_word("0", SemanticCategory.VOID)
        self.add_word("00", SemanticCategory.VOID)
        self.add_word("000", SemanticCategory.VOID)
        
        # Emergence words
        self.add_word("01", SemanticCategory.EMERGENCE)
        self.add_word("001", SemanticCategory.EMERGENCE)
        self.add_word("0001", SemanticCategory.EMERGENCE)
        
        # Return words
        self.add_word("10", SemanticCategory.RETURN)
        self.add_word("100", SemanticCategory.RETURN)
        self.add_word("1000", SemanticCategory.RETURN)
        
        # Oscillation words
        self.add_word("0101", SemanticCategory.OSCILLATION)
        self.add_word("1010", SemanticCategory.OSCILLATION)
        self.add_word("010101", SemanticCategory.OSCILLATION)
        
        # Set up relationships
        self._establish_relationships()
    
    def add_word(self, pattern: str, category: SemanticCategory) -> TraceWord:
        """Add a word to the lexicon"""
        if '11' in pattern:
            raise ValueError(f"Pattern {pattern} violates φ-constraint")
        
        # Calculate properties
        entropy = self._calculate_entropy(pattern)
        fib_value = self._calculate_fibonacci_value(pattern)
        
        # Create embedding
        semantic_vec = self.semantic_space.embed_pattern(pattern)
        
        word = TraceWord(
            pattern=pattern,
            category=category,
            frequency=0.0,  # Will be updated from corpus
            entropy=entropy,
            fibonacci_value=fib_value,
            semantic_vector=semantic_vec
        )
        
        self.words[pattern] = word
        self.categories[category].append(word)
        
        return word
    
    def _calculate_entropy(self, pattern: str) -> float:
        """Calculate Shannon entropy of pattern"""
        if not pattern:
            return 0.0
        
        zeros = pattern.count('0')
        ones = pattern.count('1')
        total = len(pattern)
        
        p0 = zeros / total
        p1 = ones / total
        
        entropy = 0.0
        if p0 > 0:
            entropy -= p0 * math.log2(p0)
        if p1 > 0:
            entropy -= p1 * math.log2(p1)
        
        return entropy
    
    def _calculate_fibonacci_value(self, pattern: str) -> int:
        """Calculate Zeckendorf value of pattern"""
        if not pattern:
            return 0
        
        # Standard Fibonacci sequence
        fib = [1, 1]
        while len(fib) < len(pattern) + 1:
            fib.append(fib[-1] + fib[-2])
        
        value = 0
        for i, bit in enumerate(reversed(pattern)):
            if bit == '1':
                value += fib[i + 1]  # F(2), F(3), ...
        
        return value
    
    def _establish_relationships(self):
        """Establish semantic relationships between words"""
        # Antonym relationships
        if "01" in self.words and "10" in self.words:
            self.words["01"].antonym = "10"
            self.words["10"].antonym = "01"
        
        # Related words (same category)
        for category, words in self.categories.items():
            if len(words) > 1:
                for i, word in enumerate(words):
                    for j, other in enumerate(words):
                        if i != j:
                            word.related_words.add(other.pattern)
    
    def analyze_corpus(self, traces: List[str]):
        """Analyze a corpus to update word frequencies and discover new words"""
        # Count all subpatterns
        pattern_counts = Counter()
        total_count = 0
        
        for trace in traces:
            if '11' in trace:
                continue
            
            # Extract all valid subpatterns
            for length in range(1, min(8, len(trace) + 1)):
                for i in range(len(trace) - length + 1):
                    pattern = trace[i:i+length]
                    if '11' not in pattern:
                        pattern_counts[pattern] += 1
                        total_count += 1
        
        # Update frequencies for existing words
        for pattern, word in self.words.items():
            if pattern in pattern_counts:
                word.frequency = pattern_counts[pattern] / total_count
        
        # Discover new words (frequent patterns)
        threshold = 0.001  # Minimum frequency
        for pattern, count in pattern_counts.items():
            freq = count / total_count
            if freq >= threshold and pattern not in self.words:
                # Categorize new word
                category = self._categorize_pattern(pattern)
                self.add_word(pattern, category)
                self.words[pattern].frequency = freq
        
        # Re-establish relationships with new words
        self._establish_relationships()
    
    def _categorize_pattern(self, pattern: str) -> SemanticCategory:
        """Automatically categorize a pattern"""
        if all(c == '0' for c in pattern):
            return SemanticCategory.VOID
        
        if pattern == pattern[::-1]:
            return SemanticCategory.SYMMETRIC
        
        # Check for oscillation
        if len(pattern) >= 4:
            is_oscillating = all(
                pattern[i] != pattern[i+1] 
                for i in range(len(pattern)-1)
            )
            if is_oscillating:
                return SemanticCategory.OSCILLATION
        
        # Check transitions
        transitions = sum(1 for i in range(len(pattern)-1) 
                         if pattern[i] != pattern[i+1])
        
        if pattern[0] == '0' and pattern[-1] == '1' and transitions == 1:
            return SemanticCategory.EMERGENCE
        elif pattern[0] == '1' and pattern[-1] == '0' and transitions == 1:
            return SemanticCategory.RETURN
        
        # Check if Fibonacci value is prime
        fib_val = self._calculate_fibonacci_value(pattern)
        if self._is_prime(fib_val):
            return SemanticCategory.PRIME
        
        # High entropy patterns
        entropy = self._calculate_entropy(pattern)
        if entropy > 0.9:
            return SemanticCategory.COMPLEX
        
        # Default to Fibonacci category
        return SemanticCategory.FIBONACCI
    
    def _is_prime(self, n: int) -> bool:
        """Check if a number is prime"""
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True
    
    def find_synonyms(self, word: str, threshold: float = 0.5) -> List[str]:
        """Find semantically similar words"""
        if word not in self.words:
            return []
        
        target = self.words[word]
        synonyms = []
        
        for pattern, other in self.words.items():
            if pattern != word:
                distance = target.semantic_distance(other)
                if distance < threshold:
                    synonyms.append((pattern, distance))
        
        # Sort by similarity
        synonyms.sort(key=lambda x: x[1])
        return [s[0] for s in synonyms]
    
    def compose_words(self, word1: str, word2: str) -> Optional[str]:
        """Compose two words if valid"""
        if word1 not in self.words or word2 not in self.words:
            return None
        
        # Try different compositions
        candidates = [
            word1 + word2,
            word1 + "0" + word2,
            word1[:-1] + word2 if word1.endswith(word2[0]) else None
        ]
        
        for candidate in candidates:
            if candidate and '11' not in candidate:
                return candidate
        
        return None


class SemanticSpace(nn.Module):
    """
    Neural semantic space for trace words.
    Maps patterns to dense semantic vectors.
    """
    
    def __init__(self, embedding_dim: int = 32):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Pattern encoder
        self.pattern_encoder = nn.Sequential(
            nn.Linear(8, 64),  # Max pattern length 8
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim)
        )
        
        # Semantic properties encoder
        self.property_encoder = nn.Sequential(
            nn.Linear(4, 32),  # entropy, length, 0-density, transitions
            nn.ReLU(),
            nn.Linear(32, embedding_dim)
        )
        
        # Combiner
        self.combiner = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
    
    def embed_pattern(self, pattern: str) -> torch.Tensor:
        """Embed a pattern into semantic space"""
        # Convert pattern to tensor
        pattern_vec = torch.zeros(8)
        for i, bit in enumerate(pattern[:8]):
            pattern_vec[i] = float(bit)
        
        # Calculate properties
        entropy = self._entropy(pattern)
        length = len(pattern) / 8.0  # Normalized
        zero_density = pattern.count('0') / len(pattern)
        transitions = sum(1 for i in range(len(pattern)-1) 
                         if pattern[i] != pattern[i+1]) / max(1, len(pattern)-1)
        
        properties = torch.tensor([entropy, length, zero_density, transitions])
        
        # Encode
        pattern_emb = self.pattern_encoder(pattern_vec)
        property_emb = self.property_encoder(properties)
        
        # Combine
        combined = torch.cat([pattern_emb, property_emb], dim=0)
        semantic_emb = self.combiner(combined.unsqueeze(0)).squeeze(0)
        
        return semantic_emb
    
    def _entropy(self, pattern: str) -> float:
        """Calculate normalized entropy"""
        if not pattern:
            return 0.0
        
        p0 = pattern.count('0') / len(pattern)
        p1 = 1 - p0
        
        if p0 == 0 or p1 == 0:
            return 0.0
        
        return -(p0 * math.log2(p0) + p1 * math.log2(p1))


class WordRelationships:
    """
    Analyzes and maintains relationships between trace words.
    """
    
    def __init__(self, lexicon: TraceLexicon):
        self.lexicon = lexicon
        self.composition_rules: Dict[Tuple[str, str], str] = {}
        self.decomposition_rules: Dict[str, List[Tuple[str, str]]] = {}
        
    def learn_composition_rules(self, traces: List[str]):
        """Learn how words compose from trace corpus"""
        # Find frequent bigrams
        bigram_counts = Counter()
        
        for trace in traces:
            words = self._segment_into_words(trace)
            for i in range(len(words) - 1):
                if words[i] in self.lexicon.words and words[i+1] in self.lexicon.words:
                    bigram_counts[(words[i], words[i+1])] += 1
        
        # If no natural bigrams, create some rules from common patterns
        if not bigram_counts:
            # Add some basic composition rules
            basic_rules = [
                ("0", "1", "01"),
                ("1", "0", "10"),
                ("00", "1", "001"),
                ("01", "0", "010"),
                ("10", "0", "100")
            ]
            for w1, w2, composed in basic_rules:
                if w1 in self.lexicon.words and w2 in self.lexicon.words:
                    self.composition_rules[(w1, w2)] = composed
                    if composed not in self.decomposition_rules:
                        self.decomposition_rules[composed] = []
                    self.decomposition_rules[composed].append((w1, w2))
        else:
            # Learn valid compositions from bigrams
            for (w1, w2), count in bigram_counts.items():
                if count >= 1:  # Lower threshold
                    composed = self.lexicon.compose_words(w1, w2)
                    if composed:
                        self.composition_rules[(w1, w2)] = composed
                        
                        # Also track decomposition
                        if composed not in self.decomposition_rules:
                            self.decomposition_rules[composed] = []
                        self.decomposition_rules[composed].append((w1, w2))
    
    def _segment_into_words(self, trace: str) -> List[str]:
        """Segment trace into known words (greedy)"""
        words = []
        i = 0
        
        while i < len(trace):
            # Try longest match first
            found = False
            for length in range(min(8, len(trace) - i), 0, -1):
                pattern = trace[i:i+length]
                if pattern in self.lexicon.words:
                    words.append(pattern)
                    i += length
                    found = True
                    break
            
            if not found:
                # Single bit fallback
                words.append(trace[i])
                i += 1
        
        return words
    
    def find_word_family(self, root: str) -> Set[str]:
        """Find all words related to a root word"""
        if root not in self.lexicon.words:
            return set()
        
        family = {root}
        to_explore = [root]
        
        while to_explore:
            current = to_explore.pop()
            
            # Add related words
            if current in self.lexicon.words:
                for related in self.lexicon.words[current].related_words:
                    if related not in family:
                        family.add(related)
                        to_explore.append(related)
            
            # Add compositions
            for (w1, w2), composed in self.composition_rules.items():
                if w1 == current or w2 == current:
                    if composed not in family:
                        family.add(composed)
                        to_explore.append(composed)
        
        return family


class LexicalAnalyzer:
    """
    Performs lexical analysis on trace texts.
    """
    
    def __init__(self, lexicon: TraceLexicon):
        self.lexicon = lexicon
        
    def analyze_text(self, trace: str) -> Dict[str, any]:
        """Perform complete lexical analysis"""
        # Basic statistics
        analysis = {
            'length': len(trace),
            'valid': '11' not in trace,
            'entropy': self._calculate_entropy(trace),
            'word_count': 0,
            'vocabulary_size': 0,
            'category_distribution': Counter(),
            'rare_words': [],
            'common_words': []
        }
        
        if not analysis['valid']:
            return analysis
        
        # Segment into words
        words = self._optimal_segmentation(trace)
        analysis['word_count'] = len(words)
        analysis['words'] = words
        
        # Analyze vocabulary
        unique_words = set(words)
        analysis['vocabulary_size'] = len(unique_words)
        
        # Category distribution
        for word in words:
            if word in self.lexicon.words:
                category = self.lexicon.words[word].category
                analysis['category_distribution'][category] += 1
        
        # Find rare and common words
        word_freqs = [(w, self.lexicon.words[w].frequency) 
                      for w in unique_words 
                      if w in self.lexicon.words]
        word_freqs.sort(key=lambda x: x[1])
        
        if word_freqs:
            analysis['rare_words'] = [w for w, f in word_freqs[:3]]
            analysis['common_words'] = [w for w, f in word_freqs[-3:]]
        
        return analysis
    
    def _optimal_segmentation(self, trace: str) -> List[str]:
        """Find optimal word segmentation using dynamic programming"""
        n = len(trace)
        if n == 0:
            return []
        
        # dp[i] = (cost, segmentation) for trace[0:i]
        dp = [(float('inf'), [])] * (n + 1)
        dp[0] = (0, [])
        
        for i in range(1, n + 1):
            for j in range(max(0, i - 8), i):  # Max word length 8
                pattern = trace[j:i]
                
                if pattern in self.lexicon.words:
                    # Known word - low cost
                    cost = dp[j][0] + 1
                elif '11' not in pattern:
                    # Unknown but valid - higher cost
                    cost = dp[j][0] + len(pattern)
                else:
                    continue  # Invalid pattern
                
                if cost < dp[i][0]:
                    dp[i] = (cost, dp[j][1] + [pattern])
        
        return dp[n][1]
    
    def _calculate_entropy(self, trace: str) -> float:
        """Calculate trace entropy"""
        if not trace:
            return 0.0
        
        p0 = trace.count('0') / len(trace)
        p1 = 1 - p0
        
        if p0 == 0 or p1 == 0:
            return 0.0
        
        return -(p0 * math.log2(p0) + p1 * math.log2(p1))


class LexiconMetrics:
    """
    Computes various metrics on the trace lexicon.
    """
    
    @staticmethod
    def vocabulary_growth(lexicon: TraceLexicon, corpus: List[str]) -> List[int]:
        """Track vocabulary growth as corpus is processed"""
        growth = []
        temp_lexicon = TraceLexicon()
        
        for i, trace in enumerate(corpus):
            temp_lexicon.analyze_corpus([trace])
            growth.append(len(temp_lexicon.words))
        
        return growth
    
    @staticmethod
    def zipf_coefficient(lexicon: TraceLexicon) -> float:
        """Calculate Zipf's law coefficient for word frequencies"""
        # Get frequencies in descending order
        freqs = sorted([w.frequency for w in lexicon.words.values() 
                       if w.frequency > 0], reverse=True)
        
        if len(freqs) < 2:
            return 0.0
        
        # Fit power law: freq = C / rank^α
        # log(freq) = log(C) - α * log(rank)
        log_freqs = [math.log(f) for f in freqs[:50]]  # Top 50
        log_ranks = [math.log(i+1) for i in range(len(log_freqs))]
        
        # Linear regression
        n = len(log_freqs)
        sum_x = sum(log_ranks)
        sum_y = sum(log_freqs)
        sum_xy = sum(x*y for x, y in zip(log_ranks, log_freqs))
        sum_x2 = sum(x*x for x in log_ranks)
        
        alpha = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        
        return -alpha  # Return positive exponent
    
    @staticmethod
    def semantic_coherence(lexicon: TraceLexicon) -> float:
        """Measure semantic coherence of categories"""
        coherence_scores = []
        
        for category, words in lexicon.categories.items():
            if len(words) < 2:
                continue
            
            # Average pairwise distance within category
            distances = []
            for i, w1 in enumerate(words):
                for j, w2 in enumerate(words[i+1:], i+1):
                    dist = w1.semantic_distance(w2)
                    if dist < float('inf'):
                        distances.append(dist)
            
            if distances:
                avg_distance = sum(distances) / len(distances)
                coherence_scores.append(1.0 / (1.0 + avg_distance))
        
        return sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0.0


class TraceLexiconTests(unittest.TestCase):
    """Test trace lexicon functionality"""
    
    def setUp(self):
        self.lexicon = TraceLexicon()
        
        # Test corpus
        self.corpus = [
            "0101010101",
            "0010010010",
            "1001001001",
            "0100100100",
            "00100010001000",
            "10101010101010",
            "00000001",
            "10000000",
            "01010010100101"
        ]
        
        self.lexicon.analyze_corpus(self.corpus)
    
    def test_fundamental_words(self):
        """Test: Fundamental words are present"""
        fundamental = ["0", "1", "00", "01", "10", "000"]
        
        for word in fundamental:
            self.assertIn(word, self.lexicon.words)
    
    def test_phi_constraint(self):
        """Test: No word contains '11'"""
        for pattern, word in self.lexicon.words.items():
            self.assertNotIn('11', pattern)
    
    def test_semantic_categories(self):
        """Test: Words are properly categorized"""
        # Void words
        self.assertEqual(self.lexicon.words["0"].category, SemanticCategory.VOID)
        self.assertEqual(self.lexicon.words["00"].category, SemanticCategory.VOID)
        
        # Emergence words
        self.assertEqual(self.lexicon.words["01"].category, SemanticCategory.EMERGENCE)
        
        # Return words
        self.assertEqual(self.lexicon.words["10"].category, SemanticCategory.RETURN)
    
    def test_fibonacci_values(self):
        """Test: Fibonacci values are correctly calculated"""
        test_cases = [
            ("1", 1),     # F(2) = 1
            ("10", 2),    # F(3) = 2
            ("100", 3),   # F(4) = 3
            ("101", 4),   # F(2) + F(4) = 1 + 3 = 4
            ("1000", 5),  # F(5) = 5
        ]
        
        for pattern, expected in test_cases:
            if pattern in self.lexicon.words:
                actual = self.lexicon.words[pattern].fibonacci_value
                self.assertEqual(actual, expected)
    
    def test_word_relationships(self):
        """Test: Word relationships are established"""
        # Antonyms
        if "01" in self.lexicon.words and "10" in self.lexicon.words:
            self.assertEqual(self.lexicon.words["01"].antonym, "10")
            self.assertEqual(self.lexicon.words["10"].antonym, "01")
        
        # Related words
        void_words = self.lexicon.categories[SemanticCategory.VOID]
        if len(void_words) > 1:
            # Each void word should be related to others
            for word in void_words:
                self.assertGreater(len(word.related_words), 0)
    
    def test_word_composition(self):
        """Test: Words can be composed validly"""
        # Test valid composition
        result = self.lexicon.compose_words("01", "00")
        self.assertIsNotNone(result)
        self.assertNotIn('11', result)
        
        # Test invalid composition (would create 11)
        result = self.lexicon.compose_words("01", "10")
        if result:
            self.assertNotIn('11', result)
    
    def test_semantic_embedding(self):
        """Test: Semantic embeddings have correct properties"""
        # All words should have embeddings
        for word in self.lexicon.words.values():
            self.assertIsNotNone(word.semantic_vector)
            self.assertEqual(word.semantic_vector.shape[0], 32)
        
        # Similar words should have close embeddings
        if "0" in self.lexicon.words and "00" in self.lexicon.words:
            dist = self.lexicon.words["0"].semantic_distance(self.lexicon.words["00"])
            self.assertLess(dist, 10.0)  # Reasonable threshold
    
    def test_synonym_finding(self):
        """Test: Can find semantically similar words"""
        if "01" in self.lexicon.words:
            synonyms = self.lexicon.find_synonyms("01", threshold=5.0)
            # Should find some similar emergence patterns
            self.assertGreater(len(synonyms), 0)
    
    def test_lexical_analysis(self):
        """Test: Lexical analysis works correctly"""
        analyzer = LexicalAnalyzer(self.lexicon)
        
        trace = "0101001010"
        analysis = analyzer.analyze_text(trace)
        
        self.assertEqual(analysis['length'], 10)
        self.assertTrue(analysis['valid'])
        self.assertGreater(analysis['word_count'], 0)
        self.assertGreater(analysis['vocabulary_size'], 0)
    
    def test_word_relationships_learning(self):
        """Test: Word relationships can be learned"""
        relationships = WordRelationships(self.lexicon)
        relationships.learn_composition_rules(self.corpus)
        
        # Should learn some composition rules
        self.assertGreater(len(relationships.composition_rules), 0)
    
    def test_metrics(self):
        """Test: Lexicon metrics are computed correctly"""
        # Zipf coefficient
        zipf = LexiconMetrics.zipf_coefficient(self.lexicon)
        self.assertGreater(zipf, 0)  # Should follow power law
        
        # Semantic coherence
        coherence = LexiconMetrics.semantic_coherence(self.lexicon)
        self.assertGreaterEqual(coherence, 0)
        self.assertLessEqual(coherence, 1)


def visualize_trace_lexicon():
    """Visualize the trace lexicon and its properties"""
    print("=" * 60)
    print("Trace Lexicon: The Vocabulary of Collapse")
    print("=" * 60)
    
    # Create and populate lexicon
    lexicon = TraceLexicon()
    corpus = [
        "01010101010101",
        "00100100100100",
        "10010010010010",
        "01001001001001",
        "00010001000100",
        "10101010101010",
        "00000000000001",
        "10000000000000",
        "01010010100101",
        "00101001010010"
    ]
    
    lexicon.analyze_corpus(corpus)
    
    # 1. Show vocabulary by category
    print("\n1. Vocabulary by Category:")
    for category in SemanticCategory:
        words = lexicon.categories[category]
        if words:
            patterns = [w.pattern for w in words[:5]]  # Top 5
            print(f"   {category.value}: {', '.join(patterns)}")
    
    # 2. Most frequent words
    print("\n2. Most Frequent Words:")
    freq_words = sorted(lexicon.words.values(), 
                       key=lambda w: w.frequency, 
                       reverse=True)[:10]
    
    print("   Pattern | Frequency | Category    | Fib Value")
    print("   --------|-----------|-------------|----------")
    for word in freq_words:
        if word.frequency > 0:
            print(f"   {word.pattern:7} | {word.frequency:9.3f} | "
                  f"{word.category.value:11} | {word.fibonacci_value:9}")
    
    # 3. Word relationships
    print("\n3. Word Relationships:")
    relationships = WordRelationships(lexicon)
    relationships.learn_composition_rules(corpus)
    
    print("   Antonym pairs:")
    shown = set()
    for word in lexicon.words.values():
        if word.antonym and word.pattern not in shown:
            print(f"   {word.pattern} ↔ {word.antonym}")
            shown.add(word.pattern)
            shown.add(word.antonym)
    
    print("\n   Composition examples:")
    for (w1, w2), composed in list(relationships.composition_rules.items())[:5]:
        print(f"   {w1} + {w2} → {composed}")
    
    # 4. Semantic analysis
    print("\n4. Semantic Analysis:")
    analyzer = LexicalAnalyzer(lexicon)
    
    test_trace = "0101001000100101"
    analysis = analyzer.analyze_text(test_trace)
    
    print(f"   Trace: {test_trace}")
    print(f"   Words: {' '.join(analysis['words'])}")
    print(f"   Word count: {analysis['word_count']}")
    print(f"   Vocabulary size: {analysis['vocabulary_size']}")
    print(f"   Category distribution:")
    for cat, count in analysis['category_distribution'].most_common():
        print(f"      {cat.value}: {count}")
    
    # 5. Lexicon metrics
    print("\n5. Lexicon Metrics:")
    
    # Vocabulary size
    print(f"   Total vocabulary: {len(lexicon.words)} words")
    
    # Zipf coefficient
    zipf = LexiconMetrics.zipf_coefficient(lexicon)
    print(f"   Zipf coefficient: {zipf:.2f}")
    
    # Semantic coherence
    coherence = LexiconMetrics.semantic_coherence(lexicon)
    print(f"   Semantic coherence: {coherence:.3f}")
    
    # 6. Word families
    print("\n6. Word Families:")
    if "01" in lexicon.words:
        family = relationships.find_word_family("01")
        print(f"   Family of '01': {', '.join(sorted(family)[:8])}")
    
    print("\n" + "=" * 60)
    print("The lexicon emerges from collapse patterns")
    print("=" * 60)


if __name__ == "__main__":
    # Run visualization
    visualize_trace_lexicon()
    
    # Run tests
    print("\n\nRunning formal verification tests...\n")
    unittest.main(verbosity=2)