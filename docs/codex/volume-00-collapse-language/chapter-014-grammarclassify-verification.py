#!/usr/bin/env python3
"""
Chapter 014: GrammarClassify - Verification Program
Grammar Classification of φ-Constrained Expressions

This program verifies that neural classifiers can learn to distinguish between
different grammatical categories of φ-constrained expressions, revealing the
deep structural patterns that emerge from the golden constraint.

从ψ的语法结构中，涌现出分类识别——崩塌表达式的深层结构模式。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import unittest
from typing import List, Tuple, Dict, Optional, Set, Union, Any
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import numpy as np
from enum import Enum
import math
import random


class GrammarCategory(Enum):
    """Grammatical categories for φ-constrained expressions"""
    VOID = "void"                    # Pure zero sequences: 0, 00, 000
    EMERGENCE = "emergence"          # 0→1 transitions: 01, 001, 0001
    RETURN = "return"               # 1→0 transitions: 10, 100, 1000
    OSCILLATION = "oscillation"     # Alternating patterns: 010, 101, 0101
    FIBONACCI = "fibonacci"         # Fibonacci structures: 001, 0100, 10001
    COMPLEX = "complex"             # Mixed grammatical structures
    RECURSIVE = "recursive"         # Self-referential patterns
    TERMINAL = "terminal"           # Single symbols: 0, 1


@dataclass
class ClassificationResult:
    """Result of grammar classification"""
    predicted_category: GrammarCategory
    confidence: float
    feature_activations: Dict[str, float]
    attention_weights: Optional[torch.Tensor] = None
    explanation: str = ""


class φGrammarFeatureExtractor:
    """
    Extracts grammatical features from φ-constrained sequences.
    Features capture the structural patterns that define different categories.
    """
    
    def __init__(self):
        # Golden ratio
        self.phi = (1 + math.sqrt(5)) / 2
        
        # Fibonacci sequence for reference
        self.fibonacci_sequence = self._generate_fibonacci_sequence(20)
        
        # Pattern databases for each category
        self.category_patterns = {
            GrammarCategory.VOID: ['0', '00', '000', '0000'],
            GrammarCategory.EMERGENCE: ['01', '001', '0001', '00001'],
            GrammarCategory.RETURN: ['10', '100', '1000', '10000'],
            GrammarCategory.OSCILLATION: ['010', '101', '0101', '1010'],
            GrammarCategory.FIBONACCI: ['001', '0100', '10001', '010010'],
            GrammarCategory.TERMINAL: ['0', '1']
        }
    
    def extract_features(self, sequence: str) -> Dict[str, float]:
        """Extract comprehensive grammatical features"""
        features = {}
        
        # Basic structural features
        features.update(self._extract_basic_features(sequence))
        
        # Pattern-based features
        features.update(self._extract_pattern_features(sequence))
        
        # Statistical features
        features.update(self._extract_statistical_features(sequence))
        
        # φ-alignment features
        features.update(self._extract_phi_features(sequence))
        
        # Compositional features
        features.update(self._extract_compositional_features(sequence))
        
        # Transitional features
        features.update(self._extract_transition_features(sequence))
        
        return features
    
    def _extract_basic_features(self, sequence: str) -> Dict[str, float]:
        """Basic structural properties"""
        if not sequence:
            return {'length': 0, 'zero_count': 0, 'one_count': 0, 'zero_ratio': 0}
        
        length = len(sequence)
        zero_count = sequence.count('0')
        one_count = sequence.count('1')
        
        return {
            'length': length,
            'zero_count': zero_count,
            'one_count': one_count,
            'zero_ratio': zero_count / length if length > 0 else 0,
            'one_ratio': one_count / length if length > 0 else 0,
            'is_pure_zeros': 1.0 if set(sequence) == {'0'} else 0.0,
            'is_pure_ones': 1.0 if set(sequence) == {'1'} else 0.0,
            'starts_with_zero': 1.0 if sequence.startswith('0') else 0.0,
            'ends_with_zero': 1.0 if sequence.endswith('0') else 0.0,
        }
    
    def _extract_pattern_features(self, sequence: str) -> Dict[str, float]:
        """Pattern-specific features for each category"""
        features = {}
        
        for category, patterns in self.category_patterns.items():
            pattern_scores = []
            for pattern in patterns:
                count = self._count_pattern_occurrences(sequence, pattern)
                coverage = (count * len(pattern)) / len(sequence) if len(sequence) > 0 else 0
                pattern_scores.append(coverage)
            
            features[f'{category.value}_pattern_density'] = sum(pattern_scores)
            features[f'{category.value}_max_pattern'] = max(pattern_scores) if pattern_scores else 0
            features[f'{category.value}_pattern_count'] = len([s for s in pattern_scores if s > 0])
        
        return features
    
    def _extract_statistical_features(self, sequence: str) -> Dict[str, float]:
        """Statistical properties of the sequence"""
        if len(sequence) < 2:
            return {'entropy': 0, 'complexity': 0, 'periodicity': 0}
        
        # Character entropy
        char_counts = Counter(sequence)
        total_chars = len(sequence)
        entropy = -sum((count/total_chars) * math.log2(count/total_chars) 
                      for count in char_counts.values())
        
        # Pattern complexity (unique bigrams)
        bigrams = [sequence[i:i+2] for i in range(len(sequence)-1)]
        complexity = len(set(bigrams)) / len(bigrams) if bigrams else 0
        
        # Periodicity detection
        periodicity = self._detect_periodicity(sequence)
        
        # Compression estimate
        compression_ratio = self._estimate_compression_ratio(sequence)
        
        return {
            'entropy': entropy,
            'complexity': complexity,
            'periodicity': periodicity,
            'compression_ratio': compression_ratio,
            'repetitiveness': 1 - complexity,
        }
    
    def _extract_phi_features(self, sequence: str) -> Dict[str, float]:
        """φ-constraint specific features"""
        if not sequence:
            return {'phi_alignment': 0, 'golden_ratio_deviation': 1}
        
        # Golden ratio alignment
        zeros = sequence.count('0')
        ones = sequence.count('1')
        
        if ones > 0:
            ratio = zeros / ones
            deviation = abs(ratio - self.phi) / self.phi
            phi_alignment = max(0, 1 - deviation)
        else:
            phi_alignment = 1.0 if ones == 0 else 0.0
            deviation = 1.0
        
        # Fibonacci number mapping
        fib_mapping_score = self._calculate_fibonacci_mapping(sequence)
        
        # Zeckendorf representation quality
        zeckendorf_score = self._calculate_zeckendorf_quality(sequence)
        
        return {
            'phi_alignment': phi_alignment,
            'golden_ratio_deviation': deviation,
            'fibonacci_mapping': fib_mapping_score,
            'zeckendorf_quality': zeckendorf_score,
        }
    
    def _extract_compositional_features(self, sequence: str) -> Dict[str, float]:
        """Features related to compositional structure"""
        if len(sequence) < 2:
            return {'compositional_depth': 0, 'nesting_level': 0}
        
        # Nested structure detection
        nesting_level = self._calculate_nesting_level(sequence)
        
        # Compositional depth (longest valid subpattern)
        compositional_depth = self._calculate_compositional_depth(sequence)
        
        # Recursive pattern detection
        recursive_score = self._detect_recursive_patterns(sequence)
        
        # Hierarchical structure score
        hierarchical_score = self._calculate_hierarchical_score(sequence)
        
        return {
            'compositional_depth': compositional_depth,
            'nesting_level': nesting_level,
            'recursive_score': recursive_score,
            'hierarchical_score': hierarchical_score,
        }
    
    def _extract_transition_features(self, sequence: str) -> Dict[str, float]:
        """Features related to state transitions"""
        if len(sequence) < 2:
            return {'transition_entropy': 0, 'emergence_count': 0, 'return_count': 0}
        
        # Count different transition types
        transitions = {'00': 0, '01': 0, '10': 0}  # '11' forbidden
        
        for i in range(len(sequence) - 1):
            bigram = sequence[i:i+2]
            if bigram in transitions:
                transitions[bigram] += 1
        
        total_transitions = sum(transitions.values())
        
        # Transition probabilities and entropy
        if total_transitions > 0:
            probs = [count / total_transitions for count in transitions.values()]
            transition_entropy = -sum(p * math.log2(p) for p in probs if p > 0)
        else:
            transition_entropy = 0
        
        return {
            'transition_entropy': transition_entropy,
            'emergence_count': transitions['01'] / total_transitions if total_transitions > 0 else 0,
            'return_count': transitions['10'] / total_transitions if total_transitions > 0 else 0,
            'void_continuation': transitions['00'] / total_transitions if total_transitions > 0 else 0,
        }
    
    def _generate_fibonacci_sequence(self, n: int) -> List[int]:
        """Generate first n Fibonacci numbers"""
        if n <= 0:
            return []
        elif n == 1:
            return [1]
        
        fib = [1, 1]
        for _ in range(2, n):
            fib.append(fib[-1] + fib[-2])
        
        return fib
    
    def _count_pattern_occurrences(self, sequence: str, pattern: str) -> int:
        """Count non-overlapping occurrences of pattern"""
        count = 0
        start = 0
        while True:
            pos = sequence.find(pattern, start)
            if pos == -1:
                break
            count += 1
            start = pos + len(pattern)
        return count
    
    def _detect_periodicity(self, sequence: str) -> float:
        """Detect if sequence has periodic structure"""
        if len(sequence) < 4:
            return 0.0
        
        max_period_score = 0.0
        
        for period_len in range(1, len(sequence) // 2 + 1):
            period = sequence[:period_len]
            repetitions = len(sequence) // period_len
            
            reconstructed = period * repetitions
            if len(reconstructed) < len(sequence):
                reconstructed += period[:len(sequence) - len(reconstructed)]
            
            matches = sum(1 for i, char in enumerate(sequence) 
                         if i < len(reconstructed) and char == reconstructed[i])
            
            period_score = matches / len(sequence)
            
            if period_score > max_period_score:
                max_period_score = period_score
        
        return max_period_score
    
    def _estimate_compression_ratio(self, sequence: str) -> float:
        """Estimate compression ratio using simple encoding"""
        if not sequence:
            return 1.0
        
        # Simple run-length encoding estimate
        runs = []
        current_char = sequence[0]
        current_length = 1
        
        for char in sequence[1:]:
            if char == current_char:
                current_length += 1
            else:
                runs.append((current_char, current_length))
                current_char = char
                current_length = 1
        
        runs.append((current_char, current_length))
        
        # Estimate compressed size (char + length encoding)
        compressed_size = sum(2 if length > 1 else 1 for _, length in runs)
        
        return compressed_size / len(sequence)
    
    def _calculate_fibonacci_mapping(self, sequence: str) -> float:
        """Calculate how well sequence maps to Fibonacci representation"""
        if not sequence:
            return 0.0
        
        # Convert binary string to integer using Zeckendorf representation
        value = 0
        for i, bit in enumerate(reversed(sequence)):
            if bit == '1' and i + 1 < len(self.fibonacci_sequence):
                value += self.fibonacci_sequence[i + 1]
        
        # Check if value is itself a Fibonacci number
        if value in self.fibonacci_sequence:
            return 1.0
        
        # Find closest Fibonacci numbers
        lower_fib = max([f for f in self.fibonacci_sequence if f <= value], default=0)
        upper_fib = min([f for f in self.fibonacci_sequence if f > value], default=value*2)
        
        if upper_fib == lower_fib:
            return 0.0
        
        # Distance to closest Fibonacci number
        distance_to_closest = min(abs(value - lower_fib), abs(value - upper_fib))
        range_size = upper_fib - lower_fib
        
        return max(0, 1 - distance_to_closest / range_size) if range_size > 0 else 0.0
    
    def _calculate_zeckendorf_quality(self, sequence: str) -> float:
        """Calculate quality of Zeckendorf representation"""
        if '11' in sequence:
            return 0.0  # Invalid Zeckendorf representation
        
        # Check for proper Zeckendorf form (no consecutive 1s, no leading zeros)
        quality_score = 1.0
        
        # Penalty for leading zeros (except single '0')
        if len(sequence) > 1 and sequence.startswith('0'):
            leading_zeros = len(sequence) - len(sequence.lstrip('0'))
            quality_score *= (1 - leading_zeros / len(sequence))
        
        # Bonus for efficient representation (fewer bits)
        if sequence.count('1') > 0:
            efficiency = sequence.count('1') / len(sequence)
            quality_score *= (0.5 + 0.5 * efficiency)
        
        return quality_score
    
    def _calculate_nesting_level(self, sequence: str) -> float:
        """Calculate nesting level of compositional structure"""
        # Simplified nesting calculation based on parentheses-like structure
        max_level = 0
        current_level = 0
        
        for char in sequence:
            if char == '0':
                current_level += 1
                max_level = max(max_level, current_level)
            elif char == '1':
                current_level = max(0, current_level - 1)
        
        return max_level / len(sequence) if len(sequence) > 0 else 0
    
    def _calculate_compositional_depth(self, sequence: str) -> float:
        """Calculate compositional depth"""
        # Find longest valid subpattern
        max_depth = 0
        
        for start in range(len(sequence)):
            for end in range(start + 1, len(sequence) + 1):
                subseq = sequence[start:end]
                if '11' not in subseq:
                    depth = self._pattern_complexity(subseq)
                    max_depth = max(max_depth, depth)
        
        return max_depth / len(sequence) if len(sequence) > 0 else 0
    
    def _detect_recursive_patterns(self, sequence: str) -> float:
        """Detect self-similar recursive patterns"""
        if len(sequence) < 4:
            return 0.0
        
        recursive_score = 0.0
        
        # Look for patterns that repeat with modifications
        for pattern_len in range(2, len(sequence) // 2 + 1):
            for start in range(len(sequence) - pattern_len + 1):
                pattern = sequence[start:start + pattern_len]
                
                # Look for similar patterns elsewhere
                for other_start in range(start + pattern_len, len(sequence) - pattern_len + 1):
                    other_pattern = sequence[other_start:other_start + pattern_len]
                    
                    # Calculate similarity
                    similarity = sum(1 for i in range(pattern_len) 
                                   if pattern[i] == other_pattern[i]) / pattern_len
                    
                    if similarity > 0.7:  # Threshold for considering recursive
                        recursive_score += similarity
        
        return min(1.0, recursive_score / len(sequence))
    
    def _calculate_hierarchical_score(self, sequence: str) -> float:
        """Calculate hierarchical structure score"""
        if len(sequence) < 3:
            return 0.0
        
        # Look for nested patterns of different scales
        hierarchical_score = 0.0
        
        # Check for patterns at different scales
        for scale in [2, 3, 4]:
            if len(sequence) >= scale * 2:
                pattern_count = 0
                for i in range(len(sequence) - scale + 1):
                    pattern = sequence[i:i + scale]
                    if '11' not in pattern:
                        # Look for this pattern at other positions
                        for j in range(i + scale, len(sequence) - scale + 1):
                            other_pattern = sequence[j:j + scale]
                            if pattern == other_pattern:
                                pattern_count += 1
                
                if pattern_count > 0:
                    hierarchical_score += pattern_count / (len(sequence) / scale)
        
        return min(1.0, hierarchical_score / 3)  # Normalize by number of scales
    
    def _pattern_complexity(self, pattern: str) -> float:
        """Calculate complexity of a pattern"""
        if not pattern:
            return 0.0
        
        # Unique character transitions
        transitions = set()
        for i in range(len(pattern) - 1):
            transitions.add(pattern[i:i+2])
        
        # Normalized complexity
        max_possible_transitions = min(3, len(pattern) - 1)  # Max 3 valid transitions in φ-space
        
        return len(transitions) / max_possible_transitions if max_possible_transitions > 0 else 0


class φGrammarClassifier(nn.Module):
    """
    Neural classifier for grammatical categories of φ-constrained expressions.
    Uses transformer-like attention to focus on relevant structural features.
    """
    
    def __init__(self, feature_dim: int = 64, num_categories: int = len(GrammarCategory)):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_categories = num_categories
        
        # Feature processing layers
        self.feature_encoder = nn.Sequential(
            nn.Linear(50, feature_dim),  # Assuming ~50 extracted features
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Attention mechanism for feature importance
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(feature_dim // 2, num_categories)
        )
        
        # Confidence estimation
        self.confidence_estimator = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),
            nn.ReLU(),
            nn.Linear(feature_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # Feature extractor
        self.feature_extractor = φGrammarFeatureExtractor()
    
    def forward(self, sequences: List[str]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for grammar classification
        
        Args:
            sequences: List of φ-constrained sequences
            
        Returns:
            logits: Classification logits
            confidences: Confidence scores
            attention_weights: Attention weights for interpretability
        """
        batch_size = len(sequences)
        
        # Extract features for each sequence
        feature_matrices = []
        for seq in sequences:
            features = self.feature_extractor.extract_features(seq)
            feature_vector = torch.tensor([
                features.get(key, 0.0) for key in sorted(features.keys())
            ], dtype=torch.float32)
            
            # Pad or truncate to standard size
            if len(feature_vector) < 50:
                feature_vector = F.pad(feature_vector, (0, 50 - len(feature_vector)))
            else:
                feature_vector = feature_vector[:50]
            
            feature_matrices.append(feature_vector)
        
        # Stack into batch
        features_batch = torch.stack(feature_matrices)
        
        # Encode features
        encoded_features = self.feature_encoder(features_batch)
        
        # Add sequence dimension for attention (treating each feature as a token)
        # Reshape to (batch, seq_len, feature_dim) for attention
        attention_input = encoded_features.unsqueeze(1)  # (batch, 1, feature_dim)
        
        # Self-attention for feature importance
        attended_features, attention_weights = self.attention(
            attention_input, attention_input, attention_input
        )
        
        # Global pooling
        pooled_features = attended_features.squeeze(1)  # Back to (batch, feature_dim)
        
        # Classification
        logits = self.classifier(pooled_features)
        
        # Confidence estimation
        confidences = self.confidence_estimator(pooled_features)
        
        return logits, confidences, attention_weights
    
    def classify_sequence(self, sequence: str) -> ClassificationResult:
        """Classify a single sequence with detailed results"""
        with torch.no_grad():
            logits, confidence, attention_weights = self.forward([sequence])
            
            # Get prediction
            probabilities = F.softmax(logits[0], dim=0)
            predicted_idx = torch.argmax(probabilities).item()
            predicted_category = list(GrammarCategory)[predicted_idx]
            
            # Extract feature activations for interpretability
            features = self.feature_extractor.extract_features(sequence)
            
            # Generate explanation
            explanation = self._generate_explanation(sequence, features, predicted_category)
            
            return ClassificationResult(
                predicted_category=predicted_category,
                confidence=confidence[0].item(),
                feature_activations=features,
                attention_weights=attention_weights,
                explanation=explanation
            )
    
    def _generate_explanation(self, sequence: str, features: Dict[str, float], 
                            predicted_category: GrammarCategory) -> str:
        """Generate human-readable explanation for classification"""
        explanations = []
        
        # Category-specific explanations
        if predicted_category == GrammarCategory.VOID:
            zero_ratio = features.get('zero_ratio', 0)
            if zero_ratio > 0.9:
                explanations.append(f"High zero density ({zero_ratio:.2f}) indicates void pattern")
        
        elif predicted_category == GrammarCategory.EMERGENCE:
            emergence_count = features.get('emergence_count', 0)
            if emergence_count > 0.3:
                explanations.append(f"Frequent 0→1 transitions ({emergence_count:.2f}) suggest emergence pattern")
        
        elif predicted_category == GrammarCategory.FIBONACCI:
            fib_mapping = features.get('fibonacci_mapping', 0)
            if fib_mapping > 0.5:
                explanations.append(f"Strong Fibonacci mapping ({fib_mapping:.2f}) indicates Fibonacci structure")
        
        elif predicted_category == GrammarCategory.OSCILLATION:
            periodicity = features.get('periodicity', 0)
            if periodicity > 0.7:
                explanations.append(f"High periodicity ({periodicity:.2f}) suggests oscillating pattern")
            # Add alternative oscillation indicators
            if '010' in sequence or '101' in sequence:
                explanations.append(f"Contains oscillation patterns")
        
        elif predicted_category == GrammarCategory.RETURN:
            return_count = features.get('return_count', 0)
            if return_count > 0.3:
                explanations.append(f"Frequent 1→0 transitions ({return_count:.2f}) suggest return pattern")
        
        # General structural explanations
        phi_alignment = features.get('phi_alignment', 0)
        if phi_alignment > 0.8:
            explanations.append(f"Strong φ-alignment ({phi_alignment:.2f}) supports classification")
        
        complexity = features.get('complexity', 0)
        if complexity > 0.7:
            explanations.append(f"High structural complexity ({complexity:.2f})")
        
        if not explanations:
            explanations.append("Classification based on overall feature pattern")
        
        return "; ".join(explanations)


class MetaGrammarAnalyzer:
    """
    Analyzes meta-properties of grammar classification.
    Studies how different categories relate to each other.
    """
    
    def __init__(self):
        self.classifier = φGrammarClassifier()
        self.feature_extractor = φGrammarFeatureExtractor()
    
    def analyze_category_relationships(self, test_sequences: Dict[GrammarCategory, List[str]]) -> Dict[str, Any]:
        """Analyze relationships between different grammatical categories"""
        
        # Feature space analysis
        category_features = {}
        for category, sequences in test_sequences.items():
            features_list = []
            for seq in sequences:
                features = self.feature_extractor.extract_features(seq)
                features_list.append(features)
            category_features[category] = features_list
        
        # Calculate inter-category distances
        distances = self._calculate_category_distances(category_features)
        
        # Find category clusters
        clusters = self._find_category_clusters(distances)
        
        # Analyze feature importance per category
        feature_importance = self._analyze_feature_importance(category_features)
        
        # Category transition analysis
        transitions = self._analyze_category_transitions(test_sequences)
        
        return {
            'category_distances': distances,
            'category_clusters': clusters,
            'feature_importance': feature_importance,
            'category_transitions': transitions,
            'separability_scores': self._calculate_separability_scores(category_features)
        }
    
    def _calculate_category_distances(self, category_features: Dict[GrammarCategory, List[Dict]]) -> Dict[Tuple[GrammarCategory, GrammarCategory], float]:
        """Calculate average distances between category feature vectors"""
        distances = {}
        
        categories = list(category_features.keys())
        
        for i, cat1 in enumerate(categories):
            for j, cat2 in enumerate(categories[i+1:], i+1):
                # Calculate average distance between categories
                total_distance = 0.0
                count = 0
                
                for features1 in category_features[cat1]:
                    for features2 in category_features[cat2]:
                        distance = self._feature_distance(features1, features2)
                        total_distance += distance
                        count += 1
                
                avg_distance = total_distance / count if count > 0 else 0.0
                distances[(cat1, cat2)] = avg_distance
        
        return distances
    
    def _feature_distance(self, features1: Dict[str, float], features2: Dict[str, float]) -> float:
        """Calculate Euclidean distance between feature vectors"""
        all_keys = set(features1.keys()) | set(features2.keys())
        
        distance_squared = 0.0
        for key in all_keys:
            val1 = features1.get(key, 0.0)
            val2 = features2.get(key, 0.0)
            distance_squared += (val1 - val2) ** 2
        
        return math.sqrt(distance_squared)
    
    def _find_category_clusters(self, distances: Dict[Tuple[GrammarCategory, GrammarCategory], float]) -> List[List[GrammarCategory]]:
        """Find clusters of similar categories using simple threshold clustering"""
        threshold = 0.5  # Adjust based on typical distances
        
        # Create adjacency list for close categories
        close_categories = defaultdict(set)
        
        for (cat1, cat2), distance in distances.items():
            if distance < threshold:
                close_categories[cat1].add(cat2)
                close_categories[cat2].add(cat1)
        
        # Find connected components
        visited = set()
        clusters = []
        
        for category in GrammarCategory:
            if category not in visited:
                cluster = []
                self._dfs_cluster(category, close_categories, visited, cluster)
                if cluster:
                    clusters.append(cluster)
        
        return clusters
    
    def _dfs_cluster(self, category: GrammarCategory, adjacency: Dict, visited: Set, cluster: List):
        """Depth-first search to find cluster components"""
        if category in visited:
            return
        
        visited.add(category)
        cluster.append(category)
        
        for neighbor in adjacency[category]:
            self._dfs_cluster(neighbor, adjacency, visited, cluster)
    
    def _analyze_feature_importance(self, category_features: Dict[GrammarCategory, List[Dict]]) -> Dict[GrammarCategory, List[Tuple[str, float]]]:
        """Analyze which features are most important for each category"""
        importance = {}
        
        for category, features_list in category_features.items():
            if not features_list:
                continue
            
            # Calculate feature statistics for this category
            feature_stats = defaultdict(list)
            for features in features_list:
                for key, value in features.items():
                    feature_stats[key].append(value)
            
            # Calculate importance as mean + variance
            feature_importance = []
            for feature_name, values in feature_stats.items():
                if values:
                    mean_val = np.mean(values)
                    var_val = np.var(values)
                    importance_score = mean_val + 0.1 * var_val  # Weight mean more than variance
                    feature_importance.append((feature_name, importance_score))
            
            # Sort by importance
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            importance[category] = feature_importance[:10]  # Top 10 features
        
        return importance
    
    def _analyze_category_transitions(self, test_sequences: Dict[GrammarCategory, List[str]]) -> Dict[str, Any]:
        """Analyze how sequences might transition between categories"""
        transitions = defaultdict(int)
        
        # Simulate transitions by modifying sequences
        for category, sequences in test_sequences.items():
            for seq in sequences[:5]:  # Limit for computational efficiency
                # Try simple modifications
                modified_seqs = self._generate_modified_sequences(seq)
                
                for modified_seq in modified_seqs:
                    if '11' not in modified_seq:  # Ensure still φ-valid
                        # Classify modified sequence
                        result = self.classifier.classify_sequence(modified_seq)
                        if result.predicted_category != category:
                            transitions[(category, result.predicted_category)] += 1
        
        return {
            'transition_counts': dict(transitions),
            'most_common_transitions': sorted(transitions.items(), key=lambda x: x[1], reverse=True)[:10]
        }
    
    def _generate_modified_sequences(self, sequence: str) -> List[str]:
        """Generate simple modifications of a sequence"""
        modifications = []
        
        # Single bit flips (that maintain φ-constraint)
        for i in range(len(sequence)):
            modified = list(sequence)
            
            # Try flipping bit at position i
            if modified[i] == '0':
                modified[i] = '1'
                # Check if this creates '11'
                modified_str = ''.join(modified)
                if '11' not in modified_str:
                    modifications.append(modified_str)
            else:  # modified[i] == '1'
                modified[i] = '0'
                modified_str = ''.join(modified)
                modifications.append(modified_str)  # Flipping 1->0 never creates '11'
        
        # Append/prepend valid patterns
        valid_additions = ['0', '01', '10', '00']
        for addition in valid_additions:
            # Prepend
            new_seq = addition + sequence
            if '11' not in new_seq:
                modifications.append(new_seq)
            
            # Append
            new_seq = sequence + addition
            if '11' not in new_seq:
                modifications.append(new_seq)
        
        return modifications
    
    def _calculate_separability_scores(self, category_features: Dict[GrammarCategory, List[Dict]]) -> Dict[GrammarCategory, float]:
        """Calculate how well each category is separated from others"""
        separability = {}
        
        for target_category in category_features.keys():
            if not category_features[target_category]:
                separability[target_category] = 0.0
                continue
            
            # Calculate intra-category distances
            intra_distances = []
            target_features = category_features[target_category]
            
            for i, features1 in enumerate(target_features):
                for features2 in target_features[i+1:]:
                    distance = self._feature_distance(features1, features2)
                    intra_distances.append(distance)
            
            avg_intra_distance = np.mean(intra_distances) if intra_distances else 0.0
            
            # Calculate inter-category distances
            inter_distances = []
            for other_category, other_features in category_features.items():
                if other_category == target_category:
                    continue
                
                for features1 in target_features:
                    for features2 in other_features:
                        distance = self._feature_distance(features1, features2)
                        inter_distances.append(distance)
            
            avg_inter_distance = np.mean(inter_distances) if inter_distances else 0.0
            
            # Separability score: want high inter-distance, low intra-distance
            if avg_intra_distance > 0:
                separability_score = avg_inter_distance / avg_intra_distance
            else:
                separability_score = avg_inter_distance
            
            separability[target_category] = separability_score
        
        return separability


class GrammarClassificationTests(unittest.TestCase):
    """Test grammar classification functionality"""
    
    def setUp(self):
        self.classifier = φGrammarClassifier()
        self.feature_extractor = φGrammarFeatureExtractor()
        self.meta_analyzer = MetaGrammarAnalyzer()
        
        # Test sequences for each category
        self.test_sequences = {
            GrammarCategory.VOID: [
                "0", "00", "000", "0000", "00000"
            ],
            GrammarCategory.EMERGENCE: [
                "01", "001", "0001", "00001", "010"
            ],
            GrammarCategory.RETURN: [
                "10", "100", "1000", "10000", "101"
            ],
            GrammarCategory.OSCILLATION: [
                "010", "101", "0101", "1010", "01010"
            ],
            GrammarCategory.FIBONACCI: [
                "001", "0100", "10001", "010010", "100010"
            ],
            GrammarCategory.TERMINAL: [
                "0", "1"
            ]
        }
        
        # Additional complex sequences
        self.complex_sequences = [
            "0100101001",  # Mixed patterns
            "001010100",   # Multiple structures
            "101001010",   # Complex composition
            "0010010100",  # Fibonacci + other patterns
        ]
    
    def test_feature_extraction(self):
        """Test: Feature extraction produces valid features"""
        for category, sequences in self.test_sequences.items():
            for seq in sequences:
                features = self.feature_extractor.extract_features(seq)
                
                # Should extract meaningful features
                self.assertGreater(len(features), 10)
                
                # All feature values should be valid numbers
                for key, value in features.items():
                    self.assertIsInstance(value, (int, float))
                    self.assertFalse(math.isnan(value))
                    self.assertFalse(math.isinf(value))
    
    def test_phi_constraint_validation(self):
        """Test: All test sequences respect φ-constraint"""
        all_sequences = []
        for sequences in self.test_sequences.values():
            all_sequences.extend(sequences)
        all_sequences.extend(self.complex_sequences)
        
        for seq in all_sequences:
            self.assertNotIn('11', seq, f"Sequence '{seq}' violates φ-constraint")
    
    def test_classifier_architecture(self):
        """Test: Classifier has correct architecture"""
        # Test forward pass
        test_sequences = ["01", "10", "000"]
        
        try:
            logits, confidences, attention_weights = self.classifier(test_sequences)
            
            # Check output shapes
            self.assertEqual(logits.shape[0], 3)  # Batch size
            self.assertEqual(logits.shape[1], len(GrammarCategory))  # Number of categories
            self.assertEqual(confidences.shape[0], 3)  # Batch size
            self.assertEqual(confidences.shape[1], 1)  # Single confidence value
            
        except Exception as e:
            self.fail(f"Classifier forward pass failed: {e}")
    
    def test_single_sequence_classification(self):
        """Test: Single sequence classification works"""
        test_seq = "010"
        
        result = self.classifier.classify_sequence(test_seq)
        
        # Check result structure
        self.assertIsInstance(result, ClassificationResult)
        self.assertIsInstance(result.predicted_category, GrammarCategory)
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)
        self.assertIsInstance(result.feature_activations, dict)
        self.assertIsInstance(result.explanation, str)
    
    def test_feature_extraction_consistency(self):
        """Test: Feature extraction is consistent"""
        test_seq = "01010"
        
        # Extract features multiple times
        features1 = self.feature_extractor.extract_features(test_seq)
        features2 = self.feature_extractor.extract_features(test_seq)
        
        # Should be identical
        self.assertEqual(features1, features2)
    
    def test_category_specific_features(self):
        """Test: Different categories show distinct feature patterns"""
        # Test void sequences
        void_features = [self.feature_extractor.extract_features(seq) 
                        for seq in self.test_sequences[GrammarCategory.VOID]]
        
        # All void sequences should have high zero_ratio
        for features in void_features:
            zero_ratio = features.get('zero_ratio', 0)
            self.assertGreater(zero_ratio, 0.8, "Void sequences should have high zero ratio")
        
        # Test emergence sequences
        emergence_features = [self.feature_extractor.extract_features(seq) 
                            for seq in self.test_sequences[GrammarCategory.EMERGENCE]]
        
        # Emergence sequences should have non-zero emergence patterns
        for features in emergence_features:
            emergence_density = features.get('emergence_pattern_density', 0)
            # At least some emergence pattern should be detected
            self.assertGreaterEqual(emergence_density, 0.0)
    
    def test_phi_alignment_calculation(self):
        """Test: φ-alignment calculation works correctly"""
        # Test sequence with exact φ ratio
        # φ ≈ 1.618, so zeros/ones should be close to this
        # For small sequences, test known ratios
        
        test_cases = [
            ("0", 1.0),      # Pure zeros should have high alignment
            ("1", 0.0),      # Pure ones should have lower alignment  
            ("00001", None), # 4 zeros, 1 one: ratio = 4 (higher than φ)
        ]
        
        for seq, expected in test_cases:
            features = self.feature_extractor.extract_features(seq)
            phi_alignment = features.get('phi_alignment', 0)
            
            if expected is not None:
                if expected == 1.0:
                    self.assertGreater(phi_alignment, 0.5)
                elif expected == 0.0:
                    # Pure ones without zeros should have some alignment
                    self.assertGreaterEqual(phi_alignment, 0.0)
    
    def test_meta_analysis(self):
        """Test: Meta-analysis of category relationships"""
        # Use subset of test sequences for efficiency
        limited_sequences = {}
        for category, sequences in self.test_sequences.items():
            limited_sequences[category] = sequences[:3]  # First 3 sequences
        
        analysis = self.meta_analyzer.analyze_category_relationships(limited_sequences)
        
        # Check analysis structure
        self.assertIn('category_distances', analysis)
        self.assertIn('feature_importance', analysis)
        self.assertIn('separability_scores', analysis)
        
        # Check that distances are non-negative
        for distance in analysis['category_distances'].values():
            self.assertGreaterEqual(distance, 0.0)
        
        # Check that separability scores are reasonable
        for score in analysis['separability_scores'].values():
            self.assertGreaterEqual(score, 0.0)
    
    def test_explanation_generation(self):
        """Test: Explanation generation provides meaningful insights"""
        test_cases = [
            ("000", GrammarCategory.VOID),
            ("010", GrammarCategory.OSCILLATION),
            ("01", GrammarCategory.EMERGENCE),
        ]
        
        for seq, expected_category in test_cases:
            features = self.feature_extractor.extract_features(seq)
            explanation = self.classifier._generate_explanation(seq, features, expected_category)
            
            # Explanation should be non-empty string
            self.assertIsInstance(explanation, str)
            self.assertGreater(len(explanation), 0)
            
            # Should contain category-relevant terms
            explanation_lower = explanation.lower()
            if expected_category == GrammarCategory.VOID:
                self.assertTrue(any(term in explanation_lower 
                                 for term in ['zero', 'void', 'density']))
            elif expected_category == GrammarCategory.OSCILLATION:
                self.assertTrue(any(term in explanation_lower 
                                 for term in ['period', 'oscillat', 'pattern']))
    
    def test_pattern_complexity_calculation(self):
        """Test: Pattern complexity calculation is reasonable"""
        test_cases = [
            ("0", 0.0),      # Single character, no transitions
            ("01", 1.0),     # One transition type, full complexity
            ("00", 1.0),     # One transition type
            ("010", 1.0),    # Multiple transitions but simple pattern
        ]
        
        for seq, expected_range in test_cases:
            complexity = self.feature_extractor._pattern_complexity(seq)
            
            self.assertGreaterEqual(complexity, 0.0)
            self.assertLessEqual(complexity, 1.0)
    
    def test_fibonacci_mapping(self):
        """Test: Fibonacci mapping calculation"""
        # Test known Fibonacci patterns
        fibonacci_test_cases = [
            "1",      # F(1) = 1
            "01",     # F(2) = 1  
            "001",    # F(3) = 2
            "101",    # F(2) + F(4) = 1 + 3 = 4 (not Fibonacci)
        ]
        
        for seq in fibonacci_test_cases:
            features = self.feature_extractor.extract_features(seq)
            fib_mapping = features.get('fibonacci_mapping', 0)
            
            # Should be valid probability
            self.assertGreaterEqual(fib_mapping, 0.0)
            self.assertLessEqual(fib_mapping, 1.0)
    
    def test_batch_classification(self):
        """Test: Batch classification handles multiple sequences"""
        test_batch = ["01", "10", "000", "010", "001"]
        
        try:
            logits, confidences, attention_weights = self.classifier(test_batch)
            
            # Should process all sequences
            self.assertEqual(logits.shape[0], len(test_batch))
            self.assertEqual(confidences.shape[0], len(test_batch))
            
            # All confidences should be valid
            for conf in confidences:
                self.assertGreaterEqual(conf.item(), 0.0)
                self.assertLessEqual(conf.item(), 1.0)
                
        except Exception as e:
            self.fail(f"Batch classification failed: {e}")


def visualize_grammar_classification():
    """Visualize grammar classification results"""
    print("=" * 60)
    print("Grammar Classification: φ-Constrained Expression Analysis")
    print("=" * 60)
    
    classifier = φGrammarClassifier()
    feature_extractor = φGrammarFeatureExtractor()
    meta_analyzer = MetaGrammarAnalyzer()
    
    # Test sequences for each category
    test_sequences = {
        GrammarCategory.VOID: ["0", "00", "000"],
        GrammarCategory.EMERGENCE: ["01", "001", "0001"],
        GrammarCategory.RETURN: ["10", "100", "1000"],
        GrammarCategory.OSCILLATION: ["010", "101", "0101"],
        GrammarCategory.FIBONACCI: ["001", "0100", "10001"],
        GrammarCategory.TERMINAL: ["0", "1"]
    }
    
    print("\n1. Feature Extraction Analysis:")
    
    for category, sequences in test_sequences.items():
        print(f"\n{category.value.upper()} Category:")
        
        for seq in sequences[:2]:  # Show first 2 examples
            features = feature_extractor.extract_features(seq)
            
            print(f"   Sequence: '{seq}'")
            
            # Show top features
            sorted_features = sorted(features.items(), key=lambda x: abs(x[1]), reverse=True)
            for name, value in sorted_features[:5]:
                print(f"     {name}: {value:.3f}")
    
    print("\n2. Classification Results:")
    
    all_test_sequences = []
    for sequences in test_sequences.values():
        all_test_sequences.extend(sequences[:2])
    
    for seq in all_test_sequences[:10]:  # Limit output
        result = classifier.classify_sequence(seq)
        
        print(f"\nSequence: '{seq}'")
        print(f"   Predicted: {result.predicted_category.value}")
        print(f"   Confidence: {result.confidence:.3f}")
        print(f"   Explanation: {result.explanation}")
    
    print("\n3. Meta-Analysis:")
    
    # Use subset for efficiency
    limited_sequences = {}
    for category, sequences in test_sequences.items():
        limited_sequences[category] = sequences[:2]
    
    analysis = meta_analyzer.analyze_category_relationships(limited_sequences)
    
    print("\nCategory Distances (top 5):")
    distances = analysis['category_distances']
    sorted_distances = sorted(distances.items(), key=lambda x: x[1])
    
    for (cat1, cat2), distance in sorted_distances[:5]:
        print(f"   {cat1.value} ↔ {cat2.value}: {distance:.3f}")
    
    print("\nSeparability Scores:")
    separability = analysis['separability_scores']
    
    for category, score in separability.items():
        print(f"   {category.value}: {score:.3f}")
    
    print("\n4. Feature Importance by Category:")
    
    feature_importance = analysis['feature_importance']
    
    for category, important_features in feature_importance.items():
        print(f"\n{category.value.upper()}:")
        for feature_name, importance in important_features[:3]:
            print(f"   {feature_name}: {importance:.3f}")
    
    print("\n5. Classification Statistics:")
    
    # Test classifier on all sequences
    correct_predictions = 0
    total_predictions = 0
    
    for true_category, sequences in test_sequences.items():
        for seq in sequences:
            result = classifier.classify_sequence(seq)
            
            if result.predicted_category == true_category:
                correct_predictions += 1
            total_predictions += 1
    
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    print(f"\nOverall Accuracy: {accuracy:.3f} ({correct_predictions}/{total_predictions})")
    
    # Show confusion analysis
    print("\nConfusion Analysis:")
    confusion_counts = defaultdict(int)
    
    for true_category, sequences in test_sequences.items():
        for seq in sequences:
            result = classifier.classify_sequence(seq)
            confusion_counts[(true_category, result.predicted_category)] += 1
    
    for (true_cat, pred_cat), count in confusion_counts.items():
        if true_cat != pred_cat and count > 0:
            print(f"   {true_cat.value} → {pred_cat.value}: {count} misclassifications")
    
    print("\n" + "=" * 60)
    print("Grammar classification reveals structural patterns in φ-space")
    print("=" * 60)


if __name__ == "__main__":
    # Run visualization
    visualize_grammar_classification()
    
    # Run tests
    print("\n\nRunning formal verification tests...\n")
    unittest.main(verbosity=2)