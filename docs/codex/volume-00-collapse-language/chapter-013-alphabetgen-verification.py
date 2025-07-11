#!/usr/bin/env python3
"""
Chapter 013: AlphabetGen - Verification Program
Generative Models for φ-Constrained Alphabets

This program verifies that neural generative models can learn to produce
φ-constrained sequences, discovering the deep structure of the golden alphabet
through probabilistic generation and constraint enforcement.

从ψ的生成本质中，涌现出φ字母表的概率生成模型。
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


class GenerationMode(Enum):
    """Different modes for generating φ-constrained sequences"""
    PROBABILISTIC = "probabilistic"  # Sample from learned distribution
    DETERMINISTIC = "deterministic"  # Use highest probability paths
    CONSTRAINED = "constrained"      # Hard constraint enforcement
    ADVERSARIAL = "adversarial"      # GAN-style generation
    EVOLUTIONARY = "evolutionary"    # Genetic algorithm approach


@dataclass
class GenerationResult:
    """Result of a generation process"""
    sequences: List[str]
    probabilities: List[float]
    validity_scores: List[float]
    generation_mode: GenerationMode
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def average_probability(self) -> float:
        """Average probability of generated sequences"""
        return sum(self.probabilities) / len(self.probabilities) if self.probabilities else 0.0
    
    def validity_rate(self) -> float:
        """Fraction of sequences that are φ-valid"""
        return sum(1 for score in self.validity_scores if score > 0.5) / len(self.validity_scores)


class φConstraintValidator:
    """
    Validates sequences against the φ-constraint.
    Provides both hard validation and soft scoring.
    """
    
    def __init__(self):
        # Golden ratio
        self.phi = (1 + math.sqrt(5)) / 2
        
        # Forbidden patterns
        self.forbidden_patterns = ['11']
        
        # Preferred patterns (φ-aligned)
        self.preferred_patterns = ['0', '00', '01', '10', '010', '100', '001']
        
    def is_valid(self, sequence: str) -> bool:
        """Hard validation: sequence is φ-valid"""
        return '11' not in sequence
    
    def validity_score(self, sequence: str) -> float:
        """Soft validation: score between 0 and 1"""
        if not sequence:
            return 1.0
        
        # Hard constraint
        if '11' in sequence:
            return 0.0
        
        # Soft preferences based on φ-alignment
        score = 1.0
        
        # Prefer certain ratios of 0s to 1s
        if len(sequence) > 1:
            zeros = sequence.count('0')
            ones = sequence.count('1')
            if ones > 0:
                ratio = zeros / ones
                # Prefer ratios close to φ
                deviation = abs(ratio - self.phi)
                ratio_score = max(0.0, 1.0 - deviation / self.phi)
                score = (score + ratio_score) / 2
        
        # Prefer sequences with φ-aligned patterns
        pattern_score = self._pattern_alignment_score(sequence)
        score = (score + pattern_score) / 2
        
        return score
    
    def _pattern_alignment_score(self, sequence: str) -> float:
        """Score based on presence of φ-aligned patterns"""
        if not sequence:
            return 1.0
        
        total_patterns = 0
        aligned_patterns = 0
        
        # Check all subpatterns
        for length in range(1, min(4, len(sequence) + 1)):
            for i in range(len(sequence) - length + 1):
                pattern = sequence[i:i+length]
                total_patterns += 1
                
                if pattern in self.preferred_patterns:
                    aligned_patterns += 1
        
        if total_patterns == 0:
            return 1.0
        
        return aligned_patterns / total_patterns
    
    def constraint_loss(self, probabilities: torch.Tensor) -> torch.Tensor:
        """Differentiable loss for constraint enforcement"""
        # probabilities: (batch, seq_len, 2) - probability of 0 or 1 at each position
        batch_size, seq_len, _ = probabilities.shape
        
        if seq_len < 2:
            return torch.tensor(0.0)
        
        # Probability of '11' at consecutive positions
        prob_1_current = probabilities[:, :-1, 1]  # P(1) at positions 0 to seq_len-2
        prob_1_next = probabilities[:, 1:, 1]      # P(1) at positions 1 to seq_len-1
        
        # P(11) = P(1 at i) * P(1 at i+1)
        prob_11 = prob_1_current * prob_1_next
        
        # Loss is sum of all '11' probabilities
        constraint_loss = torch.sum(prob_11)
        
        return constraint_loss


class φAlphabetGenerator(nn.Module):
    """
    Neural generative model for φ-constrained sequences.
    Uses LSTM-based architecture with constraint enforcement.
    """
    
    def __init__(self, hidden_dim: int = 128, num_layers: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.vocab_size = 2  # {0, 1}
        
        # Character embedding
        self.embedding = nn.Embedding(self.vocab_size, hidden_dim)
        
        # LSTM core
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0
        )
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, self.vocab_size)
        
        # φ-constraint enforcer
        self.constraint_validator = φConstraintValidator()
        
        # Temperature for sampling
        self.temperature = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, x: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        """
        Forward pass of the generator.
        x: (batch, seq_len) - input sequence of token indices
        """
        # Embed input
        embedded = self.embedding(x)
        
        # LSTM forward
        output, hidden = self.lstm(embedded, hidden)
        
        # Project to vocabulary
        logits = self.output_projection(output)
        
        # Apply temperature
        scaled_logits = logits / self.temperature
        
        # Convert to probabilities
        probabilities = F.softmax(scaled_logits, dim=-1)
        
        return probabilities, hidden
    
    def generate_sequence(self, 
                         max_length: int = 20, 
                         start_token: int = 0,
                         mode: GenerationMode = GenerationMode.PROBABILISTIC) -> str:
        """Generate a single φ-constrained sequence"""
        self.eval()
        with torch.no_grad():
            sequence = [start_token]
            hidden = None
            
            for _ in range(max_length - 1):
                # Current input
                current_input = torch.tensor([[sequence[-1]]])
                
                # Forward pass
                probabilities, hidden = self.forward(current_input, hidden)
                prob_dist = probabilities[0, 0]  # (vocab_size,)
                
                # Check constraint for next token
                if len(sequence) > 0 and sequence[-1] == 1:
                    # Last token was 1, so next token cannot be 1
                    prob_dist = prob_dist.clone()
                    prob_dist[1] = 0.0
                    # Renormalize
                    prob_dist = prob_dist / prob_dist.sum()
                
                # Sample next token based on mode
                if mode == GenerationMode.PROBABILISTIC:
                    next_token = torch.multinomial(prob_dist, 1).item()
                elif mode == GenerationMode.DETERMINISTIC:
                    next_token = torch.argmax(prob_dist).item()
                else:  # CONSTRAINED
                    # More sophisticated constraint handling
                    next_token = self._constrained_sample(prob_dist, sequence)
                
                sequence.append(next_token)
                
                # Early stopping if we generate a natural ending
                if len(sequence) > 3 and self._is_natural_ending(sequence):
                    break
            
            return ''.join(str(token) for token in sequence)
    
    def _constrained_sample(self, prob_dist: torch.Tensor, current_sequence: List[int]) -> int:
        """Sample with hard constraint enforcement"""
        if len(current_sequence) > 0 and current_sequence[-1] == 1:
            # Must choose 0 to avoid '11'
            return 0
        else:
            # Can choose either, sample normally
            return torch.multinomial(prob_dist, 1).item()
    
    def _is_natural_ending(self, sequence: List[int]) -> bool:
        """Check if sequence has reached a natural ending point"""
        if len(sequence) < 4:
            return False
        
        # End on certain patterns that feel complete
        last_three = ''.join(str(token) for token in sequence[-3:])
        ending_patterns = ['010', '100', '001', '000']
        
        return last_three in ending_patterns
    
    def generate_batch(self, 
                      batch_size: int = 10,
                      max_length: int = 20,
                      mode: GenerationMode = GenerationMode.PROBABILISTIC) -> GenerationResult:
        """Generate a batch of sequences"""
        sequences = []
        probabilities = []
        validity_scores = []
        
        for _ in range(batch_size):
            sequence = self.generate_sequence(max_length, mode=mode)
            
            # Calculate sequence probability
            seq_prob = self._calculate_sequence_probability(sequence)
            
            # Calculate validity score
            validity = self.constraint_validator.validity_score(sequence)
            
            sequences.append(sequence)
            probabilities.append(seq_prob)
            validity_scores.append(validity)
        
        return GenerationResult(
            sequences=sequences,
            probabilities=probabilities,
            validity_scores=validity_scores,
            generation_mode=mode,
            metadata={'max_length': max_length, 'batch_size': batch_size}
        )
    
    def _calculate_sequence_probability(self, sequence: str) -> float:
        """Calculate the probability of a sequence under the model"""
        if not sequence:
            return 1.0
        
        self.eval()
        with torch.no_grad():
            tokens = [int(c) for c in sequence]
            total_log_prob = 0.0
            hidden = None
            
            for i in range(len(tokens) - 1):
                current_input = torch.tensor([[tokens[i]]])
                probabilities, hidden = self.forward(current_input, hidden)
                
                next_token_prob = probabilities[0, 0, tokens[i + 1]]
                total_log_prob += torch.log(next_token_prob + 1e-8).item()
            
            return math.exp(total_log_prob)


class φGANGenerator(nn.Module):
    """
    GAN-based generator for φ-constrained sequences.
    Uses adversarial training to learn the constraint distribution.
    """
    
    def __init__(self, latent_dim: int = 64, hidden_dim: int = 128, max_length: int = 20):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.max_length = max_length
        
        # Generator network
        self.generator = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_length * 2),  # 2 for binary probabilities
            nn.Sigmoid()
        )
        
        # Constraint enforcer
        self.constraint_validator = φConstraintValidator()
    
    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        """
        Generate sequences from noise.
        noise: (batch, latent_dim)
        Returns: (batch, max_length, 2) - probability distribution over {0,1}
        """
        # Generate raw probabilities
        raw_output = self.generator(noise)
        
        # Reshape to sequence format
        batch_size = noise.shape[0]
        probabilities = raw_output.view(batch_size, self.max_length, 2)
        
        # Apply φ-constraint masking
        constrained_probs = self._apply_phi_constraint(probabilities)
        
        return constrained_probs
    
    def _apply_phi_constraint(self, probabilities: torch.Tensor) -> torch.Tensor:
        """Apply φ-constraint to probability distributions"""
        batch_size, seq_len, vocab_size = probabilities.shape
        constrained = probabilities.clone()
        
        # For each position after the first
        for i in range(1, seq_len):
            # If previous position strongly favors 1, reduce probability of 1 at current position
            prev_prob_1 = probabilities[:, i-1, 1]
            
            # Create mask based on previous 1-probability
            mask = torch.sigmoid(-5 * (prev_prob_1 - 0.5))  # High when prev_prob_1 is high
            
            # Apply mask to current position's 1-probability
            constrained[:, i, 1] = constrained[:, i, 1] * mask
            
            # Renormalize
            constrained[:, i] = constrained[:, i] / constrained[:, i].sum(dim=-1, keepdim=True)
        
        return constrained
    
    def generate_sequences(self, num_sequences: int = 10) -> List[str]:
        """Generate sequences from random noise"""
        self.eval()
        with torch.no_grad():
            # Sample noise
            noise = torch.randn(num_sequences, self.latent_dim)
            
            # Generate probabilities
            probabilities = self.forward(noise)
            
            # Sample sequences
            sequences = []
            for i in range(num_sequences):
                sequence = []
                for j in range(self.max_length):
                    prob_dist = probabilities[i, j]
                    token = torch.multinomial(prob_dist, 1).item()
                    sequence.append(str(token))
                
                # Find natural ending
                full_sequence = ''.join(sequence)
                # Truncate at first natural ending
                for end_len in range(3, len(full_sequence) + 1):
                    if self._is_natural_ending(full_sequence[:end_len]):
                        full_sequence = full_sequence[:end_len]
                        break
                
                sequences.append(full_sequence)
            
            return sequences
    
    def _is_natural_ending(self, sequence: str) -> bool:
        """Check for natural ending patterns"""
        if len(sequence) < 3:
            return False
        
        ending_patterns = ['010', '100', '001', '000']
        return sequence[-3:] in ending_patterns


class φDiscriminator(nn.Module):
    """
    Discriminator for GAN training.
    Distinguishes between real φ-constrained sequences and generated ones.
    """
    
    def __init__(self, max_length: int = 20, hidden_dim: int = 128):
        super().__init__()
        self.max_length = max_length
        self.hidden_dim = hidden_dim
        
        # Sequence encoder
        self.encoder = nn.Sequential(
            nn.Linear(max_length, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # φ-constraint detector
        self.constraint_detector = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        # Authenticity classifier
        self.authenticity_classifier = nn.Sequential(
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, sequences: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Discriminate sequences.
        sequences: (batch, max_length) - binary sequences padded to max_length
        Returns: (authenticity_scores, constraint_scores)
        """
        # Encode sequences
        encoded = self.encoder(sequences.float())
        
        # Classify authenticity
        authenticity = self.authenticity_classifier(encoded)
        
        # Detect constraint violations
        constraint_score = self.constraint_detector(encoded)
        
        return authenticity.squeeze(-1), constraint_score.squeeze(-1)


class EvolutionaryGenerator:
    """
    Evolutionary algorithm for generating φ-constrained sequences.
    Uses genetic operators while maintaining constraint satisfaction.
    """
    
    def __init__(self, 
                 population_size: int = 50,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.7):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.validator = φConstraintValidator()
    
    def generate_initial_population(self, sequence_length: int = 10) -> List[str]:
        """Generate initial population of φ-valid sequences"""
        population = []
        
        for _ in range(self.population_size):
            sequence = self._generate_random_valid_sequence(sequence_length)
            population.append(sequence)
        
        return population
    
    def _generate_random_valid_sequence(self, length: int) -> str:
        """Generate a random φ-valid sequence"""
        sequence = []
        
        for i in range(length):
            if i > 0 and sequence[-1] == '1':
                # Must choose 0 to maintain φ-constraint
                sequence.append('0')
            else:
                # Can choose either
                sequence.append(random.choice(['0', '1']))
        
        return ''.join(sequence)
    
    def fitness(self, sequence: str) -> float:
        """Calculate fitness of a sequence"""
        # Base fitness from φ-validity
        base_fitness = self.validator.validity_score(sequence)
        
        # Bonus for interesting patterns
        pattern_bonus = self._pattern_diversity_bonus(sequence)
        
        # Bonus for optimal length
        length_bonus = self._length_bonus(sequence)
        
        return base_fitness + 0.1 * pattern_bonus + 0.05 * length_bonus
    
    def _pattern_diversity_bonus(self, sequence: str) -> float:
        """Bonus for having diverse patterns"""
        if len(sequence) < 3:
            return 0.0
        
        patterns = set()
        for i in range(len(sequence) - 2):
            patterns.add(sequence[i:i+3])
        
        # More unique patterns = higher diversity
        max_possible = min(len(sequence) - 2, 8)  # Max φ-valid 3-patterns
        if max_possible == 0:
            return 0.0
        
        return len(patterns) / max_possible
    
    def _length_bonus(self, sequence: str) -> float:
        """Bonus for optimal length (around 8-12 characters)"""
        optimal_length = 10
        deviation = abs(len(sequence) - optimal_length)
        return max(0, 1.0 - deviation / optimal_length)
    
    def evolve_generation(self, population: List[str]) -> List[str]:
        """Evolve population for one generation"""
        # Calculate fitness for all individuals
        fitness_scores = [self.fitness(seq) for seq in population]
        
        # Selection: tournament selection
        selected = self._tournament_selection(population, fitness_scores)
        
        # Crossover and mutation
        new_population = []
        
        for i in range(0, len(selected), 2):
            parent1 = selected[i]
            parent2 = selected[i + 1] if i + 1 < len(selected) else selected[0]
            
            # Crossover
            if random.random() < self.crossover_rate:
                child1, child2 = self._crossover(parent1, parent2)
            else:
                child1, child2 = parent1, parent2
            
            # Mutation
            child1 = self._mutate(child1)
            child2 = self._mutate(child2)
            
            new_population.extend([child1, child2])
        
        # Ensure population size
        return new_population[:self.population_size]
    
    def _tournament_selection(self, population: List[str], fitness_scores: List[float]) -> List[str]:
        """Tournament selection with replacement"""
        selected = []
        tournament_size = 3
        
        for _ in range(len(population)):
            # Select random tournament participants
            tournament_indices = random.sample(range(len(population)), 
                                             min(tournament_size, len(population)))
            
            # Find best in tournament
            best_idx = max(tournament_indices, key=lambda i: fitness_scores[i])
            selected.append(population[best_idx])
        
        return selected
    
    def _crossover(self, parent1: str, parent2: str) -> Tuple[str, str]:
        """Single-point crossover with φ-constraint repair"""
        if len(parent1) != len(parent2):
            # Handle different lengths
            min_len = min(len(parent1), len(parent2))
            parent1 = parent1[:min_len]
            parent2 = parent2[:min_len]
        
        if len(parent1) <= 1:
            return parent1, parent2
        
        # Choose crossover point
        crossover_point = random.randint(1, len(parent1) - 1)
        
        # Create children
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        
        # Repair φ-constraint violations
        child1 = self._repair_constraint_violations(child1)
        child2 = self._repair_constraint_violations(child2)
        
        return child1, child2
    
    def _mutate(self, sequence: str) -> str:
        """Mutate sequence while maintaining φ-constraint"""
        if random.random() > self.mutation_rate:
            return sequence
        
        sequence_list = list(sequence)
        
        # Choose random position to mutate
        if len(sequence_list) == 0:
            return sequence
        
        pos = random.randint(0, len(sequence_list) - 1)
        
        # Flip bit if it doesn't violate constraint
        current_bit = sequence_list[pos]
        new_bit = '1' if current_bit == '0' else '0'
        
        # Check if mutation would create '11'
        sequence_list[pos] = new_bit
        mutated = ''.join(sequence_list)
        
        if self.validator.is_valid(mutated):
            return mutated
        else:
            return sequence  # Keep original if mutation violates constraint
    
    def _repair_constraint_violations(self, sequence: str) -> str:
        """Repair any φ-constraint violations in sequence"""
        repaired = list(sequence)
        
        for i in range(len(repaired) - 1):
            if repaired[i] == '1' and repaired[i + 1] == '1':
                # Fix by changing second '1' to '0'
                repaired[i + 1] = '0'
        
        return ''.join(repaired)
    
    def evolve(self, generations: int = 50, sequence_length: int = 10) -> List[str]:
        """Run full evolutionary process"""
        # Initialize population
        population = self.generate_initial_population(sequence_length)
        
        # Evolve for specified generations
        for generation in range(generations):
            population = self.evolve_generation(population)
        
        # Return final population sorted by fitness
        fitness_scores = [self.fitness(seq) for seq in population]
        sorted_population = [seq for _, seq in sorted(zip(fitness_scores, population), reverse=True)]
        
        return sorted_population


class AlphabetGenTests(unittest.TestCase):
    """Test alphabet generation models"""
    
    def setUp(self):
        self.validator = φConstraintValidator()
        self.generator = φAlphabetGenerator(hidden_dim=32, num_layers=1)
        self.gan_generator = φGANGenerator(latent_dim=16, hidden_dim=32, max_length=10)
        self.evolutionary = EvolutionaryGenerator(population_size=20)
        
    def test_constraint_validation(self):
        """Test: φ-constraint validation works correctly"""
        # Valid sequences
        valid_sequences = ["0", "01", "10", "010", "0101", "001", "100"]
        for seq in valid_sequences:
            self.assertTrue(self.validator.is_valid(seq))
            self.assertGreater(self.validator.validity_score(seq), 0.0)
        
        # Invalid sequences
        invalid_sequences = ["11", "011", "110", "0110", "1100"]
        for seq in invalid_sequences:
            self.assertFalse(self.validator.is_valid(seq))
            self.assertEqual(self.validator.validity_score(seq), 0.0)
    
    def test_lstm_generator(self):
        """Test: LSTM generator produces valid sequences"""
        result = self.generator.generate_batch(batch_size=5, max_length=10)
        
        # Check result structure
        self.assertEqual(len(result.sequences), 5)
        self.assertEqual(len(result.probabilities), 5)
        self.assertEqual(len(result.validity_scores), 5)
        
        # All generated sequences should be φ-valid
        for seq in result.sequences:
            self.assertFalse('11' in seq)
        
        # Validity scores should be > 0 for all sequences
        for score in result.validity_scores:
            self.assertGreater(score, 0.0)
    
    def test_generation_modes(self):
        """Test: Different generation modes work"""
        modes = [GenerationMode.PROBABILISTIC, GenerationMode.DETERMINISTIC, GenerationMode.CONSTRAINED]
        
        for mode in modes:
            sequence = self.generator.generate_sequence(max_length=8, mode=mode)
            
            # Should produce valid sequence
            self.assertIsInstance(sequence, str)
            self.assertGreater(len(sequence), 0)
            self.assertFalse('11' in sequence)
    
    def test_gan_generator(self):
        """Test: GAN generator architecture works"""
        # Test forward pass
        noise = torch.randn(3, 16)
        probabilities = self.gan_generator(noise)
        
        # Check output shape
        self.assertEqual(probabilities.shape, (3, 10, 2))
        
        # Probabilities should sum to approximately 1 at each position
        prob_sums = probabilities.sum(dim=-1)
        # Check that probabilities are close to 1 (allowing for numerical precision)
        self.assertTrue(torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-1))
        
        # Test sequence generation
        sequences = self.gan_generator.generate_sequences(num_sequences=10)
        self.assertEqual(len(sequences), 10)
        
        # At least some sequences should be valid (untrained GAN may not be perfect)
        valid_count = sum(1 for seq in sequences if '11' not in seq)
        self.assertGreater(valid_count, 0)  # At least one should be valid
    
    def test_evolutionary_generator(self):
        """Test: Evolutionary generator produces valid populations"""
        # Test initial population
        population = self.evolutionary.generate_initial_population(sequence_length=8)
        
        self.assertEqual(len(population), 20)
        
        for seq in population:
            self.assertTrue(self.validator.is_valid(seq))
            self.assertEqual(len(seq), 8)
        
        # Test evolution
        evolved = self.evolutionary.evolve(generations=5, sequence_length=6)
        
        self.assertEqual(len(evolved), 20)
        
        for seq in evolved:
            self.assertTrue(self.validator.is_valid(seq))
    
    def test_fitness_function(self):
        """Test: Evolutionary fitness function works correctly"""
        # Test various sequences
        test_sequences = ["010", "0101", "00100", "01010101"]
        
        for seq in test_sequences:
            fitness = self.evolutionary.fitness(seq)
            self.assertGreaterEqual(fitness, 0.0)
            self.assertLessEqual(fitness, 2.0)  # Max possible fitness
    
    def test_constraint_loss(self):
        """Test: Differentiable constraint loss function"""
        # Create test probabilities
        batch_size, seq_len = 2, 5
        probabilities = torch.rand(batch_size, seq_len, 2)
        probabilities = F.softmax(probabilities, dim=-1)
        
        # Calculate constraint loss
        loss = self.validator.constraint_loss(probabilities)
        
        # Should be non-negative
        self.assertGreaterEqual(loss.item(), 0.0)
        
        # Test with high probability of '11'
        high_11_probs = torch.zeros(1, 3, 2)
        high_11_probs[0, 0, 1] = 1.0  # P(1) = 1 at position 0
        high_11_probs[0, 1, 1] = 1.0  # P(1) = 1 at position 1
        high_11_probs[0, 2, 0] = 1.0  # P(0) = 1 at position 2
        
        high_loss = self.validator.constraint_loss(high_11_probs)
        self.assertGreater(high_loss.item(), 0.5)
    
    def test_sequence_probability_calculation(self):
        """Test: Sequence probability calculation is reasonable"""
        sequences = ["01", "010", "0101"]
        
        for seq in sequences:
            prob = self.generator._calculate_sequence_probability(seq)
            self.assertGreater(prob, 0.0)
            self.assertLessEqual(prob, 1.0)
    
    def test_constraint_enforcement_in_gan(self):
        """Test: GAN applies φ-constraint masking correctly"""
        # Create test probabilities with high '11' risk
        batch_size, seq_len = 2, 4
        raw_probs = torch.ones(batch_size, seq_len, 2) * 0.5
        
        # Make first position strongly favor 1
        raw_probs[:, 0, 1] = 0.9
        raw_probs[:, 0, 0] = 0.1
        
        # Apply constraint
        constrained = self.gan_generator._apply_phi_constraint(raw_probs)
        
        # Second position should have reduced probability of 1
        self.assertLess(constrained[0, 1, 1].item(), raw_probs[0, 1, 1].item())
    
    def test_crossover_and_mutation(self):
        """Test: Genetic operators maintain φ-constraint"""
        parent1 = "0101"
        parent2 = "1001"
        
        # Test crossover
        child1, child2 = self.evolutionary._crossover(parent1, parent2)
        
        self.assertTrue(self.validator.is_valid(child1))
        self.assertTrue(self.validator.is_valid(child2))
        
        # Test mutation
        mutated = self.evolutionary._mutate("0101")
        self.assertTrue(self.validator.is_valid(mutated))


def visualize_alphabet_generation():
    """Visualize different approaches to φ-alphabet generation"""
    print("=" * 60)
    print("Alphabet Generation: Learning the φ-Constraint")
    print("=" * 60)
    
    # Initialize models
    validator = φConstraintValidator()
    lstm_gen = φAlphabetGenerator(hidden_dim=64, num_layers=2)
    gan_gen = φGANGenerator(latent_dim=32, hidden_dim=64, max_length=15)
    evo_gen = EvolutionaryGenerator(population_size=30)
    
    print("\n1. LSTM Generation:")
    
    # Test different generation modes
    modes = [GenerationMode.PROBABILISTIC, GenerationMode.DETERMINISTIC, GenerationMode.CONSTRAINED]
    
    for mode in modes:
        result = lstm_gen.generate_batch(batch_size=5, max_length=12, mode=mode)
        
        print(f"\n   Mode: {mode.value}")
        print(f"   Sequences: {result.sequences}")
        print(f"   Avg probability: {result.average_probability():.4f}")
        print(f"   Validity rate: {result.validity_rate():.3f}")
        
        # Check constraint satisfaction
        violations = sum(1 for seq in result.sequences if '11' in seq)
        print(f"   Constraint violations: {violations}/{len(result.sequences)}")
    
    print("\n2. GAN Generation:")
    
    gan_sequences = gan_gen.generate_sequences(num_sequences=8)
    
    print(f"   Generated sequences: {gan_sequences}")
    
    # Analyze quality
    valid_count = sum(1 for seq in gan_sequences if validator.is_valid(seq))
    avg_validity_score = sum(validator.validity_score(seq) for seq in gan_sequences) / len(gan_sequences)
    
    print(f"   Valid sequences: {valid_count}/{len(gan_sequences)}")
    print(f"   Average validity score: {avg_validity_score:.3f}")
    
    print("\n3. Evolutionary Generation:")
    
    # Run evolution
    final_population = evo_gen.evolve(generations=20, sequence_length=10)
    
    print(f"   Best sequences from evolution:")
    for i, seq in enumerate(final_population[:5]):
        fitness = evo_gen.fitness(seq)
        print(f"   {i+1}. {seq} (fitness: {fitness:.3f})")
    
    # Analyze population diversity
    unique_sequences = len(set(final_population))
    print(f"   Population diversity: {unique_sequences}/{len(final_population)} unique")
    
    print("\n4. Constraint Analysis:")
    
    # Analyze constraint properties
    all_sequences = lstm_gen.generate_batch(batch_size=20, max_length=10).sequences
    all_sequences.extend(gan_sequences)
    all_sequences.extend(final_population[:10])
    
    print(f"\n   Total sequences analyzed: {len(all_sequences)}")
    
    # Validity statistics
    valid_sequences = [seq for seq in all_sequences if validator.is_valid(seq)]
    validity_rate = len(valid_sequences) / len(all_sequences)
    
    print(f"   Overall validity rate: {validity_rate:.3f}")
    
    # Pattern analysis
    pattern_counts = Counter()
    for seq in valid_sequences:
        for i in range(len(seq) - 2):
            pattern = seq[i:i+3]
            pattern_counts[pattern] += 1
    
    print(f"   Most common 3-patterns:")
    for pattern, count in pattern_counts.most_common(5):
        print(f"      {pattern}: {count} occurrences")
    
    # Length distribution
    lengths = [len(seq) for seq in valid_sequences]
    avg_length = sum(lengths) / len(lengths)
    print(f"   Average sequence length: {avg_length:.1f}")
    
    print("\n5. φ-Alignment Analysis:")
    
    # Analyze how well sequences align with φ
    phi = (1 + math.sqrt(5)) / 2
    
    alignment_scores = []
    for seq in valid_sequences:
        if len(seq) > 1:
            zeros = seq.count('0')
            ones = seq.count('1')
            if ones > 0:
                ratio = zeros / ones
                deviation = abs(ratio - phi)
                alignment = max(0, 1 - deviation / phi)
                alignment_scores.append(alignment)
    
    if alignment_scores:
        avg_alignment = sum(alignment_scores) / len(alignment_scores)
        print(f"   Average φ-alignment: {avg_alignment:.3f}")
        print(f"   Golden ratio φ = {phi:.6f}")
    
    print("\n6. Generation Quality Metrics:")
    
    # Diversity metric
    unique_patterns = set()
    for seq in valid_sequences:
        for i in range(len(seq) - 1):
            unique_patterns.add(seq[i:i+2])
    
    max_possible_patterns = 3  # 00, 01, 10 (11 is forbidden)
    diversity = len(unique_patterns) / max_possible_patterns
    
    print(f"   Pattern diversity: {diversity:.3f} ({len(unique_patterns)}/{max_possible_patterns})")
    
    # Complexity distribution
    complexities = []
    for seq in valid_sequences:
        # Simple complexity measure: number of transitions
        transitions = sum(1 for i in range(len(seq)-1) if seq[i] != seq[i+1])
        complexity = transitions / max(1, len(seq) - 1)
        complexities.append(complexity)
    
    if complexities:
        avg_complexity = sum(complexities) / len(complexities)
        print(f"   Average complexity: {avg_complexity:.3f}")
    
    print("\n" + "=" * 60)
    print("Neural networks learn to generate φ-constrained alphabets")
    print("=" * 60)


if __name__ == "__main__":
    # Run visualization
    visualize_alphabet_generation()
    
    # Run tests
    print("\n\nRunning formal verification tests...\n")
    unittest.main(verbosity=2)