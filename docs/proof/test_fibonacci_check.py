#!/usr/bin/env python3

# Quick test of Fibonacci generation
fib = [1, 1]  # Standard Fibonacci: F_1=1, F_2=1
for i in range(2, 12):
    fib.append(fib[-1] + fib[-2])
print('Standard Fibonacci:', fib)

# Alternative with F_1=1, F_2=2
fib2 = [1, 2]
for i in range(2, 12):
    fib2.append(fib2[-1] + fib2[-2])
print('Alternative Fibonacci:', fib2)

# Test evolution
S_t = {'0', '1'}
sizes = [len(S_t)]
print(f"Initial: {S_t}, size={len(S_t)}")

for i in range(8):
    S_next = set()
    for s in S_t:
        S_next.add(s + '0')
        if not s or s[-1] != '1':
            S_next.add(s + '1')
    S_t = S_next
    sizes.append(len(S_t))
    print(f"Step {i+1}: size={len(S_t)}")

print('\nEvolution sizes:', sizes)
print('Standard Fib (from F_3):', fib[2:2+len(sizes)])
print('Alternative Fib (from F_3):', fib2[2:2+len(sizes)])