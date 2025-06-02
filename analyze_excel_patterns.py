#!/usr/bin/env python3
"""
Analyze Excel patterns to understand the limit setting logic better
"""

# Excel results analysis
excel_results = {
    'sec méca - a/b': (1.00, 0.81, 0.00),      # TypeLissage: Baisse (limite basse product < 1)
    'sec méca - a/c': (1.00, 1.00, 0.92),      # TypeLissage: Hausse 
    'sec homogène - a/b': (1.00, 1.00, 1.00),  # TypeLissage: Hausse
    'sec homogène - a/c': (1.95, 1.95, 1.95),  # TypeLissage: Hausse
    'sec hétérogène - a/b': (1.06, 1.06, 1.06), # TypeLissage: Hausse
    'sec hétérogène - a/c': (1.00, 1.00, 1.00), # TypeLissage: Hausse
    'frais méca': (1.00, 1.00, 1.00),          # TypeLissage: Hausse
    'frais manuel': (1.00, 1.00, 0.54),        # TypeLissage: Baisse (Autre gets optimized down)
    'surgelés': (1.04, 1.04, 1.04)             # TypeLissage: Hausse
}

print("Excel Pattern Analysis:")
print("=======================")

for product, (j, k, l) in excel_results.items():
    is_baisse = (j * k * l) < 1.0  # Approximation based on pattern
    print(f"{product:20} | J={j:4.2f} K={k:4.2f} L={l:4.2f} | Pattern: {'Baisse' if is_baisse else 'Hausse'}")
    
    # Analyze patterns
    if is_baisse:
        if l == 0.00:
            print(f"                     | Baisse Pattern: Autre=0, others varied")
        elif j == k == 1.00 and l < 1.00:
            print(f"                     | Baisse Pattern: Top500=Top3000=1, Autre optimized down")
        elif j == 1.00 and k < 1.00 and l == 0.00:
            print(f"                     | Baisse Pattern: Top500=1, Top3000 optimized, Autre=0")
    else:
        if j == k == l:
            print(f"                     | Hausse Pattern: All equal (J=K=L)")
        else:
            print(f"                     | Hausse Pattern: Varied (unusual)")

print("\nKey insights:")
print("1. Baisse mode can set different values for J, K, L")
print("2. Pattern (1.00, 0.81, 0.00) suggests Top500=max, Top3000=optimized, Autre=min")
print("3. Pattern (1.00, 1.00, 0.54) suggests Top500=Top3000=max, Autre=optimized")
print("4. This indicates more sophisticated limit/constraint handling than our current implementation")