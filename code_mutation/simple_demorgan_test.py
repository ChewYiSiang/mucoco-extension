#!/usr/bin/env python3

import ast
from ast_mutation import ASTNodeTransformers

def test_demorgan_direct():
    """Test De Morgan's transformer directly."""
    
    # Test case 1: Simple boolean AND
    code1 = "a and b"
    tree1 = ast.parse(code1, mode='eval')
    transformer = ASTNodeTransformers.DeMorganTransformer()
    mutated1 = transformer.visit(tree1)
    ast.fix_missing_locations(mutated1)
    result1 = ast.unparse(mutated1)
    print(f"Original: {code1}")
    print(f"Mutated:  {result1}")
    print()
    
    # Test case 2: Simple boolean OR
    code2 = "x or y"
    tree2 = ast.parse(code2, mode='eval')
    mutated2 = transformer.visit(tree2)
    ast.fix_missing_locations(mutated2)
    result2 = ast.unparse(mutated2)
    print(f"Original: {code2}")
    print(f"Mutated:  {result2}")
    print()
    
    # Test case 3: Negated AND
    code3 = "not (p and q)"
    tree3 = ast.parse(code3, mode='eval')
    mutated3 = transformer.visit(tree3)
    ast.fix_missing_locations(mutated3)
    result3 = ast.unparse(mutated3)
    print(f"Original: {code3}")
    print(f"Mutated:  {result3}")
    print()
    
    # Test case 4: Negated OR
    code4 = "not (m or n)"
    tree4 = ast.parse(code4, mode='eval')
    mutated4 = transformer.visit(tree4)
    ast.fix_missing_locations(mutated4)
    result4 = ast.unparse(mutated4)
    print(f"Original: {code4}")
    print(f"Mutated:  {result4}")
    print()
    
    # Test functional equivalence
    print("=== Functional Equivalence Tests ===")
    test_values = [(True, True), (True, False), (False, True), (False, False)]
    
    for a, b in test_values:
        # Test case 1: a and b vs not ((not a) or (not b))
        original = a and b
        mutated = not ((not a) or (not b))
        print(f"a={a}, b={b}: {original} == {mutated} -> {original == mutated}")

if __name__ == "__main__":
    test_demorgan_direct()