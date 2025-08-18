from utility.constants import SyntacticMutations, LexicalMutations, LogicalMutations
from typing import List

class MutationNode:
    def __init__(self, mutation):
        self.mutation_name = mutation
        self.children = []

lexical_mutations = [attr for attr in LexicalMutations.__dict__.keys() if not attr.startswith("__")]
lexical_nodes = [MutationNode(mutation = getattr(LexicalMutations, mutation)) for mutation in lexical_mutations]
lexical_node_names = [node.mutation_name for node in lexical_nodes]

syntactic_mutations = [attr for attr in SyntacticMutations.__dict__.keys() if not attr.startswith("__")]
syntactic_nodes = [MutationNode(mutation = getattr(SyntacticMutations, mutation)) for mutation in syntactic_mutations]
syntactic_node_names = [node.mutation_name for node in syntactic_nodes]

logical_mutations = [attr for attr in LogicalMutations.__dict__.keys() if not attr.startswith("__") and ("CONSTANT" not in attr) ]
logical_nodes = [MutationNode(mutation = getattr(LogicalMutations, mutation)) for mutation in logical_mutations]
logical_node_names = [node.mutation_name for node in logical_nodes]

# Note: Constant unfolding nodes were seperated from the other logical nodes as they are conflicting with each other by nature. 
constant_unfolding_mutations = [attr for attr in LogicalMutations.__dict__.keys() if not attr.startswith("__") and ("CONSTANT" in attr) ]
constant_unfolding_nodes = [MutationNode(mutation = getattr(LogicalMutations, mutation)) for mutation in constant_unfolding_mutations]
constant_unfolding_node_names = [node.mutation_name for node in constant_unfolding_nodes]


for node in lexical_nodes:
    node.children.extend(syntactic_node_names)
    node.children.extend(logical_node_names)
    node.children.extend(constant_unfolding_node_names)

for node in syntactic_nodes:
    node.children.extend(lexical_node_names)
    node.children.extend(logical_node_names)
    node.children.extend(constant_unfolding_node_names)

for node in logical_nodes:
    node.children.extend(lexical_node_names)
    node.children.extend(syntactic_node_names)
    node.children.extend(constant_unfolding_node_names)
    node.children.extend([name for name in logical_nodes if name != node.mutation_name])

for node in constant_unfolding_nodes:
    node.children.extend(lexical_node_names)
    node.children.extend(syntactic_node_names)
    node.children.extend(logical_node_names)


MUTATION_NODES = lexical_nodes + syntactic_nodes + logical_nodes + constant_unfolding_nodes

def check_for_mutation_conflicts(
        mutations: List[str], 
        graph: List[MutationNode] = MUTATION_NODES
    ):
    valid_mutations = [mutation for mutation in mutations if mutation is not None]
    
    for mutation_name in valid_mutations:
        target_node = _find_corresponding_mutation_node(mutation_name, graph)
        for other_mutation in valid_mutations:
            if mutation_name == other_mutation:
                continue
            elif other_mutation not in target_node.children:
                print(f"{mutation_name} and {other_mutation} are conflicting mutations and cannot be input together.")
                return False
            
    return True

def _find_corresponding_mutation_node(
        mutation_name: str,
        graph: List[MutationNode]
    ):
    for node in graph:
        if node.mutation_name == mutation_name:
            return node
    
    raise ValueError("Could not find a mutation node corresponding to this mutation name.")