import sys
import itertools
import inspect
import ast
import copy


class Node:
    def __init__(self, problem, assignment, parent=None):
        self.problem = problem
        self.assignment = assignment
        self.parent = parent


def generate_problem(N, P):
    # P1: N-Queens
    if P == 'P1':
        # 1. One queen per row: Each queen must be placed in a different row.
        # 2. One queen per column: Each queen must be placed in a different column.
        # 3. No two queens can be on the same diagonal:
            # Queens on positions (r1, c1) and (r2, c2) should not satisfy |r1 - r2| = |c1 - c2|. 
            # Queens should not satisfy |r1 - r2| = |c2 - c1|,
        keys = [f'Q{i+1}-Q{j+1}' for i, j in itertools.combinations(range(N), 2)]
        values = [lambda Qa, Qb, i=i, j=j: Qa != Qb and abs(Qa - Qb) != abs(i - j) for i, j in itertools.combinations(range(N), 2)]
        constraints = {}
        for i in range(len(keys)):
            constraints[keys[i]] = values[i]
        return {
            'variables': [f'Q{i+1}' for i in range(N)],                           # indexes handle rows
            'domains': {f'Q{i+1}': [j+1 for j in range(N)] for i in range(N)},    # domains handle columns
            'constraints': constraints                                            # constraints handle diagonals
        }
    
    # P2: Map Coloring
    elif P == 'P2':
        variables = ['WA', 'NT', 'SA', 'Q', 'NSW', 'V', 'T']
        colors = [f'c{i+1}' for i in range(N)]
        return {
            'variables': variables,
            'domains': {v: colors for v in variables},
            'constraints': {
                'WA-NT': lambda WA, NT: WA != NT,
                'WA-SA': lambda WA, SA: WA != SA,
                'NT-SA': lambda NT, SA: NT != SA,
                'NT-Q': lambda NT, Q: NT != Q,
                'SA-Q': lambda SA, Q: SA != Q,
                'SA-NSW': lambda SA, NSW: SA != NSW,
                'SA-V': lambda SA, V: SA != V,
                'Q-NSW': lambda Q, NSW: Q != NSW,
                'NSW-V': lambda NSW, V: NSW != V,
            }
        }
    
    # P3: Cryptarithmetic
    elif P == 'P3':
        if N == 0:      # TO + TO = FOR
            return {
                'variables': ['T', 'O', 'F', 'R'],
                'domains': {
                    'T': [1, 2, 3, 4, 5, 6, 7, 8, 9],
                    'O': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                    'F': [0, 1],
                    'R': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                    'C1': [0, 1],
                },
                'constraints': {
                    'T-O': lambda T, O: T != O,
                    'T-F': lambda T, F: T != F,
                    'T-R': lambda T, R: T != R,
                    'O-F': lambda O, F: O != F,
                    'O-R': lambda O, R: O != R,
                    'F-R': lambda F, R: F != R,
                    'T-O-F-R': lambda T, O, F, R: 2 * (10*T + O) == 100*F + 10*O + R,
                    'O-R*': lambda O, R: 2 * O == R or 2 * O == R + 10,
                    'T-F-O': lambda T, F, O: 2 * T == 10*F + O or 2 * T == 10*F + O - 1
                }
            }
        elif N == 1:    # TWO + TWO = FOWR
            return {
                'variables': ['T', 'W', 'O', 'F', 'R'],
                'domains': {
                    'T': [1, 2, 3, 4, 5, 6, 7, 8, 9],
                    'W': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                    'O': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                    'F': [0, 1],
                    'R': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                    'X1': [0, 1],
                    'X2': [0, 1],
                },
                'constraints': {
                    'T-W': lambda T, W: T != W,
                    'T-O': lambda T, O: T != O,
                    'T-F': lambda T, F: T != F,
                    'T-R': lambda T, R: T != R,
                    'W-O': lambda W, O: W != O,
                    'W-F': lambda W, F: W != F,
                    'W-R': lambda W, R: W != R,
                    'O-F': lambda O, F: O != F,
                    'O-R': lambda O, R: O != R,
                    'F-R': lambda F, R: F != R,
                    'T-W-O-F-R': lambda T, W, O, F, R: 2 * (100*T + 10*W + O) == 1000*F + 100*O + 10*W + R,
                    'O-R*': lambda O, R: 2 * O == R or 2 * O == R + 10,
                    'W': lambda W: 2 * W == W or 2 * W == W + 10 or 2 * W == W + 9 or 2 * W == W - 1,
                    'T-O-F': lambda T, O, F: 2 * T == 10*F + O or 2 * T == 10*F + O - 1,
                }
            }
        elif N == 2:    # TXWO + TXWO = FOWXR
            return {
                'variables': ['T', 'X', 'W', 'O', 'F', 'R'],
                'domains': {
                    'T': [1, 2, 3, 4, 5, 6, 7, 8, 9],
                    'X': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                    'W': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                    'O': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                    'F': [0, 1],
                    'R': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                    'C1': [0, 1],
                    'C2': [0, 1],
                    'C3': [0, 1],
                },
                'constraints': {
                    'T-X': lambda T, X: T != X,
                    'T-W': lambda T, W: T != W,
                    'T-O': lambda T, O: T != O,
                    'T-F': lambda T, F: T != F,
                    'T-R': lambda T, R: T != R,
                    'X-W': lambda X, W: X != W,
                    'X-O': lambda X, O: X != O,
                    'X-F': lambda X, F: X != F,
                    'X-R': lambda X, R: X != R,
                    'W-O': lambda W, O: W != O,
                    'W-F': lambda W, F: W != F,
                    'W-R': lambda W, R: W != R,
                    'O-F': lambda O, F: O != F,
                    'O-R': lambda O, R: O != R,
                    'F-R': lambda F, R: F != R,
                    'T-X-W-O-F-R': lambda T, X, W, O, F, R: 2 * (1000*T + 100*X + 10*W + O) == 10000*F + 1000*O + 100*X + 10*W + R,
                    'O-R*': lambda O, R: 2 * O == R or 2 * O == R + 10,
                    'W-X*': lambda W, X: 2 * W == X or 2 * W == X + 10 or 2 * W == X - 1 or 2 * W == X + 9,
                    'X-W*': lambda X, W: 2 * X == W or 2 * X == W + 10 or 2 * X == W - 1 or 2 * X == W + 9,
                    'T-O-F': lambda T, O, F: 2 * T == 10*F + O or 2 * T == 10*F + O - 1,
                }
            }
        elif N == 3:    # TYXWO + TYXWO = FOWXYR
            return {
                'variables': ['T', 'Y', 'X', 'W', 'O', 'F', 'R'],
                'domains': {
                    'T': [1, 2, 3, 4, 5, 6, 7, 8, 9],
                    'Y': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                    'X': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                    'W': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                    'O': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                    'F': [0, 1],
                    'R': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                    'C1': [0, 1],
                    'C2': [0, 1],
                    'C3': [0, 1],
                    'C4': [0, 1],
                },
                'constraints': {
                    'T-Y': lambda T, Y: T != Y,
                    'T-X': lambda T, X: T != X,
                    'T-W': lambda T, W: T != W,
                    'T-O': lambda T, O: T != O,
                    'T-F': lambda T, F: T != F,
                    'T-R': lambda T, R: T != R,
                    'Y-X': lambda Y, X: Y != X,
                    'Y-W': lambda Y, W: Y != W,
                    'Y-O': lambda Y, O: Y != O,
                    'Y-F': lambda Y, F: Y != F,
                    'Y-R': lambda Y, R: Y != R,
                    'X-W': lambda X, W: X != W,
                    'X-O': lambda X, O: X != O,
                    'X-F': lambda X, F: X != F,
                    'X-R': lambda X, R: X != R,
                    'W-O': lambda W, O: W != O,
                    'W-F': lambda W, F: W != F,
                    'W-R': lambda W, R: W != R,
                    'O-F': lambda O, F: O != F,
                    'O-R': lambda O, R: O != R,
                    'F-R': lambda F, R: F != R,
                    'T-Y-X-W-O-F-R': lambda T, Y, X, W, O, F, R: 2 * (10000*T + 1000*Y + 100*X + 10*W + O) == 100000*F + 10000*O + 1000*Y + 100*X + 10*W + R,
                    'O-R*': lambda O, R: 2 * O == R or 2 * O == R + 10,
                    'W-Y*': lambda W, Y: 2 * W == Y or 2 * W == Y + 10 or 2 * W == Y - 1 or 2 * W == Y + 9,
                    'X': lambda X: 2 * X == X or 2 * X == X + 10 or 2 * X == X - 1 or 2 * X == X + 9,
                    'Y-W*': lambda Y, W: 2 * Y == W or 2 * Y == W + 10 or 2 * Y == W - 1 or 2 * Y == W + 9,
                    'T-O-F': lambda T, O, F: 2 * T == 10*F + O or 2 * T == 10*F + O - 1,
                }
            }
    else:
        raise ValueError('Invalid problem type')


def number_of_remaining_values_for_variable(node, var):
    problem = node.problem
    assignment = node.assignment
    if var not in assignment:
        number = 0
        for option in problem['domains'][var]:
            if is_consistent(var, option, node):
                number += 1
        return number
    

def degree_heuristic(node, var):
    problem = node.problem
    assignment = node.assignment
    other_vars = []
    for constraint_key in problem['constraints']:
        vars = [v if '*' not in v else v[:-1] for v in constraint_key.split('-')] 
        if var in vars:
            other_vars += [v for v in vars if v != var and v not in assignment]
    other_vars = list(set(other_vars))
    return len(other_vars)


def select_unassigned_variable(node, MRV, DH):
    problem = node.problem
    assignment = node.assignment
    unassigned_variables = [var for var in problem['variables'] if var not in assignment]
    if MRV and DH:
        return min(unassigned_variables, key = lambda var: (
            number_of_remaining_values_for_variable(node, var),
            degree_heuristic(node, var),
            var
        ))
    elif MRV:
        return min(unassigned_variables, key = lambda var: (
            number_of_remaining_values_for_variable(node, var),
            var
        ))
    return next(var for var in sorted(problem['variables']) if var not in assignment)


def order_domain_values(var, node, LCV):
    problem = node.problem
    assignment = node.assignment
    if LCV:
        constrained_values = {value: 0 for value in problem['domains'][var]}
        for constraint_key in problem['constraints']:
            vars = [v if '*' not in v else v[:-1] for v in constraint_key.split('-')]
            if var in vars:
                other_unassigned_vars = [v for v in vars if v != var and v not in assignment]
                if len(other_unassigned_vars) != 1:
                    continue
                other_var = other_unassigned_vars[0]
                for value in problem['domains'][var]:
                    for other_value in problem['domains'][other_var]:
                        if not problem['constraints'][constraint_key](*[value if v == var else other_value if v == other_var else assignment[v] for v in vars]):
                            constrained_values[value] += 1
        return sorted(problem['domains'][var], key = lambda value: constrained_values[value])
    return sorted(problem['domains'][var])


def is_consistent(var, value, node):
    problem = node.problem
    assignment = node.assignment
    for constraint_key in problem['constraints']:
        vars = [v if '*' not in v else v[:-1] for v in constraint_key.split('-')]
        if var in vars:
            other_vars = [v for v in vars if v != var]
            # if all the other variables in the constraint are assigned test them, else skip
            if all(v in assignment for v in other_vars):
                other_assigned = {v: assignment[v] for v in vars if v != var}
                if not problem['constraints'][constraint_key](*[value if v == var else other_assigned[v] for v in vars]):
                    return False
    return True


def constraint_propagation(var, value, node):
    problem = node.problem
    assignment = node.assignment
    for constraint_key in problem['constraints']:
        vars = [v if '*' not in v else v[:-1] for v in constraint_key.split('-')]
        if var in vars:
            unassigned_other_vars = [v for v in vars if v not in assignment and v != var]
            if len(unassigned_other_vars) == 1:
                other_var = unassigned_other_vars[0]
                temp = problem['domains'][other_var].copy()
                for other_value in problem['domains'][other_var]:
                    if not problem['constraints'][constraint_key](*[value if v == var else other_value if v == other_var else assignment[v] for v in vars]):
                        temp.remove(other_value)
                problem['domains'][other_var] = temp
                
    return


def recursive_backtracking(node, MRV, DH, LCV, CP):
    global expanded_count
    expanded_count += 1
    problem = node.problem
    assignment = node.assignment
    if len(assignment) == len(problem['variables']):
        return assignment
    var = select_unassigned_variable(node, MRV, DH)
    for value in order_domain_values(var, node, LCV):
        if is_consistent(var, value, node):
            new_node = Node(copy.deepcopy(problem), copy.deepcopy(assignment), node)
            new_node.assignment[var] = value
            if CP:
                constraint_propagation(var, value, new_node)
            result = recursive_backtracking(new_node, MRV, DH, LCV, CP)
            if result is not None:                   # "result != failure"
                return result
            if var in assignment:
                del assignment[var]
    return None                                      # "failure"


def backtracking_search(problem, MRV, DH, LCV, CP):
    node = Node(problem, {})
    return recursive_backtracking(node, MRV, DH, LCV, CP)



argv = sys.argv
try:
    # create problem
    N = int(argv[1].strip())
    P = argv[2]
    filename = argv[3]
    problem = generate_problem(N, P)
    with open(filename, 'w') as file:
        sys.stdout = file
        print(f'variables:\n    {problem["variables"]}')
        print(f'domains:\n    {problem["domains"]}')
        print(f'constraints:')

        if P == 'P1':
            for pair, constraint in problem['constraints'].items():
                i, j = map(lambda x: int(x[1:]) - 1, pair.split('-'))  # Extract indices
                print(f"    '{pair}': lambda Q{i+1}, Q{j+1}: Q{i+1} != Q{j+1} and abs(Q{i+1} - Q{j+1}) != {abs(i - j)}\n")
        else:
            for pair, constraint in problem['constraints'].items():
                print(f'{inspect.getsource(constraint)}')
        sys.stdout = sys.__stdout__
except:
    # solve problem
    argv = sys.argv
    filename = argv[-1]
    helper_functions = argv[1:-1]
    
    expanded_count = 0
    MRV = False
    DH = False
    LCV = False
    CP = False
    if 'MRV' in helper_functions:
        MRV = True
    if 'DH' in helper_functions:
        DH = True
    if 'LCV' in helper_functions:
        LCV = True
    if 'CP' in helper_functions:
        CP = True

    with open(filename, 'r') as file:
        lines = file.readlines()
    if lines[0].strip() == 'variables:':
        variables = [v[1:-1] for v in lines[1].strip()[1:-1].split(', ')]
    else:
        raise ValueError('Invalid file format')
    if lines[2].strip() == 'domains:':
        domains = ast.literal_eval(lines[3].strip())
        pass
    else:
        raise ValueError('Invalid file format')
    if lines[4].strip() == 'constraints:':
        constraints = {}
        for i in range(5, len(lines), 2):
            line = lines[i].strip()
            if line[-1] == ',':
                line = line[:-1]
            index = line.find(':')
            constraints[line[:index].strip()[1:-1]] = eval(line[index+1:].strip())
    else:
        raise ValueError('Invalid file format')
    
    problem = {
        'variables': variables,
        'domains': domains,
        'constraints': constraints
    }
    solution = backtracking_search(problem, MRV, DH, LCV, CP)
    print(f'Number of expanded nodes: {expanded_count}')
    print(f'Solution:\n{solution}')