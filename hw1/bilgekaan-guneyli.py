import sys
import heapq as hq

class Node:
    def __init__(self, state, parent, path_cost, queue_round, action, heuristic=0, algorithm='UCS'):
        self.state = state                # (K_row, K_col, B_row, B_col, R_row, R_col, pawn_locations, obstacle_locations)
        self.parent = parent              # another node object
        self.path_cost = path_cost        # cost of the path from the root to the node
        self.queue_round = queue_round    # the round that the node is added to the queue
        self.action = action              # action object
        self.heuristic = heuristic                      # heuristic value
        self.algorithm = algorithm        # algorithm name

    def __lt__(self, other):  # more little means more priority
        if self.algorithm != 'UCS':
            if self.heuristic < other.heuristic:
                return True
            elif self.heuristic > other.heuristic:
                return False

        else:
            if self.path_cost < other.path_cost:
                return True
            elif self.path_cost > other.path_cost:
                return False
        
        # when h1 / cost is equal
        if self.queue_round < other.queue_round:
            return True
        elif self.queue_round > other.queue_round:
            return False
        
        # when queue_round is equal
        # return self.action < other.action
        if self.action < other.action:
            return True
        else:
            return False
        
    def __str__(self):
        return f'PC: {self.path_cost}, QR: {self.queue_round}, DR: {self.action.direction}, SP:{self.action.steps}'
        

class Action:
    def __init__(self, type, direction, steps):
        self.type = type
        self.direction = direction
        self.steps = steps
        if type == 'K':
            self.cost = 6
        elif type == 'B':
            self.cost = 10
        elif type == 'R':
            self.cost = 8

    def __lt__(self, other):  # more little means more priority
        if self.type != other.type:
            if self.type == 'K':
                return True
            elif self.type == 'B' and other.type == 'R':
                return True
            else:
                return False
        else:  # when type is equal
            if type == 'K':
                return self.direction < other.direction
            else:
                if self.direction < other.direction:
                    return True
                elif self.direction == other.direction:
                    return self.steps < other.steps
                else:
                    return False
            

def draw_board(node, N):
    ret = ''
    nodes = []
    while True:
        nodes.append(node)
        node = node.parent
        if node is None:
            break
    for i in range(len(nodes)-1, -1, -1):
        # ret += 'Current node\'s heuristic: ' + str(nodes[i].heuristic) + '\n'
        # ret += 'Current path cost: ' + str(nodes[i].path_cost) + '\n'
        K_row, K_col, B_row, B_col, R_row, R_col, pawn_locations, obstacle_locations = nodes[i].state
        board = [['.' for _ in range(N)] for _ in range(N)]
        if K_row != -1 or K_col != -1:
            board[K_row][K_col] = 'K'
        if B_row != -1 or B_col != -1:
            board[B_row][B_col] = 'B'
        if R_row != -1 or R_col != -1:
            board[R_row][R_col] = 'R'
        for key, val in pawn_locations.items():
            board[key[0]][key[1]] = val
        for i in range(len(obstacle_locations)):
            board[obstacle_locations[i][0]][obstacle_locations[i][1]] = 'x'
        for i in range(N):
            ret += ' '.join(board[i]) + '\n'
        ret += '*' * N + '\n'
    return ret


def choose_h(algorithm, heuristic, node):
    h = 0
    if algorithm == 'GS':
        if heuristic == 'h1':
            h = h1(node)
        else:
            h = h2(node)
    elif algorithm == 'AS':
        if heuristic == 'h1':
            h = node.path_cost + h1(node)
        else:
            h = node.path_cost + h2(node)
    return h


def expand(node, current_round, N, heuristic, algorithm):  # returns a list of new nodes
    expanded_nodes = []  # list of new nodes

    K_row, K_col, B_row, B_col, R_row, R_col, pawn_locations, obstacle_locations = node.state

    if K_row != -1 and K_col != -1:
        new_K_locations = [(K_row+2, K_col-1), (K_row+1, K_col-2), (K_row-1, K_col-2), (K_row-2, K_col-1), (K_row-2, K_col+1), (K_row-1, K_col+2), (K_row+1, K_col+2), (K_row+2, K_col+1)]
        for i in range(len(new_K_locations)):
            # checking out, obstacle, main pieces
            if (new_K_locations[i][0] < 0) or (new_K_locations[i][0] >= N) or (new_K_locations[i][1] < 0) or (new_K_locations[i][1] >= N) or (new_K_locations[i] in obstacle_locations) or (new_K_locations[i] == (B_row, B_col)) or (new_K_locations[i] in (R_row, R_col)):
                continue
            elif new_K_locations[i] in pawn_locations:
                # expand with capturing
                child_pawns = pawn_locations.copy()
                del child_pawns[new_K_locations[i]]
                child_state = (new_K_locations[i][0], new_K_locations[i][1], B_row, B_col, R_row, R_col, child_pawns, obstacle_locations)
                child = Node(child_state, node, node.path_cost + K_cost, current_round, Action('K', i, 0), 0, algorithm)
                child.heuristic = choose_h(algorithm, heuristic, child)
                expanded_nodes.append(child)
            else:
                # expand without capturing
                child_state = (new_K_locations[i][0], new_K_locations[i][1], B_row, B_col, R_row, R_col, pawn_locations, obstacle_locations)            
                child = Node(child_state, node, node.path_cost + K_cost, current_round, Action('K', i, 0), 0, algorithm)
                child.heuristic = choose_h(algorithm, heuristic, child)
                expanded_nodes.append(child)

    if B_row != -1 and B_col != -1:
        r = B_row+1
        c = B_col-1
        step = 1
        while r < N and c > -1:
            if (r, c) in obstacle_locations or (r, c) == (K_row, K_col) or (r, c) == (R_row, R_col):
                break
            elif (r, c) in pawn_locations:
                # expand with capturing
                child_pawns = pawn_locations.copy()
                del child_pawns[(r, c)]
                child_state = (K_row, K_col, r, c, R_row, R_col, child_pawns, obstacle_locations)            
                child = Node(child_state, node, node.path_cost + B_cost, current_round, Action('B', 0, step), 0, algorithm)
                child.heuristic = choose_h(algorithm, heuristic, child)
                expanded_nodes.append(child)
                break
            else:
                # expand without capturing
                child_state = (K_row, K_col, r, c, R_row, R_col, pawn_locations, obstacle_locations)
                child = Node(child_state, node, node.path_cost + B_cost, current_round, Action('B',  0, step), 0, algorithm)
                child.heuristic = choose_h(algorithm, heuristic, child)
                expanded_nodes.append(child)
            r += 1
            c -= 1
            step += 1

        r = B_row-1
        c = B_col-1
        step = 1
        while r > -1 and c > -1:
            if (r, c) in obstacle_locations or (r, c) == (K_row, K_col) or (r, c) == (R_row, R_col):
                break
            elif (r, c) in pawn_locations:
                # expand with capturing
                child_pawns = pawn_locations.copy()
                del child_pawns[(r, c)]
                child_state = (K_row, K_col, r, c, R_row, R_col, child_pawns, obstacle_locations)
                child = Node(child_state, node, node.path_cost + B_cost, current_round, Action('B', 1, step), 0, algorithm)
                child.heuristic = choose_h(algorithm, heuristic, child)
                expanded_nodes.append(child)
                break
            else:
                # expand without capturing
                child_state = (K_row, K_col, r, c, R_row, R_col, pawn_locations, obstacle_locations)
                child = Node(child_state, node, node.path_cost + B_cost, current_round, Action('B', 1, step), 0, algorithm)
                child.heuristic = choose_h(algorithm, heuristic, child)
                expanded_nodes.append(child)
            r -= 1
            c -= 1
            step += 1

        r = B_row-1
        c = B_col+1
        step = 1
        while r > -1 and c < N:
            if (r, c) in obstacle_locations or (r, c) == (K_row, K_col) or (r, c) == (R_row, R_col):
                break
            elif (r, c) in pawn_locations:
                # expand with capturing
                child_pawns = pawn_locations.copy()
                del child_pawns[(r, c)]
                child_state = (K_row, K_col, r, c, R_row, R_col, child_pawns, obstacle_locations)
                child = Node(child_state, node, node.path_cost + B_cost, current_round, Action('B', 2, step), 0, algorithm)
                child.heuristic = choose_h(algorithm, heuristic, child)
                expanded_nodes.append(child)
                break
            else:
                # expand without capturing
                child_state = (K_row, K_col, r, c, R_row, R_col, pawn_locations, obstacle_locations)
                child = Node(child_state, node, node.path_cost + B_cost, current_round, Action('B', 2, step), 0, algorithm)
                child.heuristic = choose_h(algorithm, heuristic, child)
                expanded_nodes.append(child)
            r -= 1
            c += 1
            step += 1
        
        r = B_row+1
        c = B_col+1
        step = 1
        while r < N and c < N:
            if (r, c) in obstacle_locations or (r, c) == (K_row, K_col) or (r, c) == (R_row, R_col):
                break
            elif (r, c) in pawn_locations:
                # expand with capturing
                child_pawns = pawn_locations.copy()
                del child_pawns[(r, c)]
                child_state = (K_row, K_col, r, c, R_row, R_col, child_pawns, obstacle_locations)
                child = Node(child_state, node, node.path_cost + B_cost, current_round, Action('B', 3, step), 0, algorithm)
                child.heuristic = choose_h(algorithm, heuristic, child)
                expanded_nodes.append(child)
                break
            else:
                # expand without capturing
                child_state = (K_row, K_col, r, c, R_row, R_col, pawn_locations, obstacle_locations)
                child = Node(child_state, node, node.path_cost + B_cost, current_round, Action('B',  3, step), 0, algorithm)
                child.heuristic = choose_h(algorithm, heuristic, child)
                expanded_nodes.append(child)
            r += 1
            c += 1
            step += 1

    if R_row != -1 and R_col != -1:
        r = R_row+1
        c = R_col
        step = 1
        while r < N:
            if (r, c) in obstacle_locations or (r, c) == (K_row, K_col) or (r, c) == (B_row, B_col):
                break
            elif (r, c) in pawn_locations:
                # expand with capturing
                child_pawns = pawn_locations.copy()
                del child_pawns[(r, c)]
                child_state = (K_row, K_col, B_row, B_col, r, c, child_pawns, obstacle_locations)
                child = Node(child_state, node, node.path_cost + R_cost, current_round, Action('R', 0, step), 0, algorithm)
                child.heuristic = choose_h(algorithm, heuristic, child)
                expanded_nodes.append(child)
                break
            else:
                # expand without capturing
                child_state = (K_row, K_col, B_row, B_col, r, c, pawn_locations, obstacle_locations)
                child = Node(child_state, node, node.path_cost + R_cost, current_round, Action('R', 0, step), 0, algorithm)
                child.heuristic = choose_h(algorithm, heuristic, child)
                expanded_nodes.append(child)
            r += 1
            step += 1

        r = R_row
        c = R_col-1
        step = 1
        while c > -1:
            if (r, c) in obstacle_locations or (r, c) == (K_row, K_col) or (r, c) == (B_row, B_col):
                break
            elif (r, c) in pawn_locations:
                # expand with capturing
                child_pawns = pawn_locations.copy()
                del child_pawns[(r, c)]
                child_state = (K_row, K_col, B_row, B_col, r, c, child_pawns, obstacle_locations)
                child = Node(child_state, node, node.path_cost + R_cost, current_round, Action('R', 1, step), 0, algorithm)
                child.heuristic = choose_h(algorithm, heuristic, child)
                expanded_nodes.append(child)
                break
            else:
                # expand without capturing
                child_state = (K_row, K_col, B_row, B_col, r, c, pawn_locations, obstacle_locations)
                child = Node(child_state, node, node.path_cost + R_cost, current_round, Action('R', 1, step), 0, algorithm)
                child.heuristic = choose_h(algorithm, heuristic, child)
                expanded_nodes.append(child)
            c -= 1
            step += 1

        r = R_row-1
        c = R_col
        step = 1
        while r > -1:
            if (r, c) in obstacle_locations or (r, c) == (K_row, K_col) or (r, c) == (B_row, B_col):
                break
            elif (r, c) in pawn_locations:
                # expand with capturing
                child_pawns = pawn_locations.copy()
                del child_pawns[(r, c)]
                child_state = (K_row, K_col, B_row, B_col, r, c, child_pawns, obstacle_locations)
                child = Node(child_state, node, node.path_cost + R_cost, current_round, Action('R', 2, step), 0, algorithm)
                child.heuristic = choose_h(algorithm, heuristic, child)
                expanded_nodes.append(child)
                break
            else:
                # expand without capturing
                child_state = (K_row, K_col, B_row, B_col, r, c, pawn_locations, obstacle_locations)
                child = Node(child_state, node, node.path_cost + R_cost, current_round, Action('R', 2, step), 0, algorithm)
                child.heuristic = choose_h(algorithm, heuristic, child)
                expanded_nodes.append(child)
            r -= 1
            step += 1

        r = R_row
        c = R_col+1
        step = 1
        while c < N:
            if (r, c) in obstacle_locations or (r, c) == (K_row, K_col) or (r, c) == (B_row, B_col):
                break
            elif (r, c) in pawn_locations:
                # expand with capturing
                child_pawns = pawn_locations.copy()
                del child_pawns[(r, c)]
                child_state = (K_row, K_col, B_row, B_col, r, c, child_pawns, obstacle_locations)
                child = Node(child_state, node, node.path_cost + R_cost, current_round, Action('R', 3, step), 0, algorithm)
                child.heuristic = choose_h(algorithm, heuristic, child)
                expanded_nodes.append(child)
                break
            else:
                # expand without capturing
                child_state = (K_row, K_col, B_row, B_col, r, c, pawn_locations, obstacle_locations)
                child = Node(child_state, node, node.path_cost + R_cost, current_round, Action('R', 3, step), 0, algorithm)
                child.heuristic = choose_h(algorithm, heuristic, child)
                expanded_nodes.append(child)
            c += 1
            step += 1
    
    return expanded_nodes


def h1(node):
    R_row = node.state[4]
    R_col = node.state[5]
    pawn_locations = node.state[6]
    pawn_rows = [loc[0] for loc in pawn_locations.keys()]
    pawn_cols = [loc[1] for loc in pawn_locations.keys()]
    if pawn_locations == {}:
        return 0
    elif R_row in pawn_rows:
        return len(pawn_locations) * 8         # REMAINING Mİ DEĞİL Mİ ÖĞREN
    elif R_col in pawn_cols:
        return len(pawn_locations) * 8
    else:
        return (len(pawn_locations) + 1) * 8


def h2(node):
    K_row, K_col, B_row, B_col, R_row, R_col, pawn_locations, obstacle_locations = node.state
    is_there_knight = True
    is_there_bishop = True
    is_there_rook = True 
    if K_row == -1 and K_col == -1:
        is_there_knight = False
    if B_row == -1 and B_col == -1:
        is_there_bishop = False
    if R_row == -1 and R_col == -1:
        is_there_rook = False

    if pawn_locations == {}:
        return 0

    # 1. at tekte yiyebiliyorsa (pawn * at maliyeti)
    K_locations = [(K_row+2, K_col-1), (K_row+1, K_col-2), (K_row-1, K_col-2), (K_row-2, K_col-1), (K_row-2, K_col+1), (K_row-1, K_col+2), (K_row+1, K_col+2), (K_row+2, K_col+1)]
    if is_there_knight:
        for item in pawn_locations.keys():
            if item in K_locations:
                return K_cost * len(pawn_locations)
    
    # 2. kale tekte yiyebiliyorsa (pawn * kale maliyeti)
    if is_there_rook:
        for item in pawn_locations.keys():
            if item[0] == R_row or item[1] == R_col:
                return R_cost * len(pawn_locations)

    # 3. fil tekte yiyebiliyorsa (pawn * fil maliyeti)
    bishop_locations = []
    i = 1
    if is_there_bishop:
        while True:
            is_move = False
            r1, c1 = B_row+i, B_col-i  # +, -
            r2, c2 = B_row-i, B_col-i  # -, -
            r3, c3 = B_row-i, B_col+i  # -, +
            r4, c4 = B_row+i, B_col+i  # +, +
            if r1 < N and c1 > -1:
                bishop_locations.append((r1, c1))
                is_move = True
            if r2 > -1 and c2 > -1:
                bishop_locations.append((r2, c2))
                is_move = True
            if r3 > -1 and c3 < N:
                bishop_locations.append((r3, c3))
                is_move = True
            if r4 < N and c4 < N:
                bishop_locations.append((r4, c4))
                is_move = True
            i += 1
            if not is_move:
                break

        for item in pawn_locations.keys():
            if item in bishop_locations:
                return B_cost * len(pawn_locations)

    # 4. at iki hamlede yiyebiliyorsa veya tek başına ise ((pawn+1) * at maliyeti)
    K_locations_new = [(r+2, c-1) for r, c in K_locations] + [(r+1, c-2) for r, c in K_locations] + [(r-1, c-2) for r, c in K_locations] + [(r-2, c-1) for r, c in K_locations] + [(r-2, c+1) for r, c in K_locations] + [(r-1, c+2) for r, c in K_locations] + [(r+1, c+2) for r, c in K_locations] + [(r+2, c+1) for r, c in K_locations]
    if is_there_knight:
        for item in pawn_locations.keys():
            if item in K_locations_new:
                return K_cost * (len(pawn_locations) + 1)
        
    # 5. (geri kalan taşlar + 1) * kale maliyeti
    if is_there_rook:
        return R_cost * (len(pawn_locations) + 1)

    # 6. (geri kalan taşlar + 2) * at maliyeti (kale yok) (atx3 < filx2) 
    if is_there_knight:
        return K_cost * (len(pawn_locations) + 2)
    
    # 7. (geri kalan taşlar + 1) * fil maliyeti (kale yok, at yok) (fil bir yere ya 2 hamlede gidebilir ya da asla gidemez)
    if is_there_bishop:
        return B_cost * (len(pawn_locations) + 1)
    

def graph_search(first_node, N, algorithm, heuristic):
    closed = []
    ret_h1, ret_h2 = h1(first_node), h2(first_node)
    # these are the h1 values, we will have to add the option of h2
    if algorithm != 'UCS':
        if heuristic == 'h1':
            first_node.heuristic = h1(first_node)
        else:
            first_node.heuristic = h2(first_node)     # since path_cost is 0, there is no difference between greedy and A*
    
    fringe = [first_node]
    hq.heapify(fringe)
    current_round = 1
    expanded = 0
    current_node = None

    while True:
        if len(fringe) == 0:
            return ('failure', 0, '')
        current_node = hq.heappop(fringe)
        # expanded += 1
        if len(current_node.state[6]) == 0:
            drawings = draw_board(current_node, N)
            return (expanded, current_node.path_cost, drawings, ret_h1, ret_h2)
        if current_node.state not in closed:
            closed.append(current_node.state)
            expanded += 1
            expanded_nodes = expand(current_node, current_round, N, heuristic, algorithm)
            # expanded += len(expanded_nodes)
            for node in expanded_nodes:
                #if node.state not in closed:
                # closed.append(node.state)
                hq.heappush(fringe, node)
                # expanded += 1
        current_round += 1


argv = sys.argv
board_list = []
with open(argv[1], 'r') as board_file:
    for line in board_file:
        line = line.lstrip('\ufeff').rstrip('\n').split(' ')
        board_list.append(line)

algs = argv[3].split('/')
try:
    heurs = argv[4].split('/')
except:
    heurs = []

options = []
for alg in algs:
    if alg == 'UCS':
        options.append(('UCS', None))
    else:
        for heur in heurs:
            options.append((alg, heur))

B_cost = 10
K_cost = 6
R_cost = 8
N = len(board_list)
K_row = -1
K_col = -1
B_row = -1
B_col = -1
R_row = -1
R_col = -1
pawn_locations = {}
obstacle_locations = []


# find initial positions of the main pieces
for i in range(len(board_list)):
    for j in range(len(board_list[i])):
        if board_list[i][j] == 'K':
            K_row = i
            K_col = j
        elif board_list[i][j] == 'B':
            B_row = i
            B_col = j
        elif board_list[i][j] == 'R':
            R_row = i
            R_col = j
        elif board_list[i][j] == 'x':
            obstacle_locations.append((i, j))
        elif board_list[i][j].isdigit():
            pawn_locations[(i, j)] = board_list[i][j]

pawn_count = len(pawn_locations)
initial_state = (K_row, K_col, B_row, B_col, R_row, R_col, pawn_locations, obstacle_locations)

with open(argv[2], 'w') as output_file:
    sys.stdout = output_file
    for option in options:
        # print(f'Algorithm: {option[0]}, Heuristic: {option[1]}')
        first_node = Node(initial_state, None, 0, 0, None, 0, alg)
        expanded, cost, drawings, ret_h1, ret_h2 = graph_search(first_node, N, option[0], option[1])
        print(f'expanded: {expanded}\npath-cost: {cost}\nh1: {ret_h1}\nh2: {ret_h2}\n{drawings}')

sys.stdout = sys.__stdout__