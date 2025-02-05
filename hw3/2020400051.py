
def create_initial_belief():
    belief = {}
    for i in range(N):
        for j in range(N):
            if map[i][j] != 'x':
                belief[(0, i, j)] = 1 / num_of_gaps
                belief[(1, i, j)] = 1 / num_of_gaps
                belief[(2, i, j)] = 1 / num_of_gaps
                belief[(3, i, j)] = 1 / num_of_gaps
    return belief


def read_obs_and_action(text):
    step = text.split(',')
    if len(step) == 2:
        action = step[0][1:]
        observation = step[1][4:]
    else:
        if ':' in step[0]:
            action = None
            observation = step[0][4:]
        else:
            action = step[0][1:]
            observation = None
    return action, observation


def zeros_like_belief():
    belief = {}
    for i in range(N):
        for j in range(N):
            if map[i][j] != 'x':
                belief[(0, i, j)] = 0
                belief[(1, i, j)] = 0
                belief[(2, i, j)] = 0
                belief[(3, i, j)] = 0
    return belief


def distance_to_nearest_obstacle(d, r, c):
    if d == 0:
        for i in range(r, -1, -1):
            if map[i][c] == 'x':
                return r - i
    elif d == 1:
        for i in range(c, N+1):
            if map[r][i] == 'x':
                return i - c
    elif d == 2:
        for i in range(r, N+1):
            if map[i][c] == 'x':
                return i - r
    elif d == 3:
        for i in range(c, -1, -1):
            if map[r][i] == 'x':
                return c - i
    else:
        raise ValueError('Invalid direction')
    return 0


def multiply_beliefs(belief1, belief2):
    result = zeros_like_belief()
    for key in belief1.keys():
        result[key] = belief1[key] * belief2[key]
    return result


def P_transition(s, action):
    d, r, c = s

    stay_prob = (r+1) / (2*N)
    drift_prob = (c+1) / (2*N)
    action_prob = 1 - stay_prob - drift_prob

    if action == 'ccw':
        new_direction = (d - 1) % number_of_directions
        new_state = new_direction, r, c
    elif action == 'cw':
        new_direction = (d + 1) % number_of_directions
        new_state = new_direction, r, c
    elif action == 'forward':
        if d == 0:
            new_state = d, r - 1, c
        elif d == 1:
            new_state = d, r, c + 1
        elif d == 2:
            new_state = d, r + 1, c
        elif d == 3:
            new_state = d, r, c-1
        else:
            raise ValueError('Invalid direction')
    else:
        raise ValueError(f'Invalid action: {action}')
    
    return stay_prob, drift_prob, action_prob, new_state
            

def P_observation(s, observation):
    d, r, c = s
    obs_back = int(observation[0])
    obs_forw = int(observation[1])

    dist_back = distance_to_nearest_obstacle((d-2) % number_of_directions, r, c)
    dist_forw = distance_to_nearest_obstacle(d, r, c)

    if dist_forw == 1 and obs_forw != 1:
        return 0
    elif dist_back == 1 and obs_back != 1:
        return 0
    
    if obs_back == 1:
        back_prob = 1 / (dist_back**2)
    elif obs_back == 2:
        back_prob = 2 / (dist_back**2)
    elif obs_back == 0:
        back_prob = 1 - (3 / (dist_back**2))
    else:
        raise ValueError('Invalid observation')
    
    if obs_forw == 1:
        forw_prob = 1 / (dist_forw**2)
    elif obs_forw == 2:
        forw_prob = 2 / (dist_forw**2)
    elif obs_forw == 0:
        forw_prob = 1 - (3 / (dist_forw**2))
    else:
        raise ValueError('Invalid observation')
    
    return back_prob * forw_prob


def forward(query):
    belief = create_initial_belief()
    
    for k in range(len(query)):
        action, observation = read_obs_and_action(query[k])

        # P(R) = P_transition * belief
        if action is not None:
            transition_belief = zeros_like_belief()

            for key in belief.keys():
                stay_prob, drift_prob, action_prob, after_action_state = P_transition(key, action)

                # staying in the same cell
                transition_belief[key] += stay_prob * belief[key]

                # drifting to the right
                if key[2] + 1 < N and map[key[1]][key[2] + 1] != 'x':
                    transition_belief[(key[0], key[1], key[2] + 1)] += drift_prob * belief[key]
                else:
                    transition_belief[key] += drift_prob * belief[key]

                # action (forward or turn)
                new_d, new_r, new_c = after_action_state[0], after_action_state[1], after_action_state[2]
                if new_r >= 0 and new_r < N and new_c >= 0 and new_c < N and map[new_r][new_c] != 'x':
                    transition_belief[after_action_state] += action_prob * belief[key]
                else:
                    transition_belief[key] += action_prob * belief[key]

            belief = transition_belief
        
        # P_observation * P(R)
        if observation is not None:
            observation_belief = zeros_like_belief()
            for key in belief.keys():
                observation_belief[key] = P_observation(key, observation) * belief[key]
            belief = observation_belief

        # Normalization
        total_sum = sum(belief.values())
        for key in belief.keys():
            belief[key] /= total_sum
        
        
    return belief


def smoothing(k, query):
    forward_belief = forward(query[:k + 1])

    backward_belief = zeros_like_belief()
    for key in backward_belief.keys():
        backward_belief[key] = 1

    for i in range(len(query) - 1, k, -1):
        action, observation = read_obs_and_action(query[i])
        new_backward_belief = zeros_like_belief()

        for key1 in backward_belief.keys():
            res = 0
            stay_prob_key1, drift_prob_key1, action_prob_key1, after_action_state_key1 = P_transition(key1, action) # (stay_prob, drift_prob, action_prob, after_action_state)
            d1, r1, c1 = key1
            is_drift = False
            is_move = False
            if (d1, r1, c1+1) in backward_belief:
                is_drift = True
            if after_action_state_key1 in backward_belief:
                is_move = True
            for key2 in backward_belief.keys():
                P_sensor = P_observation(key2, observation)
                d2, r2, c2 = key2
                P_motion = 0
                if key2 == after_action_state_key1:
                    P_motion += action_prob_key1
                if (d2, r2, c2-1) == key1:
                    P_motion += drift_prob_key1
                if key2 == key1:
                    P_motion += stay_prob_key1    # stay
                    # action is impossible
                    if not is_move:
                        P_motion += action_prob_key1
                    # drift but impossible 
                    if not is_drift:
                        P_motion += drift_prob_key1
                    
                                        
                res += P_motion * P_sensor * backward_belief[key2]

            new_backward_belief[key1] = res

        backward_belief = new_backward_belief

        # Normalization
        total_sum = sum(backward_belief.values())
        for key in backward_belief.keys():
            backward_belief[key] /= total_sum

    belief = multiply_beliefs(forward_belief, backward_belief)
    total_sum = sum(belief.values())
    for key in belief.keys():
        belief[key] /= total_sum

    return belief


def sum_belief(belief):
    result = {}
    for key in belief.keys():
        result[(key[1], key[2])] = sum(belief[(i, key[1], key[2])] for i in range(number_of_directions))
    return result


if __name__ == '__main__':
    global N
    global number_of_directions
    global map
    global num_of_gaps

    map_filepath    = 'map.txt'
    query_filepath  = 'query.txt'
    output_filepath = 'output.txt'

    output_text = ''
    with open(map_filepath, 'r') as map_file:
        map = [line.strip() for line in map_file]

    with open(query_filepath, 'r') as query_file:
        query_list = [line.strip().split(':', 1) for line in query_file]

    N = len(map)
    number_of_directions = 4
    num_of_obstacles = sum(row.count('x') for row in map)
    num_of_gaps = N * N - num_of_obstacles


    for query in query_list:
        query = query[0].split(' ') + query[1].split(';')

        if query[0] == 'Filtering' or query[0] == 'Prediction':
            k = int(query[2].split('=')[1])
            if query[0] == 'Filtering':
                forward_belief = forward(query[3:k+4])
            else:
                forward_belief = forward(query[3:])
            forward_belief = sum_belief(forward_belief)

            max_prob = float('-inf')
            max_location = (-1, -1)
            for key, val in forward_belief.items():
                if val > max_prob:
                    max_prob = val
                    max_location = key

            c_r_max = (max_location[1]+1, max_location[0]+1)
            output_text += f'{c_r_max}: {round(max_prob, 3)}\n\n'


        elif query[0] == 'Smoothing':
            k = int(query[2].split('=')[1])
            smoothing_belief = smoothing(k, query[3:])
            smoothing_belief = sum_belief(smoothing_belief)

            max_prob = float('-inf')
            max_location = (-1, -1)
            for key, val in smoothing_belief.items():
                if val > max_prob:
                    max_prob = val
                    max_location = key

            c_r_max = (max_location[1]+1, max_location[0]+1)
            output_text += f'{c_r_max}: {round(max_prob, 3)}\n\n'
        

    with open(output_filepath, 'w') as output_file:
        output_file.write(output_text)


