import datetime
import numpy as np

def format_matrix(matrix):
    result = ""
    result += "\n".join([" ".join(map(str, row)) for row in matrix])
    return result

def write_pomdp_file(
    states,
    actions,
    observations,
    init_state_line,
    transition_lines, 
    observation_lines, 
    reward_lines,
    discount, 
    pomdp_path=None,
    header=None,
):
    """
    Write a string that represents a POMDP model file.

    Args:
        - states (list): A list of state identifiers (str); states[0] will be the fixed initial state for the infinite horizon problem.
        - actions (list): A list of action identifiers (str).
        - observations (list): A list of observation identifiers (str).
        - transition_lines (list): A list of phrases (str) that specify the transtion probrobility kernel for the POMDP.
        - observation_lines (list): A list of phrases (str) that specify the obervation probrobility kernel for the POMDP.
        - reward_lines (list): A list of phrases (str) defining the reward structure.
        - discount (float): The discount factor.
        - pomdp_path: Path to save the POMDP file (optional)
        - header: (str) Phrase to be added as a header (comment) for the POMDP file (optional)
        
    Returns:
        POMDP file as a string (if pomdp_path is not provided)
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    result = f"# " + (header or 'Auto-generated') + "\n\n"
    # Begin constructing the POMDP file content
    result += f"discount: {discount}\n"
    result += f"values: reward\n" # only consider rewards (instead of costs)
    result += "states: " + " ".join(map(str, states)) + "\n"
    result += "actions: " + " ".join(map(str, actions)) + "\n"
    result += "observations: " + " ".join(map(str, observations)) + "\n\n"

    result += f"{init_state_line}\n\n"

    # Add transition probabilities
    result += "\n\n".join([f"{phrase}" for phrase in transition_lines])
    result += "\n\n"

    # Add observation probabilities
    result += "\n\n".join([f"{phrase}" for phrase in observation_lines])
    result += "\n\n"

    # Add reward lines
    result += "\n".join([f"{phrase}" for phrase in reward_lines])

    # Write to file if path is specified, otherwise return as string
    if pomdp_path is not None:
        with open(pomdp_path, 'w') as f:
            print(result, file=f)
    else:
        return result


def create_finite_horizon_stationary_pomdp_file(
    states,
    actions,
    observations,
    horizon,
    init_state_probs,
    transition_matrix,
    observation_matrix,
    reward_matrix,
    discount=0.999,
    pomdp_path=None,
    decimals=6,
    header=None,
):
    """
    POMDP file creation for a finite horizon problem, where observation depends
    only on the current state.
        
    Args:
        - horizon: Time horizon (H)
        - states: List of state names
        - actions: List of action names
        - observations: List of observation names
        - transition_matrix: Stationary transition matrix of shape [n_states, n_actions, n_states]
        - observation_matrix: Stationary observation matrix of shape [n_states, n_obs]
        - reward_matrix: Reward matrix of shape [n_obs, n_actions]
        - discount: Discount factor (gamma)
        - pomdp_path: Path to save the POMDP file (optional)
        - header: (str) Phrase to be added as a header (comment) for the POMDP file (optional)

    Returns:
        POMDP file as a string (if pomdp_path is not provided)
    
    NOTE: In our experiments, the order of realization is: 
           (s_0, o_0, a_0, r_0, s_1, o_1, a_1, r_1, s_2, o_2, a_2, ...) 
        where a_t can depend on (o_1:t, a_1:t-1).
        However, in the .POMDP files used by POMDP planners (e.g., SARSOP), the order of 
        realization is:
           (s'_-1, a'_-1, s'_0, o'_-1, r'_-1, a'_0, s'_1, o'_0, r'_0, ...)
        where prime (') is used to indicate the r.v. is associated with the .PODMP file, and the
        initial action a'_-1 depends only on the agent's prior. Also, the POMDO file is for an 
        infinite horizon problem with discounting. To fix this, we introduce an expanded
        state space S' = {Start, End} U (S x {0, 1, ..., H-1}), and set 
            s'_-1 = Start w.p. 1 and Start -> (s, 0) w.p. b_0(s) 
        where Start is a fixed starting state and b_0 is the initial state probability for 
        the original POMDP. We then set
            (s, H-1) -> End and End -> End w.p. 1, 
        so End is an absorbing state representing the end of horizon.
        The corresponding reward model r' satisfies 
            r'((s, h), a, *, o) = r(o, a)   for (s, h) in S' \ {Start, End}
            r'(Start, *, *, *) = r'(*, *, Start, *) = r'(End, *, *, *) = 0 
        (notice that here o ~ Z'(* | (s, h), a) = Z_h(* | s)).
        """
    n_states = len(states)
    n_obs = len(observations)

    # 1. Expand state space to include Start, time-indexed states, and an absorbing End state 
    expanded_states = ['Start'] + [f'{s}-{t}' for t in range(horizon) for s in states] + ['End']
    n_expanded_states = len(expanded_states)
    
    # 2. Transition Matrix
    transitions_by_action = {}
    for a_idx, action in enumerate(actions):
        expanded_transition = np.zeros((n_expanded_states, n_expanded_states))

        # Start transitions to {s}_0 w.p. init_state_probs[s], regardless of action
        s0_idxs = np.arange(1, n_states + 1)
        expanded_transition[0, s0_idxs] = init_state_probs
        # For each state at time = 0, ..., H-2
        for t in range(horizon - 1):
            offset_t = 1 + t * n_states
            offset_next_t = offset_t + n_states
            for s_idx in range(n_states):
                s_t_idx = offset_t + s_idx
                s_next_idxs = range(offset_next_t, offset_next_t + n_states)
                # Match the original transition probs. (shifted by time)
                expanded_transition[s_t_idx, s_next_idxs] = transition_matrix[s_idx, a_idx]
        
        # #  At time H - 1, {s}_t transitions to {s}-H w.p. 1
        # s_t_idxs = np.arange(n_states) + (1 + (horizon - 1) * n_states)
        # s_next_idxs = np.arange(n_states) + (1 + horizon * n_states)   
        # expanded_transition[s_t_idxs, s_next_idxs] = 1.0
        # States {s}-H and End always transitions to End
        expanded_transition[-(n_states+1):, -1] = 1.0

        transitions_by_action[action] = expanded_transition # .round(decimals=decimals) # to avoid numerical issue

        # 3. Observation Matrix for expanded states
        expanded_obs_probs = np.zeros((n_expanded_states, n_obs))

        # Observations for {s}_0:H-1 match {s}
        expanded_obs_probs[1:-1] = np.tile(observation_matrix, (horizon, 1))
        # Uniform observation distribution for Start and End states (which always get 0 reward)
        expanded_obs_probs[[0, -1], :] = 1 / n_obs
        expanded_obs_probs = expanded_obs_probs.round(decimals=decimals) # to avoid numerical issue

    # 4. Reward Definitions
    reward_lines = []

    # Define rewards (for regular states) according to action and the resulting observation
    for a_idx, action in enumerate(actions):
        for o_idx, observation in enumerate(observations):
            reward = reward_matrix[o_idx, a_idx]
            reward_lines += [f"R: {action} : * : * : {observation} {reward}"]

    # Zero reward for Start and End states
    # This will overwrite previous reward definitions in case of a conflict
    reward_lines.append(f"R: * : Start : * : * 0")
    reward_lines.append(f"R: * : * : Start : * 0")
    reward_lines.append(f"R: * : End : * : * 0")
    # NOTE: (s, H-1) -> End transitions may still emit reward

    # 5. Format the transition, observation into POMDP file format
    transition_lines = [f"T: {action}\n{format_matrix(transition)}" 
                        for action, transition in transitions_by_action.items()]
    observation_lines = [f"O: {action}\n{format_matrix(expanded_obs_probs)}" for action in actions]
    
    # 6. Specify initial state
    init_state_line = "start: Start"

    # Finally, save or return POMDP file
    return write_pomdp_file(
        expanded_states, actions, observations, init_state_line, transition_lines, observation_lines, reward_lines,
        discount, pomdp_path, header
    )



def create_tiger_pomdp_file(
    horizon,
    theta,
    tiger_penalty=-100,
    treasure_reward=10,
    listen_cost=-1,
    discount=0.999,
    pomdp_path=None,
):
    effective_horizon = horizon - 1 # -1 as the last step in Tiger_FiniteHorizon has no effect
    
    states =  ['Start']
    states += [s for h in range(effective_horizon) for s in [f'Tiger-Left-{h}', f'Tiger-Right-{h}']]
    states += ['End']

    actions = ["Listen", "Open-Left", "Open-Right"]
    observations = ["Hear-Left", "Hear-Right"]

    n_states = len(states)

    transitions_for_listen = np.zeros((n_states, n_states))
    transitions_for_listen[0, [1, 2]] = 0.5 # Start -> TL_0 or TR_0 w.p. 1/2
    # For t = 0, 1, ..., H-1: move 2 positions to the right, i.e.,
    #  TL_t [2t+1] -> TL_t+1 [2(t+1)+1] and TR_t [2t+2]-> TR_t+1 [2(t+1)+2] w.p. 1
    np.fill_diagonal(transitions_for_listen[1:, 3:], 1)
    transitions_for_listen[-2:, -1] = 1 # TR_H-1, End -> End w.p. 1

    transition_lines = [
        "T: Listen\n" + format_matrix(transitions_for_listen),
        "T: Open-Left : * : End 1.0", # always move to End, regardless of starting state 
        "T: Open-Right : * : End 1.0", # always move to End, regardless of starting state
    ]

    stationary_obs_probs = np.array([[0.5 + theta, 0.5 - theta],
                                    [0.5 - theta, 0.5 + theta]]).round(16)

    obs_probs_for_listen = np.vstack([np.array([0.5, 0.5])]
                                     + [stationary_obs_probs] * (effective_horizon)
                                     + [np.array([0.5, 0.5])])
    obs_lines = [
        "O: Listen \n" + format_matrix(obs_probs_for_listen),
        "O: Open-Left\nuniform",
        "O: Open-Right\nuniform",
    ]

    # R: <action> : <start-state> : <end-state> : <obs> %f
    reward_lines = ["R: Listen : Start : * : * 0"]
    reward_lines += [f"R: Listen : Tiger-Left-{h} : * : * {listen_cost}" for h in range(effective_horizon)]
    reward_lines += [f"R: Listen : Tiger-Right-{h} : * : * {listen_cost}" for h in range(effective_horizon)]
    reward_lines += [f"R: Listen : End : * : * {listen_cost}"]
    reward_lines += ["R: Open-Left : Start : * : * 0"]
    reward_lines += [f"R: Open-Left : Tiger-Left-{h} : * : * {tiger_penalty}" for h in range(effective_horizon)]
    reward_lines += [f"R: Open-Left : Tiger-Right-{h} : * : * {treasure_reward}" for h in range(effective_horizon)]
    reward_lines += ["R: Open-Left : End : * : * 0"]
    reward_lines += ["R: Open-Right : Start : * : * 0"]
    reward_lines += [f"R: Open-Right : Tiger-Left-{h} : * : * {treasure_reward}" for h in range(effective_horizon)]
    reward_lines += [f"R: Open-Right : Tiger-Right-{h} : * : * {tiger_penalty}" for h in range(effective_horizon)]
    reward_lines += ["R: Open-Right : End : * : * 0"]

    # Specify initial state
    init_state_line = "start: Start"

    return write_pomdp_file(
        states, actions, observations, init_state_line, transition_lines, obs_lines, reward_lines,
        discount, pomdp_path=pomdp_path
    )