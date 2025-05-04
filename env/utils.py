def format_observation(obs):
    return {
        'x': obs[0],
        'y': obs[1],
        'has_bucket': bool(obs[2]),
        'fire_out': bool(obs[3]),
    }

def is_terminal_state(obs):
    return (obs[0], obs[1]) == (3, 3) and obs[3] == 1
