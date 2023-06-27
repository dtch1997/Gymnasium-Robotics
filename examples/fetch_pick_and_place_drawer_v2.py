import numpy as np
import gymnasium as gym 
import gymnasium_robotics.envs # noqa: F401

def print_env_info(env: gym.Env):
    print("Qpos shape: ", env.data.qpos.shape)
    print("Qvel shape: ", env.data.qvel.shape)
    print("Observation space: ", env.observation_space)
    print("Action space: ", env.action_space)

def zero_action(env: gym.Env):
    return np.zeros(env.action_space.shape)

if __name__ == "__main__":    
    env = gym.make(
        "FetchPickAndPlaceDrawer-v2", 
        render_mode='human', 
        is_closed_on_reset = False, # Default: True
        is_cube_inside_drawer_on_reset = False # Default: True
    )
    print("Env info of FetchPickAndPlaceDrawer-v2")
    print_env_info(env)
    obs = env.reset()
    print("Observation: ", obs)

    # Use these methods to reset the drawer state

    env.reset_drawer_open()
    # env.reset_drawer_closed() 
    
    # env.reset_cube_outside_drawer()
    env.reset_cube_inside_drawer()

    try:
        while True:
            img = env.render()
            obs, reward, term, trunc, info = env.step(zero_action(env))
    except KeyboardInterrupt:
        env.close()