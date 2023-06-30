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
        cube_pos_on_reset = "in_drawer" # Default: True
    )
    print("Env info of FetchPickAndPlaceDrawer-v2")
    print_env_info(env)
    obs, info = env.reset()
    print("Observation: ", obs)
    print("\nreset cube_pos: ", obs['observation'][3:6])

    # Use these methods to reset the drawer state

    # env.reset_drawer_open()
    # env.reset_drawer_closed() 
    
    # env.reset_cube_outside_drawer()
    # env.reset_cube_inside_drawer()

    try:
        for _ in range(50):
            img = env.render()
            obs, reward, term, trunc, info = env.step(zero_action(env))
            print("Step cube_pos: ", obs['observation'][3:6])
            input()
    except KeyboardInterrupt:
        env.close()