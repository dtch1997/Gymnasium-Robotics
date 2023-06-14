import gymnasium as gym 
import gymnasium_robotics.envs # noqa: F401

def print_env_info(env: gym.Env):
    print("Qpos shape: ", env.data.qpos.shape)
    print("Qvel shape: ", env.data.qvel.shape)
    print("Observation space: ", env.observation_space)
    print("Action space: ", env.action_space)

if __name__ == "__main__":
    env = gym.make("FetchPickAndPlaceDrawer-v2", render_mode='human')
    print("Env info of FetchPickAndPlaceDrawer-v2")
    print_env_info(env)
    env.reset()
    img = env.render()
    input("Press Enter to continue...")