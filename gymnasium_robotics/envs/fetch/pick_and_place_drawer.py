import os
import numpy as np

from gymnasium import spaces
from gymnasium.utils.ezpickle import EzPickle
from gymnasium_robotics.envs.fetch import MujocoFetchEnv
from gymnasium_robotics.envs.fetch.pick_and_place import MujocoFetchPickAndPlaceEnv

MODEL_XML_PATH = os.path.join("fetch", "pick_and_place_drawer.xml")

# These values can be checked in the mujoco viewer
# python -m mujoco.viewer --mjcf=gymnasium_robotics/envs/assets/fetch/pick_and_place_drawer.xml
# Click Watch --> qpos index 22 --> drag the drawer open and closed and observe the changes
DRAWER_STATE_IDX = 22
DRAWER_OPEN = -0.12
DRAWER_CLOSED = 0.0

class MujocoFetchPickAndPlaceDrawerEnv(MujocoFetchPickAndPlaceEnv):
    """ This environment is the same as FetchPickAndPlaceEnv, but with a drawer."""

    def __init__(self, reward_type="sparse", **kwargs):
        # Changes from original PickAndPlace env: a 
        # - Changed model path
        # - Add drawer state to observation space
        initial_qpos = {
            "robot0:slide0": 0.405,
            "robot0:slide1": 0.48,
            "robot0:slide2": 0.0,
            "object0:joint": [1.25, 0.53, 0.4, 1.0, 0.0, 0.0, 0.0],
        }
        MujocoFetchEnv.__init__(
            self,
            model_path=MODEL_XML_PATH,
            has_object=True,
            block_gripper=False,
            n_substeps=20,
            gripper_extra_height=0.2,
            target_in_the_air=True,
            target_offset=0.0,
            obj_range=0.15,
            target_range=0.15,
            distance_threshold=0.05,
            initial_qpos=initial_qpos,
            reward_type=reward_type,
            **kwargs,
        )
        EzPickle.__init__(self, reward_type=reward_type, **kwargs)        

        obs = self._get_obs()
        self.observation_space = spaces.Dict(
            dict(
                desired_goal=spaces.Box(
                    -np.inf, np.inf, shape=obs["achieved_goal"].shape, dtype="float64"
                ),
                achieved_goal=spaces.Box(
                    -np.inf, np.inf, shape=obs["achieved_goal"].shape, dtype="float64"
                ),
                observation=spaces.Box(
                    -np.inf, np.inf, shape=obs["observation"].shape, dtype="float64"
                ),
                drawer_state=spaces.Box(
                    -np.inf, np.inf, shape=obs["drawer_state"].shape, dtype="float64"
                ),
            )
        )

    def get_drawer_state(self):
        """ Returns the state of the drawer. """
        return self.data.qpos[22]
    
    def _reset_sim(self):
        retval = super()._reset_sim()
        # Reset the drawer state
        self.data.qpos[22] = DRAWER_CLOSED
        return retval

    def _get_obs(self):
        base_obs = super()._get_obs()
        # Add the drawer state to the observation
        base_obs["drawer_state"] = np.array(self.get_drawer_state())
        return base_obs
