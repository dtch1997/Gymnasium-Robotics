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
DRAWER_QPOS_IDX = 22
DRAWER_QVEL_IDX = 21
DRAWER_OPEN = -0.12
DRAWER_CLOSED = 0.0

class MujocoFetchPickAndPlaceDrawerEnv(MujocoFetchPickAndPlaceEnv):
    """ This environment is the same as FetchPickAndPlaceEnv, but with a drawer."""

    def __init__(self, reward_type="sparse", **kwargs):
        # Changes from original PickAndPlace env:
        # - Changed model path
        # - Add drawer state to observation space
        initial_qpos = {
            "robot0:slide0": 0.405,
            "robot0:slide1": 0.48,
            "robot0:slide2": 0.0,
            "object0:joint": [1.25, 0.53, 0.4, 1.0, 0.0, 0.0, 0.0],
        }
        self.is_closed_on_reset = kwargs.pop("is_closed_on_reset", True)
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
                info=spaces.Dict(dict(
                    drawer_handle=spaces.Box(
                        -np.inf, np.inf, shape=obs["info"]["drawer_handle"].shape, dtype="float64"
                    ),
                    drawer_volume_min=spaces.Box(
                        -np.inf, np.inf, shape=obs["info"]["drawer_volume_min"].shape, dtype="float64"
                    ),
                    drawer_volume_max=spaces.Box(
                        -np.inf, np.inf, shape=obs["info"]["drawer_volume_max"].shape, dtype="float64"
                    ),
                    )
                ),
            )
        )

    def reset_drawer_closed(self):
        """ Resets the environment with the drawer closed. """
        self.is_closed_on_reset = True
        return self.reset()
    
    def reset_drawer_open(self):
        """ Resets the environment with the drawer open. """
        self.is_closed_on_reset = False
        return self.reset()

    def get_drawer_state(self):
        """ Returns the state of the drawer. """
        pos = self.data.qpos[DRAWER_QPOS_IDX]
        vel = self.data.qvel[DRAWER_QVEL_IDX]
        return (pos, vel)
    
    def get_site_xpos(self, site_name: str):
        """ Returns the position of a site. """
        return self.data.site(site_name).xpos.copy()

    def _reset_sim(self):
        retval = super()._reset_sim()
        # Reset the drawer state
        drawer_state = DRAWER_CLOSED if self.is_closed_on_reset else DRAWER_OPEN
        self.data.qpos[DRAWER_QPOS_IDX] = drawer_state
        self.data.qvel[DRAWER_QVEL_IDX] = 0.0
        return retval

    def _get_obs(self):
        base_obs = super()._get_obs()
        # Add the drawer state to the observation
        base_obs["drawer_state"] = np.array(self.get_drawer_state())
        base_obs["info"] = dict()
        base_obs["info"]["drawer_handle"] = self.get_site_xpos("drawer_handle")
        base_obs["info"]["drawer_volume_min"] = self.get_site_xpos("drawer_volume_min")
        base_obs["info"]["drawer_volume_max"] = self.get_site_xpos("drawer_volume_max")
        return base_obs
