#customized stacking using a self defined class

from omni.isaac.examples.base_sample import BaseSample
from omni.isaac.franka import Franka
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.tasks import BaseTask
from omni.isaac.franka.controllers import StackingController
import numpy as np

class FrankaDances(BaseTask):
    def __init__(self, name):
        super().__init__(name=name, offset=None)
        self._goal_position1 = np.array([0.5, 0.5, 0.02575])
        self._goal_position2 = np.array([0.5, 0.5, 0.07725])
        self._task_completed = False
        self._task_completed1 = False
        return
    
    def set_up_scene(self, scene):
        super().set_up_scene(scene)
        scene.add_default_ground_plane()
        self._cube1 = scene.add(DynamicCuboid(
            prim_path="/World/Red_Cube", 
            position=np.array([0.3, 0.3, 0.3]),
            name="red_cube",
            color=np.array([1, 0, 0]),
            scale=np.array([0.0515, 0.0515, 0.0515])
        ))
        self._cube2 = scene.add(DynamicCuboid(
            prim_path="/World/Green_Cube", 
            position=np.array([-0.2, -0.3, 0.3]),
            name="green_cube",
            color=np.array([0.0, 1.0, 0]),
            scale=np.array([0.0515, 0.0515, 0.0515])
        ))
        self._franka = scene.add(Franka(prim_path="/World/Roboarm", name="my_roboarm"))

    def get_observations(self):
        #observations -> dictionary (positon, target position),, current_joint
        current_joint_positions = self._franka.get_joint_positions()
        cube1_position, _ = self._cube1.get_world_pose()
        cube2_position, _ = self._cube2.get_world_pose()
        observations = {
            self._cube1.name : {
                "position" : cube1_position,
                "target_position" : self._goal_position1
            },
            self._cube2.name : {
                "position" : cube2_position,
                "target_position" : self._goal_position2
            },
            self._franka.name : {
                "joint_positions" : current_joint_positions
            }
        }
        return observations


    def pre_step(self, control_index, simulation_time):
        cube1_position, _ = self._cube1.get_world_pose()
        cube2_position, _ = self._cube2.get_world_pose()
        if np.mean(np.abs(cube1_position - self._goal_position2)) < 0.02 and not self._task_completed1:
            self._cube1.get_applied_visual_material().set_color(color=np.array([1.0, 1.0, 0]))
            self._task_completed1 = True

        if not self._task_completed and np.mean(np.abs(self._goal_position2 - cube2_position)) < 0.01:
            self._cube2.get_applied_visual_material().set_color(color=np.array([0, 1.0, 1.0]))
            self._task_achieved = True
        return


    def post_reset(self):
        self._franka.gripper.set_joint_positions(self._franka.gripper.joint_opened_positions)
        self._cube1.get_applied_visual_material().set_color(color=np.array([1, 0, 0]))
        self._cube2.get_applied_visual_material().set_color(color=np.array([0, 1.0, 0]))
        self._task_completed = False
        self._task_completed1 = False
        return
    

class HelloWorld(BaseSample):
    def __init__(self) -> None:
        super().__init__()
        return
    
    def setup_scene(self):
        world = self.get_world()
        world.add_task(FrankaDances(name="franka_dances"))
        return
    
    async def setup_post_load(self):
        self._world = self.get_world()
        self._franka = self._world.scene.get_object("my_roboarm")
        self._cube1 = self._world.scene.get_object("red_cube")
        self._cube2 = self._world.scene.get_object("green_cube")
        self._controller = StackingController(
            name="franka_stacy",
            gripper=self._franka.gripper,
            robot_articulation=self._franka,
            picking_order_cube_names=[self._cube1.name, self._cube2.name],
            robot_observation_name=self._franka.name
        )
        self._world.add_physics_callback("sim_step", callback_fn=self.physics_step)
        await self._world.play_async()
        return
    
    async def setup_post_reset(self):
        self._controller.reset()
        await self._world.play_async()
        return
    
    def physics_step(self, step_size):
        observations=self._world.get_observations()
        actions = self._controller.forward(
            observations=observations
        )
        self._franka.apply_action(actions)
        if self._controller.is_done():
            self._world.pause()
        return



