from omni.isaac.examples.base_sample import BaseSample
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.franka import Franka
from omni.isaac.franka.controllers import StackingController
import numpy as np


class HelloWorld(BaseSample):
    def __init__(self) -> None:
        super().__init__()
        return
    
    def setup_scene(self):
        world = self.get_world()
        world.scene.add_default_ground_plane()
        franka =  world.scene.add(Franka("/World/Roboarm", name="fancy_franka"))
        cube1 = world.scene.add(DynamicCuboid(
            prim_path="/World/red_cube",
            name="red_cube",
            color=np.array([1, 0, 0]),
            scale=np.array([0.0515, 0.0515, 0.0515]),
            position=np.array([0.3, 0.3, 0.3])
        ))
        cube2 = world.scene.add(DynamicCuboid(
            prim_path="/World/green_cube",
            name="green_cube",
            color=np.array([0, 1, 0]),
            scale=np.array([0.0515, 0.0515, 0.0515]),
            position=np.array([-0.2, -0.3, 0.3])))
        return
    
    async def setup_post_load(self):
        self._world = self.get_world()
        self._franka = self._world.scene.get_object("fancy_franka")
        self._cube1 = self._world.scene.get_object("red_cube")
        self._cube2 = self._world.scene.get_object("green_cube")
        self._controller = StackingController(
            name="stacking_controller",
            gripper=self._franka.gripper,
            robot_articulation=self._franka,
            picking_order_cube_names = ["red_cube", "green_cube"],
            robot_observation_name = "robot" 
            )
        
        self._world.add_physics_callback("sim_step", callback_fn=self.physics_step)
        self._franka.gripper.set_joint_positions(self._franka.gripper.joint_opened_positions)
        await self._world.play_async()
        return
    
    
    async def setup_post_reset(self):
        self._controller.reset()
        self._franka.gripper.set_joint_positions(self._franka.gripper.joint_opened_positions)
        await self._world.play_async()
        return
    

    def physics_step(self, step_size):
        cube1_position, _ = self._cube1.get_world_pose()
        cube2_position, _ = self._cube2.get_world_pose()
        current_joint_positions = self._franka.get_joint_positions()
        observations = {
            "red_cube" : {
                "position" : cube1_position,
                "target_position" : np.array([0.5, 0.5, 0.02575])
            },
            "green_cube" : {
                "position" : cube2_position,
                "target_position" : np.array([0.5, 0.5, 0.07725])
            },
            "robot" : {
                "joint_positions" : current_joint_positions
            }
        }
        actions = self._controller.forward(
            observations=observations
        )

        self._franka.apply_action(actions)
        if self._controller.is_done():
            self._world.pause()
        return
