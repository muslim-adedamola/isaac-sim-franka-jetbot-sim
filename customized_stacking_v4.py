#customized stacking with jetbots included

from omni.isaac.examples.base_sample import BaseSample
from omni.isaac.franka.tasks import Stacking
from omni.isaac.wheeled_robots.robots import WheeledRobot
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.wheeled_robots.controllers.wheel_base_pose_controller import WheelBasePoseController
from omni.isaac.wheeled_robots.controllers.differential_controller import DifferentialController
from omni.isaac.core.tasks import BaseTask
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.franka.controllers import StackingController
import numpy as np

class RobotsDancing(BaseTask):
    def __init__(self, name):
        super().__init__(name=name, offset=None)
        self._jetbot_goal_position = np.array([1.3, 0.3, 0])
        self._jetbot1_goal_position = np.array([1.3, -0.3, 0])
        self._task_event = 0
        self._stack_task = Stacking(name="nice_task", target_position=np.array([0.7, -0.3, 0.0515/2.0]))
        return
    

    def set_up_scene(self, scene):
        super().set_up_scene(scene)
        self._stack_task.set_up_scene(scene)
        assets_root_path = get_assets_root_path()
        jetbot_asset_path = assets_root_path + "/Isaac/Robots/Jetbot/jetbot.usd"
        self._jetbot = scene.add(
            WheeledRobot(
                prim_path="/World/Fancy_Robot",
                name="fancy_robot",
                wheel_dof_names=["left_wheel_joint", "right_wheel_joint"],
                create_robot=True,
                usd_path=jetbot_asset_path,
                position=np.array([0, 0.3, 0]))
        )
        self._jetbot1 = scene.add(
            WheeledRobot(
                prim_path="/World/Fancy_Jetbot",
                name="fancy_jetbot",
                wheel_dof_names=["left_wheel_joint", "right_wheel_joint"],
                create_robot=True,
                usd_path=jetbot_asset_path,
                position=np.array([0, -0.3, 0]))
        )
        # cube_names = self._stack_task.get_cube_names()
        stack_params = self._stack_task.get_params()
        self._franka = scene.get_object(stack_params["robot_name"]["value"])
        # Changes Franka's default position
        # so that it is set at this position after reset
        self._franka.set_world_pose(position=np.array([1.0, 0, 0]))
        self._franka.set_default_state(position=np.array([1.0, 0, 0]))
        return

    def get_params(self):
        # To avoid hard coding names..etc.
        stack_params = self._stack_task.get_params()
        params_representation = stack_params
        params_representation["jetbot_name"] = {"value": self._jetbot.name, "modifiable": False}
        params_representation["jetbot1_name"] = {"value": self._jetbot1.name, "modifiable": False}
        params_representation["franka_name"] = stack_params["robot_name"]
        return params_representation
    
    def get_cube_names(self):
        cube_names = self._stack_task.get_cube_names()
        return cube_names

    def get_observations(self):
        current_jetbot_position, current_jetbot_orientation = self._jetbot.get_world_pose()
        current_jetbot1_position, current_jetbot1_orientation = self._jetbot1.get_world_pose()
        observations= {
            "task_event" : self._task_event,
            self._jetbot.name: {
                "position": current_jetbot_position,
                "orientation": current_jetbot_orientation,
                "goal_position": self._jetbot_goal_position
            },
            self._jetbot1.name: {
                "position": current_jetbot1_position,
                "orientation": current_jetbot1_orientation,
                "goal_position": self._jetbot1_goal_position
            }
        }
        observations.update(self._stack_task.get_observations())
        return observations

    def pre_step(self, control_index, simulation_time):
        current_jetbot_position, _ = self._jetbot.get_world_pose()
        current_jetbot1_position, _ = self._jetbot1.get_world_pose()
        if self._task_event == 0:
            #check if first jetbot reah destination
            if np.mean(np.abs(current_jetbot_position[:2] - self._jetbot_goal_position[:2])) < 0.04:
                self._task_event += 1
                self._cube1_arrive_step_index = control_index
        elif self._task_event == 1:
            # Jetbot has 300 time steps to back off from Franka and stop
            if control_index - self._cube1_arrive_step_index == 300:
                self._task_event += 1
        elif self._task_event == 2:
            #second jetbot starts and checks if destination is reached
            if np.mean(np.abs(current_jetbot1_position[:2] - self._jetbot1_goal_position[:2])) < 0.04:
                self._task_event += 1
                self._cube2_arrive_step_index = control_index
        elif self._task_event == 3:
            if control_index - self._cube2_arrive_step_index == 300:
                self._task_event += 1
        return

    def post_reset(self):
        self._franka.gripper.set_joint_positions(self._franka.gripper.joint_opened_positions)
        self._task_event = 0
        return



class HelloWorld(BaseSample):
    def __init__(self) -> None:
        super().__init__()
        return

    def setup_scene(self):
        world = self.get_world()
        # Add task here
        robots_dancing_task = RobotsDancing(name="nice_task")
        world.add_task(robots_dancing_task)
        return
    
    async def setup_post_load(self):
        self._world = self.get_world()
        task = self._world.get_task("nice_task")
        task_params = task.get_params()
        self._jetbot = self._world.scene.get_object(task_params["jetbot_name"]["value"])
        self._jetbot1 = self._world.scene.get_object(task_params["jetbot1_name"]["value"])
        self._franka = self._world.scene.get_object(task_params["franka_name"]["value"])
        task.get_cube_names()
        self._jetbot_controller = WheelBasePoseController(
            name="cool_controller",
            open_loop_wheel_controller=DifferentialController(
                name="simple_control",
                wheel_radius=0.03,
                wheel_base=0.1125
            ))
        self._jetbot1_controller = WheelBasePoseController(
            name="cool_controller1",
            open_loop_wheel_controller=DifferentialController(
                name="simple_control1",
                wheel_radius=0.03,
                wheel_base=0.1125
            ))
        self._franka_controller = StackingController(
            name="franka_stacks",
            gripper=self._franka.gripper,
            robot_articulation = self._franka,
            picking_order_cube_names = task.get_cube_names(),
            robot_observation_name = self._franka.name
        )
        self._world.add_physics_callback("sim_step", callback_fn=self.physics_step)
        await self._world.play_async()
        return
    
    async def setup_post_reset(self):
        self._franka_controller.reset()
        self._jetbot_controller.reset()
        self._jetbot1_controller.reset()
        await self._world.play_async()
        return
    
    def physics_step(self, step_size):
        current_observations = self._world.get_observations()
        if current_observations["task_event"] == 0:
            self._jetbot.apply_wheel_actions(
                self._jetbot_controller.forward(
                    start_position=current_observations[self._jetbot.name]["position"],
                    start_orientation=current_observations[self._jetbot.name]["orientation"],
                    goal_position=current_observations[self._jetbot.name]["goal_position"]))
        elif current_observations["task_event"] == 1:
            self._jetbot.apply_wheel_actions(ArticulationAction(joint_velocities=[-8, -8]))
        elif current_observations["task_event"] == 2:
            self._jetbot.apply_wheel_actions(ArticulationAction(joint_velocities=[0.0, 0.0]))
            # Pick up the block
            self._jetbot1.apply_wheel_actions(
                self._jetbot1_controller.forward(
                    start_position=current_observations[self._jetbot1.name]["position"],
                    start_orientation=current_observations[self._jetbot1.name]["orientation"],
                    goal_position=current_observations[self._jetbot1.name]["goal_position"]))
        elif current_observations["task_event"] == 3:
            self._jetbot1.apply_wheel_actions(ArticulationAction(joint_velocities=[-8, -8]))
        elif current_observations["task_event"] == 4:
            self._jetbot1.apply_wheel_actions(ArticulationAction(joint_velocities=[0.0, 0.0]))
            actions = self._franka_controller.forward(
                observations=current_observations
                )
            self._franka.apply_action(actions)
        # Pause once the controller is done
        if self._franka_controller.is_done():
            self._world.pause()
        return
