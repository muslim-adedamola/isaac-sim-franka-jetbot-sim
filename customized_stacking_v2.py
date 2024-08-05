#customized stacking from Pick and Place Tasks with Stacking Controller

from omni.isaac.examples.base_sample import BaseSample
from omni.isaac.franka.tasks import Stacking
from omni.isaac.franka.controllers import StackingController

class HelloWorld(BaseSample):
    def __init__(self) -> None:
        super().__init__()
        return
    
    def setup_scene(self):
        world = self.get_world()
        world.add_task(Stacking(name="franka_stacks"))
        return
    
    async def setup_post_load(self):
        self._world = self.get_world()
        self._franka_task = self._world.get_task("franka_stacks")
        task_params = self._franka_task.get_params()
        self._franka = self._world.scene.get_object(task_params["robot_name"]["value"])
        self._controller = StackingController(
            name="stacking_controller",
            gripper=self._franka.gripper,
            robot_articulation=self._franka,
            picking_order_cube_names=self._franka_task.get_cube_names(),
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
        observations = self._franka_task.get_observations()
        actions = self._controller.forward(
            observations = observations
        )
        self._franka.apply_action(actions)
        if self._controller.is_done():
            self._world().pause()
        return
    
