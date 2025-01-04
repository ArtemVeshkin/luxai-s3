from enum import Enum
from typing import Dict
from action_type import ActionType
import tasks


class TaskType(Enum):
    SLEEP = 0
    FIND_RELICS = 1
    FIND_REWARDS = 2
    HARVEST = 3
    ATTACK = 4

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name
    
    @property
    def check_fn(self):
        return {
            TaskType.SLEEP: tasks.sleep_check_fn,
            TaskType.FIND_RELICS: tasks.find_relics_check_fn,
            TaskType.FIND_REWARDS: tasks.find_rewards_check_fn,
            TaskType.HARVEST: tasks.harvest_check_fn,
            TaskType.ATTACK: tasks.attack_check_fn,
        }[self]

    @property
    def run_fn(self):
        return {
            TaskType.SLEEP: tasks.sleep_run_fn,
            TaskType.FIND_RELICS: tasks.find_relics_run_fn,
            TaskType.FIND_REWARDS: tasks.find_rewards_run_fn,
            TaskType.HARVEST: tasks.harvest_run_fn,
            TaskType.ATTACK: tasks.attack_run_fn,
        }[self]


class Task:
    def __init__(self):
        self.task_type: TaskType = TaskType.SLEEP
        self.params: Dict = {}


    def __repr__(self):
        return f'TaskType={self.task_type}, params={self.params}'
    

    def check_can_set_task(self, task_type: TaskType, params: Dict = {}) -> bool:
        return task_type.check_fn(params)


    def try_set_task(self, task_type: TaskType, params: Dict = {}) -> bool:
        can_set_task = self.check_can_set_task(task_type, params)
        if can_set_task:
            self.task_type = task_type
            self.params = params
        return can_set_task


    def run_task(self) -> ActionType:
        return self.task_type.run_fn(self.params)
        
    

    def clean(self):
        self.task_type = TaskType.SLEEP
        self.params = {}
