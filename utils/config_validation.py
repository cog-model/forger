from pydantic import BaseModel, Extra
from typing import List


class BufferCfg(BaseModel, extra=Extra.forbid):
    size: int = 450000
    episodes_to_decay: int = 50
    min_demo_proportion: float = 0.0
    cpp: bool = False


class WrappersCfg(BaseModel, extra=Extra.forbid):
    frame_stack: int = 2
    frame_skip: int = 4
    render: bool = False


class AgentCfg(BaseModel, extra=Extra.forbid):
    episodes: int = 250
    save_dir: str = None

    frames_to_update: int = 2000
    update_quantity: int = 600
    update_target_net_mod: int = 3000
    replay_start_size: int = 20000

    batch_size: int = 32
    gamma: float = 0.99
    n_step: int = 10
    l2: float = 1e-5
    margin: float = 0.4
    learning_rate: float = 0.0001

    # eps-greedy
    initial_epsilon: float = 0.1
    final_epsilon: float = 0.01
    epsilon_time_steps: int = 100000


class Action(BaseModel):
    name: str = None
    target: str = None


class Subtask(BaseModel):
    item_name: str = None
    item_count: int = None
    start_idx: int = None
    end_idx: int = None
    actions: List[Action] = []


class GlobalCfg(BaseModel, extra=Extra.forbid):
    buffer: BufferCfg = BufferCfg()
    wrappers: WrappersCfg = WrappersCfg()
    agent: AgentCfg = AgentCfg()


class Task(BaseModel, extra=Extra.forbid):
    evaluation: bool = False
    environment: str = "MineRLTreechop-v0"
    max_train_steps: int = 1000000
    max_train_episodes: int = 1000000000
    pretrain_num_updates: int = 1000000
    source: str = None
    from_scratch: bool = False
    agent_type: str = None
    cfg: GlobalCfg = GlobalCfg()
    subtasks: List[Subtask] = None
    data_dir: str = 'demonstrations'
    model_name: str = 'tf1_minerl_dqfd'


class Pipeline(BaseModel, extra=Extra.forbid):
    pipeline: List[Task] = None
