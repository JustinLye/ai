from dataclasses import dataclass


@dataclass
class configuration():
    learning_rate: float
    minibatch_size: int
    discount_factor: float
    number_of_episodes: int
    max_number_of_steps_per_episode: int
    epsilon_start: float
    epsilon_stop: float
    epsilon_decay: float
    gymnasium_environment: str

