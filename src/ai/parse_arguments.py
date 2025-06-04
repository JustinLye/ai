import argparse

from ai.configuration import configuration

def parse_arguments(argument_list:list) -> configuration:
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning-rate", type=float, help="Learning rate.", default=5e-4)
    parser.add_argument("--minibatch-size", type=int, help="Minibatch size", default=64)
    parser.add_argument("--discount-factor", type=float, help="Discount factor", default=0.99)
    parser.add_argument("--number-of-episodes", type=int, help="Number of episodes", default=2000)
    parser.add_argument("--max-number-of-steps-per-episode", type=int, help="Max number of steps per episode", default=10000)
    parser.add_argument("--epsilon-start", type=float, help="Epsilon start value", default=1.0)
    parser.add_argument("--epsilon-stop", type=float, help="Minimum value to decay epsilon to", default=0.01)
    parser.add_argument("--epsilon-decay", type=float, help="Epsilon decay value", default=0.995)
    parser.add_argument("--gymnasium-environment", type=str, help="Gymnasium environment name", default="MsPacmanNoFrameskip-v0")
    args = parser.parse_args(argument_list)

    return configuration(
        learning_rate=args.learning_rate,
        minibatch_size=args.minibatch_size,
        discount_factor=args.discount_factor,
        number_of_episodes=args.number_of_episodes,
        max_number_of_steps_per_episode=args.max_number_of_steps_per_episode,
        epsilon_start=args.epsilon_start,
        epsilon_stop=args.epsilon_stop,
        epsilon_decay=args.epsilon_decay,
        gymnasium_environment=args.gymnasium_environment
    )