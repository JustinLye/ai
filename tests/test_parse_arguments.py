
from ai.parse_arguments import parse_arguments

def test_default_arguments():
    config = parse_arguments([])
    assert(5e-4 == config.learning_rate)
    assert(64 == config.minibatch_size)
    assert(0.99 == config.discount_factor)
    assert(2000 == config.number_of_episodes)
    assert(10000 == config.max_number_of_steps_per_episode)
    assert(1.0 == config.epsilon_start)
    assert(0.01 == config.epsilon_stop)
    assert(0.995 == config.epsilon_decay)
    assert("MsPacmanNoFrameskip-v0" == config.gymnasium_environment)

def test_should_use_learning_rate():
    config = parse_arguments(["--learning-rate", "0.0123"])
    assert(0.0123 == config.learning_rate)

def test_should_use_minibatch_size():
    config = parse_arguments(["--minibatch-size", "9999"])
    assert(9999 == config.minibatch_size)

def test_should_use_discount_factor():
    config = parse_arguments(["--discount-factor", "2.4553"])
    assert(2.4553 == config.discount_factor)

def test_should_use_number_of_episodes():
    config = parse_arguments(["--number-of-episodes", "3322"])
    assert(3322 == config.number_of_episodes)

def test_should_use_max_number_of_steps_per_episode():
    config = parse_arguments(["--max-number-of-steps-per-episode", "4532"])
    assert(4532 == config.max_number_of_steps_per_episode)

def test_should_use_epsilon_start():
    config = parse_arguments(["--epsilon-start", "999.454"])
    assert(999.454 == config.epsilon_start)

def test_should_use_epsilon_stop():
    config = parse_arguments(["--epsilon-stop", "0.8888"])
    assert(0.8888 == config.epsilon_stop)

def test_should_use_epsilon_decay():
    config = parse_arguments(["--epsilon-decay", "73.432"])
    assert(73.432 == config.epsilon_decay)

def test_should_use_gym_environment_name():
    config = parse_arguments(["--gymnasium-environment", "blah"])
    assert("blah" == config.gymnasium_environment)