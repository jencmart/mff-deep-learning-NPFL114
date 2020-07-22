#!/usr/bin/env python3
import argparse
import os

import numpy as np
import tensorflow as tf

import cart_pole_evaluator

# 47b0acaf-eb3e-11e9-9ce9-00505601122b
# 8194b193-e909-11e9-9ce9-00505601122b


class Network:
    def __init__(self, env, args):

        # The inputs have shape `env.state_shape`,
        # and the model should produce probabilities of `env.actions` actions.
        # You can use for example one hidden layer with `args.hidden_layer` and non-linear activation.
        inputs = tf.keras.layers.Input(shape=env.state_shape)
        x = tf.keras.layers.Dense(units=args.hidden_layer, activation=tf.nn.relu)(inputs)
        outputs = tf.keras.layers.Dense(units=env.actions, activation=tf.nn.softmax)(x)
        self.model = tf.keras.Model([inputs], [outputs], name="reinforce_model")

        # Use Adam optimizer with given `args.learning_rate`.
        o = tf.optimizers.Adam(lr=args.learning_rate)
        self.model.compile(optimizer=o, loss='mse')  # mse??
        # model = tf.keras.Sequential()
        # model.add(tf.keras.layers.Dense(8))
        # model.add(tf.keras.layers.Dense(1))
        # model.compile(optimizer='adam', loss='mse')

    def train(self, states, actions, returns):
        states, actions, returns = np.array(states, np.float32), np.array(actions, np.int32), np.array(returns, np.float32)
        # Train the model using the states, actions and observed returns by calling `train_on_batch`.
        # States  [ batch , 4 ]
        # Actions [ batch , 2 ]
        # returns [ batch , 1 ]

        onehot_actions = np.zeros((actions.size, actions.max() + 1), dtype=np.int32)
        onehot_actions[np.arange(actions.size), actions] = 1

        self.model.train_on_batch(x=states, y=onehot_actions, sample_weight=returns)

    def predict(self, states):
        states = np.array(states, np.float32)
        # Predict distribution over actions for the given input states
        # using the `predict_on_batch` method and calling `.numpy()` on the result to return a NumPy array.
        results = self.model.predict_on_batch(x=states)
        results = results.numpy()
        return results


def calculate_returns(rewards, gamma=0.99, subs_mean=False):
    # apply discount [1, 0.99, 0.98, 0.97]
    returns = np.array([gamma ** i * rewards[i] for i in range(len(rewards))])
    # cumsum from back:
    # [1,1,1,1,1] -> [5,4,3,2,1]
    # [1, 0.99, 0.98, 0.97]   --> [5, 4, 3.1, 2.2, 1.3]
    returns = np.cumsum(returns[::-1])[::-1]
    if subs_mean:
        returns -= returns.mean()
    return returns


if __name__ == "__main__":
    # Parse arguments
    # SUBTRACT MEAN --- better not to

    # DENSE
    # 512 -- too much (for 500 episodes...)
    # 256 -- ok
    # 128 -- too little

    # EPISODES
    # 300 -- too litle
    # 400 -- on the edge
    # 500 -- enough
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=10, type=int, help="Number of episodes to train on.")
    parser.add_argument("--episodes", default=500, type=int, help="Training episodes.")
    parser.add_argument("--hidden_layer", default=256, type=int, help="Size of hidden layer.")
    parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate.")
    parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
    parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--verbose", default=False, action="store_true", help="Verbose TF logging.")
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Fix random seeds and threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Report only errors by default
    if not args.verbose:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    # Create the environment
    env = cart_pole_evaluator.environment(discrete=False, seed=args.seed)
    possible_actions = list(range(env.actions))

    # Construct the network
    network = Network(env, args)

    # Training
    for _ in range(args.episodes // args.batch_size):
        batch_states, batch_actions, batch_returns = [], [], []
        for _ in range(args.batch_size):

            # Perform episode
            states, actions, rewards = [], [], []
            state, done = env.reset(), False
            while not done:
                if args.render_each and env.episode > 0 and env.episode % args.render_each == 0:
                    env.render()

                # `action` according to the distribution
                probabilities = network.predict([state])[0]
                action = np.random.choice(a=possible_actions, p=probabilities)
                next_state, reward, done, _ = env.step(action)
                states.append(state)
                actions.append(action)
                rewards.append(reward)

                state = next_state

            # Compute `returns` from the observed `rewards`.
            returns = calculate_returns(rewards)
            batch_states += states
            batch_actions += actions
            batch_returns += returns.tolist()

        network.train(batch_states, batch_actions, batch_returns)

    # Final evaluation
    while True:
        state, done = env.reset(True), False
        while not done:
            probabilities = network.predict([state])[0]
            action = np.argmax(probabilities)
            state, reward, done, _ = env.step(action)