#!/usr/bin/env python3
import argparse
import os

import numpy as np
import tensorflow as tf

import cart_pole_evaluator

# 47b0acaf-eb3e-11e9-9ce9-00505601122b
# 8194b193-e909-11e9-9ce9-00505601122b


class BaselineNetwork:
    def __init__(self, env, args):
        inputs = tf.keras.layers.Input(shape=env.state_shape)
        x = tf.keras.layers.Dense(units=args.hidden_layer_baseline, activation=tf.nn.relu)(inputs)
        outputs = tf.keras.layers.Dense(units=1)(x)  # single output
        self.model = tf.keras.Model([inputs], [outputs], name="baseline_model")

        o = tf.optimizers.Adam(lr=args.learning_rate_baseline)
        self.model.compile(optimizer=o, loss='mse')

    def train(self, states, returns):
        states, returns = np.array(states, np.float32), np.array(returns, np.float32)
        # States  [ batch , 4 ]
        # returns [ batch , 1 ]

        self.model.train_on_batch(x=states, y=returns)

    def predict(self, states):
        states = np.array(states, np.float32)
        results = self.model.predict_on_batch(x=states)
        results = results.numpy()
        results = results[:, 0]
        return results

class Network:
    def __init__(self, env, args):

        # The inputs have shape `env.state_shape`,
        # and the model should produce probabilities of `env.actions` actions.
        # You can use for example one hidden layer with `args.hidden_layer` and non-linear activation.
        inputs = tf.keras.layers.Input(shape=env.state_shape)
        x = tf.keras.layers.Dense(units=args.hidden_layer)(inputs)
        # dropout
        # x = tf.keras.layers.Dropout(rate=0.5)(x)
        x = tf.keras.activations.relu(x)
        outputs = tf.keras.layers.Dense(units=env.actions, activation=tf.nn.softmax)(x)
        self.model = tf.keras.Model([inputs], [outputs], name="reinforce_model")

        o = tf.optimizers.Adam(lr=args.learning_rate)
        self.model.compile(optimizer=o, loss='mse')

        self.baseline_network = BaselineNetwork(env, args)

    def train(self, states, actions, returns):
        states, actions, returns = np.array(states, np.float32), np.array(actions, np.int32), np.array(returns, np.float32)
        # States  [ batch , 4 ]
        # Actions [ batch , 2 ]
        # returns [ batch , 1 ]

        # todo: first train the baseline
        self.baseline_network.train(states, returns)

        onehot_actions = np.zeros((actions.size, actions.max() + 1), dtype=np.int32)
        onehot_actions[np.arange(actions.size), actions] = 1

        # todo: predict baseline using baseline network
        baseline = self.baseline_network.predict(states)

        returns -= baseline
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
    # Episode 150, mean 100-episode return 82.93 +-56.84
    # Episode 150, mean 100-episode return 88.21 +-51.93
    # Episode 150, mean 100-episode return 97.98 +-62.36
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", default=200, type=int, help="Training episodes.")
    parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
    parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--verbose", default=False, action="store_true", help="Verbose TF logging.")

    # batch_size
    # 15 -- ??
    # 10 -- ok
    # 07 -- ok -----------
    # 05 -- shitty
    parser.add_argument("--batch_size", default=10, type=int, help="Number of episodes to train on.")

    # GAMMA DISCOUNT
    # 1.00 -- super bad
    # 0.995 -- super bad
    # 0.99 -- ok
    # 0.985 -- super bad
    # 0.98 -- super bad
    parser.add_argument("--gamma", default=0.99, type=float, help="gamma for discount.")

    # Reinforce network
    # 512 -- too much
    # 270 -- too much ?
    # 256 -- ok
    # 220 - too little
    # 128 -- too little

    # -- RECODEX ---
    # 280 -- 212.060  // 311.050
    # 256 -- 497.160  // 235.010
    # 240 -- 500.000  // 226.640
    # 239 --
    # 238 -- 500.000  /// 478.040
    # 237 -- 444      // 41.570
    # 236 -- 46.780   // 239
    # 235 -- 325.630  // 499.560
    # 230 -- 177.950  // 500.000
    # 220 -- 104.080  // 457
    parser.add_argument("--hidden_layer", default=238, type=int, help="Size of hidden layer.")
    # 0.01 -- ok
    parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate.")
    # Baseline network
    # 512 -- too much
    # 256 -- ok
    # 220 -- awesome !!!
    # 200 -- better
    # 190 -- worse
    # 128 -- too little
    parser.add_argument("--hidden_layer_baseline", default=220, type=int, help="Size of hidden layer.")
    # 0.02 -- to much
    # 0.01 -- ok
    parser.add_argument("--learning_rate_baseline", default=0.01, type=float, help="Learning rate.")

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

                probabilities = network.predict([state])[0]
                action = np.random.choice(a=possible_actions, p=probabilities)
                next_state, reward, done, _ = env.step(action)

                states.append(state)
                actions.append(action)
                rewards.append(reward)

                state = next_state

            returns = calculate_returns(rewards, gamma=args.gamma, subs_mean=False)
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
