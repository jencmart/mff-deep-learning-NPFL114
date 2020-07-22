#!/usr/bin/env python3
import argparse
import os

import numpy as np
import tensorflow as tf

import cart_pole_pixels_evaluator


# 47b0acaf-eb3e-11e9-9ce9-00505601122b
# 8194b193-e909-11e9-9ce9-00505601122b

def get_simple_cnn_GRU(input_shape, output_classes):
    # blocks = [3, 4, 6, 3]
    # R G B
    input_shape = [input_shape[0], input_shape[1], 1]
    in1 = tf.keras.Input(shape=input_shape)
    in2 = tf.keras.Input(shape=input_shape)
    in3 = tf.keras.Input(shape=input_shape)
    all_inputs = [in1, in2, in3]
    cnn_outputs = []
    layers = [tf.keras.layers.Conv2D(4, kernel_size=(2, 2), strides=2, activation='relu', padding='same', input_shape=input_shape),
                   tf.keras.layers.Conv2D(8, (2, 2), strides=2, padding='same', activation='relu'),
                   tf.keras.layers.Conv2D(16, (2, 2), strides=2, padding='same', activation='relu'),
                    tf.keras.layers.Flatten()]

    for input in all_inputs:
        for layer in layers:
            input = layer(input)
        cnn_outputs.append(input)

    x = tf.stack(cnn_outputs, 1)

    gru_out = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=16, return_sequences=True), merge_mode='sum')(x)
    gru_out = tf.keras.layers.Flatten()(gru_out)
    output = tf.keras.layers.Dense(output_classes, activation='softmax')(gru_out)
    model = tf.keras.Model(all_inputs, [output], name="CNN-LSTM-beast")
    return model


class BaselineNetwork:
    def __init__(self, shape, classes, args):
        model = get_simple_cnn_GRU(shape, classes)
        self.model = model
        o = tf.optimizers.Adam(lr=args.learning_rate_baseline)
        loss = 'mse'
        self.model.compile(optimizer=o, loss=loss)

    def train(self, states, returns):
        # states, returns = np.array(states, np.float32), np.array(returns, np.float32)
        self.model.train_on_batch(x=states, y=returns)

    def predict(self, states):
        # states = np.array(states, np.float32)
        results = self.model.predict_on_batch(x=states)
        if isinstance(results, np.ndarray):
            pass
        else:
            results = results.numpy()
        results = results[:, 0]
        return results


class Network:
    def __init__(self, env, args):
        if '3D' in args.network:
            print("3D !!!")
            states_shape = [args.image_size, args.image_size, 3, 1]
            self.conv3d = True
        else:
            states_shape = [args.image_size, args.image_size, 3]  # env.state_shape  [80,80,3]
            self.conv3d = False

        self.resize_to = args.image_size
        self.use_baseline = args.use_baseline
        classes = env.actions

        model = get_simple_cnn_GRU(states_shape, classes)
        self.use_gru = True
        # Use Adam optimizer with given `args.learning_rate`
        self.model = model
        loss = 'categorical_crossentropy'
        self.model.compile(optimizer=tf.optimizers.Adam(lr=args.learning_rate), loss=loss)  # mse
        if self.use_baseline:
            self.baseline_network = BaselineNetwork(shape=states_shape, classes=1, args=args)


    def deal_with_states(self, states):
        # [batch, 80, 80, 3] -> [batch,x,x,3]
        states = tf.image.resize(states, [self.resize_to, self.resize_to]).numpy()
        if self.conv3d:  # todo - expand dim
            states = np.expand_dims(states, axis=states.ndim)

        if self.use_gru:
            # states = np.moveaxis(states, 3, 1)
            # print(states.shape)
            states1 = states[:,:,:,0]
            states1 = np.expand_dims(states1, axis=3)
            # exit(1)
            states2 = states[:,:,:,1]
            states2 = np.expand_dims(states2, axis=3)

            states3 = states[:,:,:,2]
            states3 = np.expand_dims(states3, axis=3)

            states = [states1, states2, states3]

        return states

    def train(self, states, actions, returns):
        # mnist = 28x28
        # cifar = 32x32
        states, actions, returns = np.array(states, np.float32), np.array(actions, np.int32), np.array(returns, np.float32)
        # Train the model using the states, actions and observed returns by calling `train_on_batch`.
        # States  [ batch , 4 ]
        # Actions [ batch , 2 ]
        # returns [ batch , 1 ]

        states = self.deal_with_states(states)

        onehot_actions = np.zeros((actions.size, 2), dtype=np.int32)
        onehot_actions[np.arange(actions.size), actions] = 1


        # todo: first train the baseline
        if self.use_baseline:
            self.baseline_network.train(states, returns)

        # todo: predict baseline using baseline network
        if self.use_baseline:
            baseline = self.baseline_network.predict(states)
            returns -= baseline

        self.model.train_on_batch(x=states, y=onehot_actions, sample_weight=returns)

    def predict(self, states):
        states = np.array(states, np.float32)
        states = self.deal_with_states(states)

        # Predict distribution over actions for the given input states
        # using the `predict_on_batch` method and calling `.numpy()` on the result to return a NumPy array.
        results = self.model.predict_on_batch(x=states)
        if isinstance(results, np.ndarray):
            pass
        else:
            results = results.numpy()
        return results


def calculate_returns(rewards, gamma=0.99):
    # apply discount [1, 0.99, 0.98, 0.97]
    returns = np.array([gamma ** i * rewards[i] for i in range(len(rewards))])
    returns = np.cumsum(returns[::-1])[::-1]
    return returns


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
    parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
    parser.add_argument("--seed", default=1, type=int, help="Random seed.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--verbose", default=False, action="store_true", help="Verbose TF logging.")

    parser.add_argument("--use_baseline", default=False, action="store_true", help="Use baseline prediciton or not")
    parser.add_argument("--image_size", default=32, type=int, help="80 x 80 images too big ... resize to ..")
    parser.add_argument("--network", default='gru', help="What network to use resnet/vgg")
    parser.add_argument("--episodes", default=900, type=int, help="Training episodes.")
    parser.add_argument("--batch_size", default=20, type=int, help="Number of episodes to train on.")
    parser.add_argument("--gamma", default=0.99, type=float, help="gamma for discount.")
    parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate.")
    parser.add_argument("--network_baseline", default='simple', help="What network to use resnet/vgg")
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
        tf.get_logger().setLevel("ERROR")

    # Create the environment
    env = cart_pole_pixels_evaluator.environment(seed=args.seed)
    possible_actions = list(range(env.actions))

    # Construct the network
    network = Network(env, args)

    # Training
    for _ in range(args.episodes // args.batch_size):
        batch_states, batch_actions, batch_returns = [], [], []

        # Batch over multiple episodes (failed / finished)
        for _ in range(args.batch_size):
            # Perform episode
            states, actions, rewards = [], [], []

            state, done = env.reset(), False
            while not done:

                # render image
                if args.render_each and env.episode > 0 and env.episode % args.render_each == 0:
                    env.render()

                # select action proportional to probability distribution given by the network
                predicitons = network.predict([state])[0]
                action = np.random.choice(a=possible_actions, p=predicitons)
                next_state, reward, done, _ = env.step(action)

                # append state action reward
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                state = next_state

            batch_states += states
            batch_actions += actions
            batch_returns += calculate_returns(rewards, gamma=args.gamma).tolist()
        network.train(batch_states, batch_actions, batch_returns)
        # if np.mean(env._episode_returns[-100:]) > 350:
        #     break

    # Final evaluation
    while True:
        state, done = env.reset(True), False
        while not done:
            probabilities = network.predict([state])[0]
            action = np.argmax(probabilities)
            state, reward, done, _ = env.step(action)




# not 0, 42