#!/usr/bin/env python3
import argparse

import numpy as np

import cart_pole_evaluator

# 47b0acaf-eb3e-11e9-9ce9-00505601122b
# 8194b193-e909-11e9-9ce9-00505601122b

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    # TODO: Define reasonable defaults and optionally more parameters
    parser.add_argument("--episodes", default=5000, type=int, help="Training episodes.")
    parser.add_argument("--epsilon", default=0.2, type=float, help="Exploration factor.")
    parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
    parser.add_argument("--render_each", default=200, type=int, help="Render some episodes.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Fix random seeds and threads
    np.random.seed(args.seed)

    # Create the environment
    env = cart_pole_evaluator.environment(discrete=True, seed=args.seed)

    # Create Q, C and other variables
    S = env.states
    A = env.actions
    Q = np.zeros([S, A], dtype=np.float)
    C = np.zeros([S, A], dtype=np.int)
    possible_actions = list(range(A))

    for _ in range(args.episodes):
        # Perform episode
        state = env.reset()
        states, actions, rewards = [], [], []

        while True:
            if args.render_each and env.episode > 0 and env.episode % args.render_each == 0:
                env.render()

            # `action` using epsilon-greedy policy:
            # P(random) = args.epsilon
            # otherwise, choose and action with maximum Q[state, action].
            if np.random.rand() <= args.epsilon:
                action = np.random.randint(0, len(possible_actions))
            else:
                action = np.argmax(Q[state])
            next_state, reward, done, _ = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)

            state = next_state
            if done:
                break

        G = 0  # returns from rewards
        # Update Q and C
        for t in reversed(range(len(actions))):
            G += rewards[t]
            C[states[t], actions[t]] += 1
            Q[states[t], actions[t]] += 1 / C[states[t], actions[t]] * (G - Q[states[t], actions[t]])

    # Final evaluation
    while True:
        state, done = env.reset(True), False
        while not done:
            action = np.argmax(Q[state])
            state, reward, done, _ = env.step(action)
