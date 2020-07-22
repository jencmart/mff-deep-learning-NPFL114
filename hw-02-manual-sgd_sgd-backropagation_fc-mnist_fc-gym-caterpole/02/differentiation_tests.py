import argparse
import datetime
import os
import re
import sys

import numpy as np
import tensorflow as tf


class DifferTest:

    def __init__(self):
        self._W1 = tf.Variable(tf.random.normal([3, 4], stddev=0.1, seed=17), trainable=True)
        self._b1 = tf.Variable(tf.ones([4]), trainable=True)

        self._W2 = tf.Variable(tf.random.normal([4, 2], stddev=0.1, seed=45), trainable=True)
        self._b2 = tf.Variable(tf.ones([2]), trainable=True)


    def get_Si(self, i, probabilities):
        return tf.gather_nd(probabilities, [0, i])

    def differentiate_softmax_wrt_logit(self, probabilities):

        # R^T -> R^T
        t = probabilities.shape[1]  # because it is row vector

        # output will be matrix (t,t)
        A = np.zeros((t, t))

        for i in range(t):
            for j in range(t):
                S_i = tf.gather_nd(probabilities, [0, i])
                S_j = tf.gather_nd(probabilities, [0, j])

                if i == j:
                    DjSi = S_i * (1.0 - S_j)
                else:
                    DjSi = -1.0 * S_j * S_i
                A[i, j] = DjSi  # podle i-teho outputu .. ano je to spravne

        return tf.constant(A)

    def differentiate_xW_by_W(self, x, W):
        DW = []

        # R^N -> R^T
        # first row -- whole W wrt first output

        # init empty array at first ...
        # output will be matrix (t,Nt)
        N = W.shape[0]
        t = W.shape[1]
        A = np.zeros((t, N*t))

        # set the correct values
        for i in range(t):
            for j in range(0,  N):
                A[i, i*N+j] = tf.gather_nd(x, [0, j])

        return tf.constant(A)

    def differentiate_xent_wrt_probabilites(self, onehot_y, probabilities):

        # vyrobime matici kde na radku vzdy jedno cislo == 1/-p_y
        vynulovane = tf.math.multiply(onehot_y, probabilities)
        delenec = tf.ones([vynulovane.shape[0], vynulovane.shape[1]], tf.float32)
        result = tf.math.scalar_mul(-1.0, tf.divide(delenec, vynulovane))
        result = tf.dtypes.cast(result, dtype=tf.float64)

        # remove nan or inf ...
        mask = tf.math.is_finite(result)
        result = tf.where(mask, result, tf.zeros_like(result))

        return result

    def differentiate_vect_tanh_wrt_vect(self, vector_before_tanh):
        #  tf.math.divide(tf.constant(1.0), tf.add(tf.constant(1.0), tf.exp(tf.negative(x))))
        # R^T -> R^T

        # output will be matrix (t,t)
        t = vector_before_tanh.shape[1]  # because it is row vector
        A = np.zeros((t, t))

        for i in range(t):
            cosh = np.cosh(tf.gather_nd(vector_before_tanh, [0, i]))  # todo  I think nonzero only on diagonal
            A[i, i] = 1 / ( cosh ** 2 )
        return (tf.constant(A))


    def differentiate_xW_wrt_b(self, W):
        DW = []

        # R^N -> R^T
        # first row -- whole W wrt first output

        # init empty array at first ...
        # output will be matrix (t,Nt)
        N = W.shape[0]
        t = W.shape[1]  # 2
        A = np.zeros((t, N * t))

        # set the correct values
        for i in range(t):
            for j in range(0, N):
                A[i, i * N + j] = 1.0

        # print(A)
        return tf.constant(A)

    def differentiate_xW_wrt_x(self, W):

        # R^N -> R^T
        # first row -- whole W wrt first output

        # init empty array at first ...
        # output will be matrix (t,N)
        N = W.shape[0]
        t = W.shape[1]
        A = np.zeros((t, N))

        # set the correct values
        for j in range(t):
            for i in range(N):
                A[j, i] = tf.gather_nd(W, [i, j])  # is this just transposition of W ??

        return (tf.constant(A))

    def predict_for_train(self, input_vector):
        # dense + tanh
        hidden_layer_before_tanh = tf.matmul(input_vector, self._W1)
        hidden_layer_before_tanh = tf.add(hidden_layer_before_tanh, self._b1)

        hidden_layer = tf.nn.tanh(hidden_layer_before_tanh)

        # dense + softmax
        output_layer = tf.matmul(hidden_layer, self._W2)
        output_layer = tf.add(output_layer, self._b2)
        probabilities = tf.math.softmax(output_layer)

        return probabilities, hidden_layer_before_tanh, hidden_layer

    def differentiate_tape(self, y, batch_x):
        with tf.GradientTape() as tape:
            probabilities, hidden_layer_before_tanh, hidden_layer = self.predict_for_train(batch_x)

            # xent
            vynulovane = tf.math.multiply(y, probabilities)
            L = -tf.math.log(tf.math.reduce_sum(vynulovane, axis=1))  # sum over x_i (all but one is 0, so log is on the outside)
            loss = tf.reduce_mean(L)

        variables = [self._W1, self._W2, self._b2, self._b1]

        # Compute the gradient of the loss with respect to variables using
        # backpropagation algorithm via `tape.gradient`
        gradients = tape.gradient(loss, variables)
        return gradients




    def run(self, batch_x, batch_y):
        auto_gradient = self.differentiate_tape(batch_y, batch_x)

        probabilities, hidden_layer_before_tanh, hidden_layer = self.predict_for_train(batch_x)

        # ############## BY  B2 ############################### # todo - best as it can be
        xent_softmax = probabilities - batch_y
        manual_gradient_by_b2 = tf.math.reduce_mean(xent_softmax, axis=0)

        # ############## BY  W2 ############################### # todo - best as it can be
        manual_gradient_by_W2 = tf.einsum("ai,aj->aij", hidden_layer, xent_softmax)
        manual_gradient_by_W2 = tf.math.reduce_mean(manual_gradient_by_W2, axis=0)

        # ############## BY  B1 ###############################  # todo - best as it can be
        xent_softmax_w2 = xent_softmax @ tf.transpose(self._W2)
        tanh = tf.math.reciprocal(tf.math.square(tf.cosh(hidden_layer_before_tanh)))
        xent_softmax_w2_tanh = tf.multiply(xent_softmax_w2, tanh)
        manual_gradient_by_b1 = tf.math.reduce_mean(xent_softmax_w2_tanh, axis=0)

        # ############## BY  W1 ###############################  # todo - best as it can be
        #   `C[a, i, j] = A[a, i] * B[a, j]` USE   `tf.einsum("ai,aj->aij", A, B)`
        manual_gradient_by_W1 = tf.einsum("ai,aj->aij", batch_x, xent_softmax_w2_tanh)
        manual_gradient_by_W1 = tf.math.reduce_mean(manual_gradient_by_W1, axis=0)

        print(auto_gradient[1])
        print("###########")
        print(manual_gradient_by_W2)
        print("###########")
        print("###########")
        print(tf.math.subtract(manual_gradient_by_W2, auto_gradient[1]))

        return manual_gradient_by_W1, manual_gradient_by_W2, manual_gradient_by_b1, manual_gradient_by_b2



if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    model = DifferTest()
    x = tf.Variable([[7., 8., 9.],  [10.,11.,12.]])  #
    y = tf.Variable([[0., 1.], [0., 1.]])  #
    model.run(x, y)

    # vynulovane = tf.math.multiply(onehot_labels, probabilities)  # elementwise
    # L = -tf.math.log(
    #     tf.math.reduce_sum(vynulovane, axis=1))  # sum over x_i (all but one is 0, so log is on the outside)
    # loss = tf.reduce_mean(L)  # avg over batch examples

