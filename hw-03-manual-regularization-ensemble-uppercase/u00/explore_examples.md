### Assignment: explore_examples

Your goal in this zero-point assignment is to explore the prepared examples.
- The [example_keras_models.py](https://github.com/ufal/npfl114/tree/master/labs/03/example_keras_models.py)
  example demonstrates three different ways of constructing Keras models
  – sequential models, functional API and model subclassing.
  
  - DONE
  
- The [example_keras_manual_batches.py](https://github.com/ufal/npfl114/tree/master/labs/03/example_keras_manual_batches.py)
  shows how to train and evaluate Keras model when using custom batches.
  
    - DONE
    
    
- The [example_manual.py](https://github.com/ufal/npfl114/tree/master/labs/03/example_manual.py)
  illustrates how to implement a manual training loop without using
  `Model.compile`, with custom `Optimizer`, loss function and metric.
  However, this example is 2-3 times slower than the previous two ones.
  
    - DONE

- The [example_manual_tf_function.py](https://github.com/ufal/npfl114/tree/master/labs/03/example_manual_tf_function.py)
  uses `tf.function` annotation to speed up execution of the previous
  example back to the level of `Model.fit`. See the
  [official `tf.function` documentation](https://www.tensorflow.org/api_docs/python/tf/function)
  for details.

    - DONE
    
    - it compiles a function into a callable TensorFlow graph.
    - tf.function constructs a callable that executes a
        -  TensorFlow graph (tf.Graph)  created by trace-compiling the TensorFlow operations in func, 
        - effectively executing func as a TensorFlow graph.
        
- func may use data-dependent control flow, including if, for, while break, continue and return statements:
- func's closure may include tf.Tensor and tf.Variable objects:
- func may also use ops with side effects, such as tf.print, tf.Variable and others:


- Caution: Passing python scalars or lists as arguments to tf.function will always build a new graph. To avoid this, pass numeric arguments as Tensors whenever possible:

