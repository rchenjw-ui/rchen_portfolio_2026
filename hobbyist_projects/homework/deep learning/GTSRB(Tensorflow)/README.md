

- With initial hyperparameters (10 epochs and default learning rate):
tf.keras.layers.Conv2D(30, (3,3), activation = "relu", input_shape = (30,30,3)),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(32, (3,3), activation = "relu", input_shape = (14,14,32)),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation = "relu", input_shape = (1152)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(64, activation = "relu", input_shape = (1152)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(NUM_CATEGORIES, activation = "softmax")


Results: 333/333 - 2s - 7ms/step - accuracy: 0.7882 - loss: 0.6581



- Final stack performance: 40/40 - 0s - 9ms/step - accuracy: 0.9960 - loss: 0.0254

100 Epochs, LR = 0.0001

([
        tf.keras.layers.Conv2D(32, (3,3), activation = "relu", input_shape = (30,30,3)),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(32, (3,3), activation = "relu", input_shape = (14,14,32)),
        tf.keras.layers.MaxPooling2D((2,2)), #(6,6,32 size),
        tf.keras.layers.Conv2D(32, (2,2), activation = "relu", input_shape = (6,6,32)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation = "relu", input_shape = (800)), #use 1152 for full,
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128, activation = "relu", input_shape = (256)),
        tf.keras.layers.Dense(64, activation = "relu", input_shape = (128)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(NUM_CATEGORIES, activation = "softmax")
    ])


What I learned from this.

The first thing I learned was how to utilize the TensorFlow API. After learning that, I now know how to assemble a model and train it. In many respects it is much easier than PyTorch. In addition, I also learned how to utilize the OpenCV library to open, scale and filter images.

The more important thing I learned, however, was hyperparameter tuning and various relevant architectural features. I could've learned the previous from documentation, but the latter is more relevant and cross-platform.
For one, I learned a critical tradeoff between learning rate and convergence. A higher learning rate allows much faster convergence on less data but also limits the accuracy achievable, while a lower learning rate gets much better final accuracy but requires much more data. Although I don't know exactly how, a good future strategy would be to initially use a very high learning rate (as high as 0.1) for the first few epochs, then rapidly drop the learning rate.
In addition, I learned about the effects of stacking layers. Adding too many convolutional layers can reduce the size of an image too far (effectively "bottlenecking" out information) but also extracts fairly salient features. Max pooling further "bottlenecks" an image. On the other hand, performance generally increases with more dense layers, at the risk of overfitting, although dropout can help.
Finally, I learned about the importance of testing on a dataset with similar scale. What I did with a smaller and less complex dataset (high learning rate and less layers) seemed to get pretty good accuracy, but on a larger dataset, it flopped. In the end, I feel that for testing models at subscale, a dataset with similar numbers of labels, but smaller size, is needed
