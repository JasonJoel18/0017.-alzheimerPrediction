import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten

# Define input shape
img_shape = (224, 224, 3)

# Load pre-trained Xception model
base_model = tf.keras.applications.Xception(include_top= False, weights= "imagenet",
                            input_shape= img_shape, pooling= 'max')

# for layer in base_model.layers:
#     layer.trainable = False
    
model = Sequential([
    tf.keras.layers.InputLayer(input_shape=img_shape),
    base_model,
    Flatten(),
    Dropout(rate= 0.3),
    Dense(128, activation= 'relu'),
    Dropout(rate= 0.25),
    Dense(4, activation= 'softmax')
])

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adamax(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Print the model summary
model.summary()

# model.build(input_shape=(None, *img_shape))
# tf.keras.utils.plot_model(model, show_shapes=True)

for images, labels in train_dataset.take(1):
    print("Image batch shape:", images.shape)
    print("Label batch shape:", labels.shape)
    
    
    
history = model.fit(
    train_dataset,
    epochs=10,
    validation_data=val_dataset,
    validation_freq=1
)