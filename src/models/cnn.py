from tensorflow.keras import layers, models, Input

def build_cnn(input_shape, num_classes=1):
    inputs = Input(shape=input_shape, name="image_input")
    x = layers.Conv2D(32, (3,3), activation='relu')(inputs)
    x = layers.MaxPooling2D(2,2)(x)
    x = layers.Conv2D(64, (3,3), activation='relu')(x)
    x = layers.MaxPooling2D(2,2)(x)
    x = layers.Conv2D(128, (3,3), activation='relu')(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(64, activation='relu', name="embedding")(x)
    outputs = layers.Dense(num_classes, activation='sigmoid', name="risk_output")(x)
    model = models.Model(inputs=inputs, outputs=outputs, name="baseline_cnn")
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
