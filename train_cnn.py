from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense,Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

base = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(128,128,3)
)

for layer in base.layers:
    layer.trainable=False

x = base.output
x = Flatten()(x)
x = Dense(128,activation='relu')(x)
output = Dense(1,activation='sigmoid')(x)

model = Model(inputs=base.input,outputs=output)

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train = datagen.flow_from_directory(
    "army_dataset",
    target_size=(128,128),
    batch_size=16,
    subset="training",
    class_mode="binary"
)

test = datagen.flow_from_directory(
    "army_dataset",
    target_size=(128,128),
    batch_size=16,
    subset="validation",
    class_mode="binary"
)

model.fit(train,epochs=10,validation_data=test)

model.save("models/cnn_model.h5")
