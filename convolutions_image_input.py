from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2

"""
ImageDataGenerator with flow_from_dataframe


testing = pd.DataFrame({'image_id':['Train_0.jpg', 'Train_1.jpg', 'Train_2.jpg',
                                    'Train_3.jpg','Train_4.jpg','Train_5.jpg',
                                    'Test_0.jpg', 'Test_1.jpg'],
                        'label_names':['first', 'two', 'first',
                                       'two', 'first', 'two',
                                       'first', 'two']}
                        )
"""
def blur_preprocessing(img):
    return cv2.blur(img, (5, 5))

img_data_generator = ImageDataGenerator(rescale=1/255, 
                                        validation_split=0.3,
                                        rotation_range = 180,
                                        horizontal_flip = True,
                                        vertical_flip = True,
                                        preprocessing_function=blur_preprocessing
                                       )


## Recreate datasets from dataframe
train_data_multi = img_data_generator.flow_from_dataframe(dataframe=testing,
                                                    directory="plant-pathology-2020-fgvc7/Testimages/",
                                                    x_col="image_id",
                                                    y_col= "label_names",
                                                    target_size=(256, 256),
                                                    class_mode='categorical',
                                                    batch_size=32,
                                                    subset='training',
                                                    shuffle=True,
                                                    seed=42)

val_data_multi = img_data_generator.flow_from_dataframe(dataframe=testing,
                                                    directory="plant-pathology-2020-fgvc7/Testimages/",
                                                    x_col="image_id",
                                                    y_col= "label_names",
                                                    target_size=(256, 256),
                                                    class_mode='categorical',
                                                    batch_size=32,
                                                    subset='validation',
                                                    shuffle=True,
                                                    seed=42)

"""
ImageDataGenerator with flow_from_directory
"""

train_generator = img_data_generator.flow_from_directory(
        "Training_Directory",  # This is the source directory for training images
        target_size=(150, 150),  # All images will be resized to 150x150
        batch_size=20,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

# Flow validation images in batches of 20 using test_datagen generator
validation_generator = img_data_generator.flow_from_directory(
        "Validation_Directory",
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')




"""
--------------------------------------------------------------------------------------------------------------
"""



"""
Image dataset from directory
"""

new_base_dir = ""

train_dataset = image_dataset_from_directory(
    new_base_dir / "train",
    image_size = (180, 180),
    batch_size=  32
)

train_dataset = image_dataset_from_directory(
    new_base_dir / "validation",
    image_size = (180, 180),
    batch_size=  32
)

train_dataset = image_dataset_from_directory(
    new_base_dir / "test",
    image_size = (180, 180),
    batch_size=  32
)
