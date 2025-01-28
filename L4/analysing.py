import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
import os


def create_model():
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomBrightness(0.1),
        tf.keras.layers.RandomContrast(0.1),
    ])
    # Create base ResNet50 model
    base_model = tf.keras.applications.ResNet50(
        input_shape=(32, 32, 3),
        include_top=False,
        weights=None,
    )
    base_model = tf.keras.Model(
        base_model.inputs, outputs=[base_model.get_layer("conv2_block3_out").output]
    )

    # Create the full model
    inputs = tf.keras.Input(shape=(32, 32, 3))
    x = data_augmentation(inputs)
    x = tf.keras.applications.resnet.preprocess_input(inputs)
    x = base_model(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x[0])
    x = tf.keras.layers.Dense(10)(x)
    model = tf.keras.Model(inputs, x)

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    return model


def load_data_from_directory(directory, class_names):
    images = []
    labels = []

    for idx, class_name in enumerate(class_names):
        class_path = os.path.join(directory, class_name)
        if os.path.exists(class_path):
            for image_name in os.listdir(class_path):
                image_path = os.path.join(class_path, image_name)
                image = tf.keras.preprocessing.image.load_img(
                    image_path,
                    color_mode='rgb',
                    target_size=(32, 32)  # Changed to 32x32
                )
                image = tf.keras.preprocessing.image.img_to_array(image)
                images.append(image)
                labels.append(idx)

    return np.array(images), np.array(labels)


def analyze_model_performance(weights_path, test_directory, class_names):
    # Create and load model
    model = create_model()
    model.load_weights(weights_path)

    # Load test data
    test_images, test_labels = load_data_from_directory(test_directory, class_names)

    # Convert labels to categorical
    test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=10)

    # Get predictions
    predictions = model.predict(test_images)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(test_labels, axis=1)

    # 1. Confusion Matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

    # 2. Classification Report
    print("\nClassification Report:")
    print(classification_report(true_classes, predicted_classes, target_names=class_names))

    # 3. Per-class accuracy
    per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
    print("\nPer-class accuracy:")
    for class_name, accuracy in zip(class_names, per_class_accuracy):
        print(f"{class_name}: {accuracy:.3f}")

    # 4. Show examples of misclassifications
    def plot_misclassified_examples(num_examples=20):
        misclassified_indices = np.where(predicted_classes != true_classes)[0]

        if len(misclassified_indices) > 0:
            num_examples = min(num_examples, len(misclassified_indices))
            fig, axes = plt.subplots(1, num_examples, figsize=(15, 3))

            if num_examples == 1:
                axes = [axes]

            for i in range(num_examples):
                idx = misclassified_indices[i]
                axes[i].imshow(test_images[idx])
                axes[i].set_title(
                    f'True: {class_names[true_classes[idx]]}\nPred: {class_names[predicted_classes[idx]]}')
                axes[i].axis('off')

            plt.tight_layout()
            plt.show()

    plot_misclassified_examples()



# Paths
model_path = r"C:\Users\Simion\Desktop\Deep-dive\L4\best_model_93_accuracy.weights.h5"
test_directory = r"C:\Users\Simion\Desktop\Deep-dive\L4\data\test"
class_names = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X']

# Run the analysis
analyze_model_performance(model_path, test_directory, class_names)