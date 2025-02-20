import os
# Désactiver les erreurs liées à Tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Désactive les logs d'information et d'avertissement de TensorFlow

import tensorflow as tf
from tensorflow.keras import layers # type: ignore
import numpy as np
import matplotlib.pyplot as plt
import time
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tqdm import tqdm  # Ajout pour la barre de progression

# Chemin vers le dataset TACO
dataset_path = "data"

# Utilisation de ImageDataGenerator pour charger et prétraiter les images
datagen = ImageDataGenerator(rescale=1./255)

# Chargement des images
batch_size = 32
img_height = 64
img_width = 64

train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(img_height, img_width), # Redimensionnement à 64x64
    batch_size=batch_size,
    class_mode=None,
    color_mode='rgb'
)


# Générateur
def make_generator_model():
    model = tf.keras.Sequential()
    
    # Couche dense pour transformer le vecteur de bruit en un volume plus grand
    model.add(layers.Dense(8*8*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    # Reshape pour obtenir un volume 3D (8x8x256)
    model.add(layers.Reshape((8, 8, 256)))

    # Couches de convolution transposée pour upsampler l'image
    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    # Ajout d'une nouvelle couche pour agrandir à 64x64x3
    model.add(layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    # Couche finale (sortie)
    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    
    return model


# Discriminateur
def make_discriminator_model():
    model = tf.keras.Sequential()
    
    # Couches de convolution pour extraire les caractéristiques de l'image
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[64, 64, 3]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model


# Fonctions de pertes et d'optimisation
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)


# Boucle d'entrainement
@tf.function
def train_step(images):
    noise = tf.random.normal([batch_size, 100])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()

        # Barre de progression pour l'époque
        progress_bar = tqdm(dataset, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch")

        for image_batch in progress_bar:
            train_step(image_batch)

        # Mise à jour de la barre de progression avec les pertes
        progress_bar.set_postfix({
            "Gen Loss": generator_loss(generator(tf.random.normal([batch_size, 100]), training=False)).numpy(),
            "Disc Loss": discriminator_loss(discriminator(image_batch, training=False), 
                                            discriminator(generator(tf.random.normal([batch_size, 100]), training=False))).numpy()
        })

        # Afficher les images générées à la fin de chaque époque
        generate_and_save_images(generator, epoch + 1, tf.random.normal([16, 100]))

        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))

    # Sauvegarder les modèles après l'entraînement
    generator.save('generator_model.h5')
    discriminator.save('discriminator_model.h5')
    print("Modèles sauvegardés avec succès.")


def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)
    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, :] * 0.5 + 0.5)  # Rescale de [-1, 1] à [0, 1]
        plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()


def test_generator(model_path, num_images=16):
    # Charger le modèle générateur
    generator = tf.keras.models.load_model(model_path)

    # Générer des images à partir de bruit aléatoire
    noise = tf.random.normal([num_images, 100])
    generated_images = generator(noise, training=False)

    # Afficher les images générées
    plt.figure(figsize=(4, 4))
    for i in range(generated_images.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(generated_images[i, :, :, :] * 0.5 + 0.5)  # Rescale de [-1, 1] à [0, 1]
        plt.axis('off')
    plt.show()


if __name__ == "__main__":
    # Créfinir les modèles
    generator = make_generator_model()
    discriminator = make_discriminator_model()

    # Entraîner le GAN
    EPOCHS = 50
    train(train_generator, EPOCHS)

    # Tester le générateur après l'entraînement
    test_generator('generator_model.h5', num_images=16)