from . import discriminator as discriminator
from . import generator as generator
import numpy as np
import os.path

from keras.optimizers import Adam
from keras.layers import Input
from keras.models import Model, load_model


class ModelGAN:
    def __init__(self, data):
        self.data = data
        self.generator = generator.build_generator()
        self.discriminator = discriminator.build_discriminator()

    def compile_models(self, disc_lr = 0.00002, disc_beta_1 = 0.5, disc_beta_2 = 0.999, 
                      gener_lr = 0.0002, gener_beta_1 = 0.5, gener_beta_2 = 0.999,
                      gan_losses = [0.999, 0.001]):
        d_optim = Adam(lr=disc_lr, beta_1=disc_beta_1, beta_2=disc_beta_2)
        g_optim = Adam(lr=gener_lr, beta_1=gener_beta_1, beta_2=gener_beta_2)

        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=d_optim,
                                   metrics=['accuracy'])

        masked_img = Input(shape=(32,32,3))
        gen_missing = self.generator(masked_img)
        self.discriminator.trainable = False

        valid = self.discriminator(gen_missing)

        self.gan = Model(masked_img , [gen_missing, valid])
        self.gan.compile(loss=['mse', 'binary_crossentropy'],
                         loss_weights=gan_losses,
                         optimizer=g_optim)

    def _crop_center(self, img):
        full_img = np.copy(img)
        y,x, _ = img.shape
        cropx = x // 2
        cropy = y // 2
        startx = x//2-(cropx//2)
        starty = y//2-(cropy//2)    
        full_img[starty:starty+cropy,startx:startx+cropx] = 255
        cropped_img = img[starty:starty+cropy,startx:startx+cropx]
        return (full_img, cropped_img)

    def _get_batch(self, batch_size):
            idx = np.random.choice(np.arange(self.data.shape[0]), batch_size, replace=False)
            return self.data[idx]

    def _prepare_imgs(self, imgs):
        cropped_images = [self._crop_center(x) for x in imgs]
        X = np.array([self._normalize_img(x[0]) for x in cropped_images])
        y = np.array([self._normalize_img(x[1]) for x in cropped_images])
        return (X, y)

    def _normalize_img(self, img):
        return (img - 127.5)/127.5
    
    def train(self, train_steps=5000, batch_size=64):
        for i in range(train_steps):
            
            # ---------------------
            #  Train Discriminator
            # ---------------------
            
            # Random input
            batch_imgs = self._get_batch(batch_size)
            X, y = self._prepare_imgs(batch_imgs)
            missing_part = y
            masked_imgs = X
            gen_missing_part = self.generator.predict(masked_imgs)
            
            # Valid and fake outputs
            valid = np.ones([batch_size, 1])
            fake = np.zeros([batch_size, 1])

                
            # Train discriminator
            d_loss_valid = self.discriminator.train_on_batch(missing_part, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_missing_part, fake)
            d_loss = 0.5 * np.add(d_loss_valid, d_loss_fake)
            
            # ---------------------
            #  Train Generator
            # ---------------------
            g_loss = self.gan.train_on_batch(masked_imgs, [missing_part, valid])
            
            # ---------------------
            #  Logs
            # ---------------------
            if i % 1000 == 0:
                self._save_model()
            if i % 100 == 0:
                print ("%d [D loss: %f, acc: %.2f%%] [G loss: %f, mse: %f]" % (i, d_loss[0], 100*d_loss[1], g_loss[0], g_loss[1]))

    def _save_model(self):
        self.generator.save('generator.hf5', True)
        self.discriminator.save('discriminator.hf5', True)

    def load_model(self):
        if os.path.isfile("generator.hf5"):
            self.generator = load_model("generator.hf5")
        if os.path.isfile("discriminator.hf5"):
            self.discriminator = load_model("discriminator.hf5")