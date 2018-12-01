import utils as utils
import numpy as np

def train(data, model_generator, model_discriminator, model_gan,
          train_steps=5000, batch_size=64):
    for i in range(train_steps):
        
        # ---------------------
        #  Train Discriminator
        # ---------------------
        
        # Random input
        batch_imgs = utils.get_batch(data, batch_size)
        X, y = utils.prepare_imgs(batch_imgs)
        missing_part = y
        masked_imgs = X
        gen_missing_part = model_generator.predict(masked_imgs)
        
        # Valid and fake outputs
        valid = np.ones([batch_size, 1])
        fake = np.zeros([batch_size, 1])

            
        # Train discriminator
        d_loss_valid = model_discriminator.train_on_batch(missing_part, valid)
        d_loss_fake = model_discriminator.train_on_batch(gen_missing_part, fake)
        d_loss = 0.5 * np.add(d_loss_valid, d_loss_fake)
        
        # ---------------------
        #  Train Generator
        # ---------------------
        g_loss = model_gan.train_on_batch(masked_imgs, [missing_part, valid])
        
        # ---------------------
        #  Logs
        # ---------------------
        if i % 100 == 0:
            print ("%d [D loss: %f, acc: %.2f%%] [G loss: %f, mse: %f]" % (i, d_loss[0], 100*d_loss[1], g_loss[0], g_loss[1]))
