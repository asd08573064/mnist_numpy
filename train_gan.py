from util.util import accuracy
from data.dataset import get_mnist, dataset
from util.layers import Linear, Batch_Norm, Softmax, Sigmoid, ReLU, Cross_Entropy, LeakyRelu, BinaryCrossEntropy, Tanh
from util.layers import Cross_Entropy
from model import NN
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def sample_one_image(generator, latent_dim, epoch, sample_id=None):
    noise = np.random.normal(0, 1, (latent_dim, 1))
    img = generator.forward(noise).reshape((1 ,28, 28))
    im = Image.fromarray((img[0] * 255).astype(np.uint8))
    if sample_id != None:
        im.save("generated_samples/ep_{}_sample.jpeg".format(epoch))
            
def sample(generator, latent_dim, epoch, sample_num=64):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
    noise = np.random.normal(0, 1, (latent_dim, sample_num))
    img = generator.forward(noise).T.reshape((sample_num ,28, 28))
    for i in range(img.shape[0]):
        plt.subplot(8, 8, i+1)
        plt.imshow(img[i] * 255, cmap='gray')
        plt.axis('off')
    
    plt.savefig('./generated_samples/image_at_epoch_{:04d}.png'.format(epoch))

def main():
    epochs = 300
    log_step = 100
    batch_size = 64
    latent_dim = 128
    g_step_per_epoch = 1
    x_train, x_val, y_train, y_val = get_mnist()
    generator = NN(input_size = latent_dim, output_size = 784, learning_rate=0.0001)
    discriminator = NN(input_size = 784, output_size = 1, learning_rate=0.0001)
    
    generator.module_list = [
        Linear(latent_dim, 1024),
        ReLU(),
        Linear(1024, 1024),
        ReLU(),
        Linear(1024, 28*28),
        Sigmoid()
    ]
    
    discriminator.module_list = [
        Linear(28*28, 256),
        LeakyRelu(),
        Linear(256, 128),
        LeakyRelu(),
        Linear(128, 1),
        Sigmoid()
    ]
    
    train_dataset = dataset(x_train, y_train, batch_size=batch_size)
    test_dataset = dataset(x_val, y_val, batch_size=batch_size)
    
    valid = np.tile(np.array([1.0]), (batch_size, 1))
    valid_hard = np.tile(np.array([1.0]), (batch_size, 1))
    fake = np.tile(np.array([0.0]), (batch_size, 1))
    fake_hard = np.tile(np.array([0.0]), (batch_size, 1))
    
    for epoch in range(epochs):
        g_loss_ = []
        d_loss_ = []
        ce_loss = BinaryCrossEntropy()
        for step, (image, label) in enumerate(train_dataset):
            img, y = image.T, label.T
            
            # ---------------------
            #  Train Discriminator
            # ---------------------
            d_label = np.concatenate([fake_hard, valid_hard])
            noise = np.random.normal(0, 1, (latent_dim, batch_size))

            gen_img = generator.forward(noise)
            output = discriminator.forward(np.concatenate([gen_img, img], axis=1))
            d_loss = ce_loss.forward(output, d_label.T)
            
            g_d_loss = ce_loss.backward(d_label.T)
            discriminator.backward(g_d_loss)
            discriminator.update()
            
            # ---------------------
            #  Train Generator
            # ---------------------
            for _ in range(g_step_per_epoch):
                noise = np.random.normal(0, 1, (latent_dim, batch_size))
                gen_img = generator.forward(noise)
                output = discriminator.forward(gen_img)
                g_loss = ce_loss.forward(output, valid.T)
                g_g_loss = ce_loss.backward(valid.T)
                g_discriminator = discriminator.backward(g_g_loss)
                generator.backward(g_discriminator)
                generator.update()
            
            g_loss_.append(g_loss) 
            d_loss_.append(d_loss) 
            
            if step % log_step == 0:
                print('[epoch {}/{}] step: {}/{}, g_loss: {}, d_loss: {}'.format(epoch, epochs, step, len(x_train)//batch_size, sum(g_loss_)/len(g_loss_), sum(d_loss_)/len(d_loss_)))
                test_acc = accuracy(discriminator, test_dataset)
                train_acc = accuracy(discriminator, train_dataset)
                print('train acc: {}, test acc: {}'.format(train_acc, test_acc))
        
        sample(generator, latent_dim, epoch)

        g_epoch_loss = sum(g_loss_)/len(g_loss_)
        d_epoch_loss = sum(d_loss_)/len(d_loss_)
        print('g_loss: {}, d_loss: {}'.format(g_epoch_loss, d_epoch_loss))
               
if __name__ == '__main__':
    main()