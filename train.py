import torch
import matplotlib.pyplot as plt

def train_model(latent_dims, discriminator, generator, batch_size,
                dis_optimizer, gen_optimizer,
                criterion, dataloader, epochs, device):
    
    testing_random_latent = torch.randn(5, latent_dims).to(device)
    
    for epoch in range(epochs):
        for batch_index, (batch_images, _) in enumerate(dataloader):
            # Label 생성
            reals = torch.ones(batch_size, 1).to(device)
            fakes = torch.zeros(batch_size, 1).to(device)

            batch_images = batch_images.view(batch_size, -1).to(device)
            
            #### DISCRIMINATOR
            # Real Image 에 대한 loss
            dis_results = discriminator(batch_images)
            dis_real_loss = criterion(dis_results, reals)
            
            # Fake Image 에 대한 loss
            latent = torch.randn(batch_size, latent_dims).to(device)
            fake_images = generator(latent).detach()
            dis_results = discriminator(fake_images)
            dis_fake_loss = criterion(dis_results, fakes)
            
            # total loss
            dis_total_loss = dis_real_loss + dis_fake_loss

            discriminator.zero_grad()
            dis_total_loss.backward()
            dis_optimizer.step()
            

            #### GENERATOR
            latent = torch.randn(batch_size, latent_dims).to(device)
            fake_images = generator(latent).to(device)
            dis_results = discriminator(fake_images)
            
            # Fake Image 에 대한 generator loss
            # 여기가 매우 중요, 우리가 generator 를 train 할 때는, log(D(G(z)))를 최대화 해야한다.
            gen_loss = criterion(dis_results, reals)
            
            discriminator.zero_grad()
            generator.zero_grad()
            gen_loss.backward()
            gen_optimizer.step()
            

        if epoch % 20 == 0:
            print("EPOCH {}: BATCH: {}, discrim_loss: {}, generator_loss: {}".format(epoch, batch_index, dis_total_loss, gen_loss))
            with torch.no_grad():
                testing_fake_images = generator(testing_random_latent)
                testing_fake_images = testing_fake_images.reshape(5, 28, 28).cpu().numpy()
                
                plt.figure(figsize=(10, 5))
                plt.title("GENERATED IMAGE, EPOCH {}".format(epoch))
                for i in range(5):
                    plt.subplot('15{}'.format(i+1))
                    plt.imshow(testing_fake_images[i], cmap='gray')
                plt.show()
            
    return discriminator, generator





            


