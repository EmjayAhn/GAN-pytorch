import torch

def train_model(discriminator, generator, batch_size,
                dis_optimizer, gen_optimizer,
                criterion, dataloader, epochs, device):
    for epoch in range(epochs):
        for batch_index, (batch_images, _) in enumerate(dataloader):
            # Label 생성
            reals = torch.ones(batch_size, 1).to(device)
            fakes = torch.zeros(batch_size, 1).to(device)

            batch_images = batch_images.view(batch_size, -1).to(device)
            
            #### DISCRIMINATOR
            # Real Image 에 대한 loss
            dis_results = discriminator(images)
            dis_real_loss = criterion(dis_results, reals)
            
            # Fake Image 에 대한 loss
            latent = torch.randn(batch_size, 64).to(device)
            fake_images = generator(latent).detach()
            dis_results = discriminator(fake_images)
            dis_fake_loss = criterion(dis_results, fakes)
            
            # total loss
            dis_total_loss = dis_real_loss + dis_fake_loss
            
            discriminator.zero_grad()
            dis_total_loss.backward()
            dis_optimizer.step()
            

            #### GENERATOR
            fake_images = generator(latent).to(device)
            dis_results = discriminator(fake_images)
            
            # Fake Image 에 대한 generator loss
            gen_loss = criterion(dis_results, fakes)

            generator.zero_grad()
            gen_loss.backward()
            gen_optimizer.step()
            
            if batch_index % 300 == 0:
                print("EPOCH {}: BATCH: {}, discrim_loss: {}, generator_loss: {}".format(epoch, batch_index, dis_total_loss, gen_loss))
            
    return discriminator, generator





            


