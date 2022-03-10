
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import List, Tuple

import torch
import torch.utils.data
import torch.nn as nn
from torchvision.utils import save_image

from utils import create_dataloader, device, initialize_run, save_checkpoint, load_checkpoint, lr_scheduler
from generator import Generator
from discriminator import Discriminator

torch.manual_seed(0)


class TrainSetup:
    def __init__(self, run_name: str, dataset_path: str, image_size: int, epochs: int, batch_size: int, num_res_blocks: int, lr: float, gaussian_noise_rate: float, lambda_adversarial: int, lambda_cycle: float, resume: bool=False) -> None:
        self.run_name = run_name
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_res_blocks = num_res_blocks
        self.lr = lr
        self.gaussian_noise_rate = gaussian_noise_rate
        self.lambda_adversarial = lambda_adversarial
        self.lambda_cycle = lambda_cycle

        self.generator_a = Generator(num_res_blocks=1).to(device)
        self.generator_b = Generator(num_res_blocks=1).to(device)
        self.discriminator_a = Discriminator(gaussian_noise_rate=self.gaussian_noise_rate).to(device)
        self.discriminator_b = Discriminator(gaussian_noise_rate=self.gaussian_noise_rate).to(device)

        self.optimizer_generator = torch.optim.Adam(list(self.generator_a.parameters()) + list(self.generator_b.parameters()), lr=self.lr, betas=(0.5, 0.999))
        # self.optimizer_discriminator = torch.optim.Adam(list(self.discriminator_a.parameters()) + list(self.discriminator_b.parameters()), lr=self.lr, betas=(0.5, 0.999))
        self.optimizer_discriminator_a = torch.optim.Adam(self.discriminator_a.parameters(), lr=self.lr, betas=(0.5, 0.999))
        self.optimizer_discriminator_b = torch.optim.Adam(self.discriminator_b.parameters(), lr=self.lr, betas=(0.5, 0.999))

        self.L1_loss = nn.L1Loss()
        self.MSE_Loss = nn.MSELoss()

        self.run_folder = f"../runs/{run_name}/"

        if resume:
            load_checkpoint(self.generator_a, self.optimizer_generator, self.lr, self.run_folder + "models/generator_model_a.pt", device)
            load_checkpoint(self.generator_b, self.optimizer_generator, self.lr, self.run_folder + "models/generator_model_b.pt", device,)
            load_checkpoint(self.discriminator_a, self.optimizer_discriminator_a, self.lr, self.run_folder + "models/discrimnator_model_a.pt", device)
            load_checkpoint(self.discriminator_b, self.optimizer_discriminator_b, self.lr, self.run_folder + "models/discrimnator_model_b.pt", device)
        else:
            initialize_run(run_path=self.run_folder)

        self.dataset = create_dataloader(dataset_path, image_size=self.image_size, batch_size=batch_size)

    def train(self) -> None:
        iteration = 0
        for epoch in range(self.epochs):
            for idx, (image_a, image_b) in enumerate(tqdm(self.dataset, desc="epoch")):
                iteration += 1

                lr_scheduler(self.optimizer_generator, iteration, 0, self.lr, 0.97, 175)
                lr_scheduler(self.optimizer_discriminator_a, iteration, 0, self.lr, 0.97, 175)
                lr_scheduler(self.optimizer_discriminator_b, iteration, 0, self.lr, 0.97, 175)

                image_a = image_a.to(device)
                image_b = image_b.to(device)

                # generate fake A face
                fake_image_a = self.generator_a(image_b)

                # train discriminator for face A
                prediction_disc_a_real = self.discriminator_a(image_a.detach())
                prediction_disc_a_fake = self.discriminator_a(fake_image_a.detach())

                disc_a_real_loss = self.MSE_Loss(prediction_disc_a_real, torch.ones_like(prediction_disc_a_real))
                disc_a_fake_loss = self.MSE_Loss(prediction_disc_a_fake, torch.zeros_like(prediction_disc_a_fake))
                disc_a_loss = disc_a_real_loss + disc_a_fake_loss

                self.optimizer_discriminator_a.zero_grad()
                disc_a_loss.backward()
                self.optimizer_discriminator_a.step()

                # generate fake B face
                fake_image_b = self.generator_b(image_a)

                # train discriminator for face A
                prediction_disc_b_real = self.discriminator_b(image_b.detach())
                prediction_disc_b_fake = self.discriminator_b(fake_image_b.detach())

                disc_b_real_loss = self.MSE_Loss(prediction_disc_b_real, torch.ones_like(prediction_disc_b_real))
                disc_b_fake_loss = self.MSE_Loss(prediction_disc_b_fake, torch.zeros_like(prediction_disc_b_fake))
                disc_b_loss = disc_b_real_loss + disc_b_fake_loss

                #disc_loss = (disc_a_loss + disc_b_loss) / 2

                # backpropagate discriminator
                self.optimizer_discriminator_b.zero_grad()
                disc_b_loss.backward()
                self.optimizer_discriminator_b.step()

                ### generators ###

                prediction_disc_a_fake = self.discriminator_a(fake_image_a.detach())
                prediction_disc_b_fake = self.discriminator_b(fake_image_b.detach())

                # adversarial loss
                gen_a_loss = self.MSE_Loss(prediction_disc_a_fake, torch.ones_like(prediction_disc_a_fake))
                gen_b_loss = self.MSE_Loss(prediction_disc_b_fake, torch.ones_like(prediction_disc_b_fake))

                # cycle loss
                cycle_image_a = self.generator_a(fake_image_b)
                cycle_image_b = self.generator_b(fake_image_a)

                cycle_loss_a = self.L1_loss(image_a, cycle_image_a)
                cycle_loss_b = self.L1_loss(image_b, cycle_image_b)

                identity_image_a = self.generator_a(image_a)
                identity_image_b = self.generator_b(image_b)

                identity_loss_a = self.L1_loss(image_a, identity_image_a)
                identity_loss_b = self.L1_loss(image_b, identity_image_b)

                gen_loss = ( 
                    (gen_a_loss * self.lambda_adversarial) + (gen_b_loss * self.lambda_adversarial) + 
                    (cycle_loss_a * self.lambda_cycle) + (cycle_loss_b * self.lambda_cycle) +
                    identity_loss_a + identity_loss_b
                )

                self.optimizer_generator.zero_grad()
                gen_loss.backward()
                self.optimizer_generator.step()
                
                if iteration % 100 == 0:
                    save_image(torch.concat((image_b * 0.5 + 0.5, fake_image_a * 0.5 + 0.5), dim=0), self.run_folder + f"result_images/{epoch}_fake_image_a.png")
                    save_image(torch.concat((image_a * 0.5 + 0.5, fake_image_b * 0.5 + 0.5), dim=0), self.run_folder + f"result_images/{epoch}_fake_image_b.png")
                    
            print(f"epoch: {epoch + 1} / {self.epochs}  -  iteration: {iteration}  -  disc_loss: {(disc_a_loss + disc_b_loss) / 2}  -  gen_loss: {gen_loss}\n  -  lr: {self.optimizer_generator.param_groups[0]['lr']}")

            save_checkpoint(self.generator_a, self.optimizer_generator, self.run_folder + "models/generator_model_a.pt")
            save_checkpoint(self.generator_b, self.optimizer_generator, self.run_folder + "models/generator_model_b.pt")
            save_checkpoint(self.discriminator_a, self.optimizer_discriminator_a, self.run_folder + "models/discrimnator_model_a.pt")
            save_checkpoint(self.discriminator_b, self.optimizer_discriminator_b, self.run_folder + "models/discrimnator_model_b.pt")

            
trainSetup = TrainSetup(
    run_name="horse-to-zebra-1",
    dataset_path="../datasets/horse_zebra/",
    image_size=200,
    epochs=200,
    batch_size=1,
    num_res_blocks=7,
    lr=6.255007261813624e-05,
    gaussian_noise_rate=0.075,
    lambda_adversarial=1,
    lambda_cycle=10,
    resume=True
)

trainSetup.train()