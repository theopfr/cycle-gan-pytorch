
from tqdm import tqdm
import numpy as np
import os
from typing import List, Tuple
import json
import torch
import torch.utils.data
import torch.nn as nn
from torchvision.utils import save_image

from utils import create_dataloader, device, initialize_run, save_checkpoint, load_checkpoint, lr_scheduler, save_train_history
from generator import Generator
from discriminator import Discriminator

torch.manual_seed(0)


def train(config: dict) -> None:
    run_name = config["run_name"]
    dataset_path = f"../datasets/{config['dataset_name']}"
    resume = config["resume"]
    save_image_intervall = config["save_image_intervall"]

    # hyperparameters
    epochs = config["epochs"]
    image_size = config["image_size"]
    batch_size = config["batch_size"]
    num_res_blocks = config["num_res_blocks"]
    lr = config["lr"]
    lr_decay_rate = config["lr_decay_rate"]
    lr_decay_intervall = config["lr_decay_intervall"]
    gaussian_noise_rate = config["gaussian_noise_rate"]
    lambda_adversarial = config["lambda_adversarial"]
    lambda_cycle = config["lambda_cycle"]
    lambda_identity = config["lambda_identity"]

    # path to the run logging folder
    run_folder = f"../runs/{run_name}/"

    # create dataset
    dataset = create_dataloader(dataset_path, image_size=image_size, batch_size=batch_size)

    # models
    generator_a = Generator(num_res_blocks=num_res_blocks).to(device)
    generator_b = Generator(num_res_blocks=num_res_blocks).to(device)
    discriminator_a = Discriminator(gaussian_noise_rate=gaussian_noise_rate).to(device)
    discriminator_b = Discriminator(gaussian_noise_rate=gaussian_noise_rate).to(device)
    
    # optimizers
    optimizer_generator = torch.optim.Adam(list(generator_a.parameters()) + list(generator_b.parameters()), lr=lr, betas=(0.5, 0.999))
    optimizer_discriminator = torch.optim.Adam(list(discriminator_a.parameters()) + list(discriminator_b.parameters()), lr=lr, betas=(0.5, 0.999))

    # losses
    L1_loss = nn.L1Loss()
    MSE_Loss = nn.MSELoss()

    start_epoch = 1

    # resume training-run or initialize new run
    if resume:
        load_checkpoint(generator_a, optimizer_generator, lr, run_folder + "models/generator_model_a.pt", device)
        load_checkpoint(generator_b, optimizer_generator, lr, run_folder + "models/generator_model_b.pt", device,)
        load_checkpoint(discriminator_a, optimizer_discriminator, lr, run_folder + "models/discrimnator_model_a.pt", device)
        load_checkpoint(discriminator_b, optimizer_discriminator, lr, run_folder + "models/discrimnator_model_b.pt", device)
        with open(run_folder + "/train_history.json", "r") as f:
            start_epoch = len(json.load(f)["disc_loss"]) + 1
    else:
        initialize_run(run_folder)

    # train loop
    iteration = 0
    for epoch in range(start_epoch, epochs):
        for idx, (image_a, image_b) in enumerate(tqdm(dataset, desc="epoch")):
            iteration += 1

            lr_scheduler(optimizer_generator, iteration, 0, lr, lr_decay_rate, lr_decay_intervall)
            lr_scheduler(optimizer_discriminator, iteration, 0, lr, lr_decay_rate, lr_decay_intervall)

            image_a = image_a.to(device)
            image_b = image_b.to(device)

            ### DISCRIMINATORS ###

            # generate fake A face
            fake_image_a = generator_a(image_b)

            # train discriminator for face A
            prediction_disc_a_real = discriminator_a(image_a.detach())
            prediction_disc_a_fake = discriminator_a(fake_image_a.detach())

            disc_a_real_loss = MSE_Loss(prediction_disc_a_real, torch.ones_like(prediction_disc_a_real))
            disc_a_fake_loss = MSE_Loss(prediction_disc_a_fake, torch.zeros_like(prediction_disc_a_fake))
            disc_a_loss = disc_a_real_loss + disc_a_fake_loss

            # generate fake B face
            fake_image_b = generator_b(image_a)

            # train discriminator for face A
            prediction_disc_b_real = discriminator_b(image_b.detach())
            prediction_disc_b_fake = discriminator_b(fake_image_b.detach())

            disc_b_real_loss = MSE_Loss(prediction_disc_b_real, torch.ones_like(prediction_disc_b_real))
            disc_b_fake_loss = MSE_Loss(prediction_disc_b_fake, torch.zeros_like(prediction_disc_b_fake))
            disc_b_loss = disc_b_real_loss + disc_b_fake_loss

            disc_loss = (disc_a_loss + disc_b_loss) / 2

            # backpropagate discriminator
            optimizer_discriminator.zero_grad()
            disc_loss.backward()
            optimizer_discriminator.step()

            ### GENERATORS ###

            prediction_disc_a_fake = discriminator_a(fake_image_a)
            prediction_disc_b_fake = discriminator_b(fake_image_b)

            # adversarial loss
            gen_a_loss = MSE_Loss(prediction_disc_a_fake, torch.ones_like(prediction_disc_a_fake))
            gen_b_loss = MSE_Loss(prediction_disc_b_fake, torch.ones_like(prediction_disc_b_fake))

            cycle_image_a = generator_a(fake_image_b)
            cycle_image_b = generator_b(fake_image_a)

            # cycle loss
            cycle_loss_a = L1_loss(image_a, cycle_image_a)
            cycle_loss_b = L1_loss(image_b, cycle_image_b)

            identity_image_a = generator_a(image_a)
            identity_image_b = generator_b(image_b)

            # identity loss
            identity_loss_a = L1_loss(image_a, identity_image_a)
            identity_loss_b = L1_loss(image_b, identity_image_b)

            # total generator loss
            gen_loss = ( 
                gen_a_loss * lambda_adversarial + gen_b_loss * lambda_adversarial +
                cycle_loss_a * lambda_cycle + cycle_loss_b * lambda_cycle +
                identity_loss_a * lambda_identity + identity_loss_b * lambda_identity
            )

            optimizer_generator.zero_grad()
            gen_loss.backward()
            optimizer_generator.step()
            
            # save generated images
            if iteration % save_image_intervall == 0:
                save_image(torch.concat((image_b * 0.5 + 0.5, fake_image_a * 0.5 + 0.5), dim=0), run_folder + f"result_images/{epoch}_fake_image_a.png")
                save_image(torch.concat((image_a * 0.5 + 0.5, fake_image_b * 0.5 + 0.5), dim=0), run_folder + f"result_images/{epoch}_fake_image_b.png")

        # save train history and model checkpoints
        save_train_history(run_name, float(disc_loss.cpu().detach()), float(gen_loss.cpu().detach()), optimizer_generator.param_groups[0]["lr"])

        save_checkpoint(generator_a, optimizer_generator, run_folder + "models/generator_model_a.pt")
        save_checkpoint(generator_b, optimizer_generator, run_folder + "models/generator_model_b.pt")
        save_checkpoint(discriminator_a, optimizer_discriminator, run_folder + "models/discrimnator_model_a.pt")
        save_checkpoint(discriminator_b, optimizer_discriminator, run_folder + "models/discrimnator_model_b.pt")

        print(f"epoch: {epoch + 1} / {epochs}  -  iteration: {iteration}  -  disc_loss: {disc_loss}  -  gen_loss: {gen_loss}\n  -  lr: {optimizer_generator.param_groups[0]['lr']}")


if __name__ == "__main__":
    config = {
        "run_name": "horse-to-zebra",
        "dataset_name": "horse_zebra",
        "save_image_intervall": 50,
        "resume": True,
        
        "epochs": 200,
        "image_size": 256,
        "batch_size": 1,
        "num_res_blocks": 9,
        "lr": 0.0002,
        "lr_decay_rate": 1,
        "lr_decay_intervall": 200,
        "gaussian_noise_rate": 0.05,
        "lambda_adversarial": 1,
        "lambda_cycle": 10,
        "lambda_identity": 10
    }

    train(config)