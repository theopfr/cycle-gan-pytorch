
import torch
import os
import numpy as np
import shutil
import json
from cycleGanDataset import cycleGanDataset


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def create_dataloader(dataset_path: str, image_size: int=128, batch_size: int=128):
    torch.manual_seed(0)

    dataset = cycleGanDataset(dataset_path, image_size)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=True,
    )
    
    return dataloader


def initialize_run(run_folder: str) -> None:
    if os.path.exists(run_folder):
        shutil.rmtree(run_folder)

    os.mkdir(run_folder)
    os.mkdir(run_folder + "/models")
    os.mkdir(run_folder + "/result_images")

    initial_train_history = {
        "disc_loss": [],
        "gen_loss": [],
        "lr": []
    }

    with open(run_folder + "/train_history.json", "w") as f:
        json.dump(initial_train_history, f, indent=4)


def save_checkpoint(model, optimizer, model_path: str) -> None:
    torch.save({
        "state_dict": model.state_dict(), 
        "optimizer": optimizer.state_dict()
    }, model_path)


def load_checkpoint(model, optimizer, lr: float, model_path: str, device) -> None:
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def save_train_history(run_name: str, disc_loss: float, gen_loss: float, lr: float) -> None:
    run_folder = f"../runs/{run_name}/train_history.json"

    with open(run_folder, "r") as f:
        train_history = json.load(f)
    
    train_history["disc_loss"].append(disc_loss)
    train_history["gen_loss"].append(gen_loss)
    train_history["lr"].append(lr)

    with open(run_folder, "w") as f:
        json.dump(train_history, f, indent=4)


def lr_scheduler(optimizer: torch.optim, current_iteration: int=0, warmup_iterations: int=0, lr_end: float=0.001, decay_rate: float=0.99, decay_intervall: int=100) -> None:
    current_iteration += 1
    current_lr = optimizer.param_groups[0]["lr"]

    if current_iteration <= warmup_iterations:
        for param_group in optimizer.param_groups:
            param_group["lr"] = (current_iteration * lr_end) / warmup_iterations
        # print("lr warm up", optimizer.param_groups[0]["lr"])

    elif current_iteration > warmup_iterations and current_iteration % decay_intervall == 0:
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr * decay_rate
        # print("lr decay:", optimizer.param_groups[0]["lr"])

