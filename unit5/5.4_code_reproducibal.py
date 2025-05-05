# Unit 5.2. Training a Multilayer Perceptron in PyTorch & Lightning
# Part 3. Training a Multilayer Perceptron in PyTorch using Lightning

import lightning as L
import torch
from shared_utilities import PyTorchMLP, LightningModel, get_dataset_loaders
from watermark import watermark



if __name__ == "__main__":

    print(watermark(packages="torch,lightning", python=True))
    print("Torch CUDA available?", torch.cuda.is_available())

    torch.manual_seed(123)

    train_loader, val_loader, test_loader = get_dataset_loaders()

    pytorch_model = PyTorchMLP(num_features=784, num_classes=10)
    lightning_model = LightningModel(model=pytorch_model, learning_rate=0.05)

    trainer = L.Trainer(
        max_epochs=10,
        accelerator="auto",  # set to "auto" or "gpu" to use GPUs if available
        devices="auto",  # Uses all available GPUs if applicable
        deterministic=True
    )

    trainer.fit(
        model=lightning_model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    train_acc = trainer.test(dataloaders=train_loader)[0]["accuracy"]
    val_acc = trainer.test(dataloaders=val_loader)[0]["accuracy"]
    test_acc = trainer.test(dataloaders=test_loader)[0]["accuracy"]
    print(
        f"Train Acc {train_acc*100:.2f}%"
        f" | Val Acc {val_acc*100:.2f}%"
        f" | Test Acc {test_acc*100:.2f}%"
    )


PATH = "lightning.pt"
torch.save(pytorch_model.state_dict(), PATH)

# To load model:
# model = PyTorchMLP(num_features=784, num_classes=10)
# model.load_state_dict(torch.load(PATH))
# model.eval()

# Train Acc 97.53% | Val Acc 96.06% | Test Acc 96.68%
# Train Acc 97.53% | Val Acc 96.06% | Test Acc 96.68%