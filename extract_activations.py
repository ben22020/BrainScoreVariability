import os
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import ImageNet
from torch.utils.data import DataLoader, Subset
from model import alexnet_v2_pytorch

def main():
    device = torch.device('cuda')

    checkpoint_paths = [
        r".\pytorch_ckpts\training_seed_01.pth",
        r".\pytorch_ckpts\training_seed_02.pth",
        r".\pytorch_ckpts\training_seed_03.pth",
        r".\pytorch_ckpts\training_seed_04.pth",
        r".\pytorch_ckpts\training_seed_05.pth",
        r".\pytorch_ckpts\training_seed_06.pth",
        r".\pytorch_ckpts\training_seed_07.pth",
        r".\pytorch_ckpts\training_seed_08.pth",
        r".\pytorch_ckpts\training_seed_09.pth",
        r".\pytorch_ckpts\training_seed_10.pth",
    ]

    # Directory to save activations
    output_dir = "model_activations"

    # preprocessing pipeline
    transform = transforms.Compose([
        transforms.Resize(256),            # Resize image to 256 pixels on the shorter side
        transforms.CenterCrop(224),        # Center crop to 224x224
        transforms.ToTensor(),             # Convert to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
    ])

    # Load validation dataset
    imagenet_val = ImageNet(root=r".\ILSVRC2012_img_val", split="val", transform=transform)

    # subset of the validation set
    subset_indices = list(range(500))
    val_subset = Subset(imagenet_val, subset_indices)
    val_loader = DataLoader(val_subset, batch_size=32, shuffle=False, num_workers=4)

    # Layers to extract activations from
    layers_to_extract = ['features.0', 'features.3', 'features.6', 'features.8', 'features.10',
                        'classifier.0', 'classifier.3']


    os.makedirs(output_dir, exist_ok=True)

    # Process each checkpoint
    for checkpoint_path in checkpoint_paths:
        # Load the model
        model = alexnet_v2_pytorch(num_classes=1000, dropout_keep_prob=0.5).to(device)
        model.load_state_dict(torch.load(checkpoint_path))
        model.eval()  

        # Extract checkpoint name for output directory
        checkpoint_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
        checkpoint_output_dir = os.path.join(output_dir, checkpoint_name)
        os.makedirs(checkpoint_output_dir, exist_ok=True)

        # Dictionary to store activations
        activations = {layer: [] for layer in layers_to_extract}

        # Hook function to capture activations
        def get_activation(name):
            def hook(model, input, output):
                activations[name].append(output.detach())
            return hook

        # Register hooks
        hooks = []
        for name, layer in model.named_modules():
            if name in layers_to_extract:
                hooks.append(layer.register_forward_hook(get_activation(name)))

        # Process validation subset
        with torch.no_grad():  # Disable gradient computation for efficiency
            for inputs, _ in val_loader:  # Ignore labels
                inputs = inputs.to(device)
                model(inputs)

        # Concatenate activations for each layer
        activations = {layer: torch.cat(activations[layer], dim=0) for layer in activations}

        # Remove hooks to free memory
        for hook in hooks:
            hook.remove()

        # Save activations for each layer
        for layer, activation in activations.items():
            save_path = os.path.join(checkpoint_output_dir, f"{layer.replace('.', '_')}_activations.pt")
            torch.save(activation, save_path)
            print(f"Saved activations for layer {layer} from checkpoint {checkpoint_name} to {save_path}")

    print("Activation extraction complete.")

if __name__ == '__main__':
    main()