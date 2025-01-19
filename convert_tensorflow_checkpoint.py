import torch
import numpy as np
import tensorflow as tf
from model import alexnet_v2_pytorch  

def load_tf_checkpoint(tf_checkpoint_path, pytorch_model):
    """
    Load a TensorFlow checkpoint and transfer weights to a PyTorch model.
    """
    reader = tf.train.load_checkpoint(tf_checkpoint_path)
    var_dict = reader.get_variable_to_shape_map()

    state_dict = pytorch_model.state_dict()

    for name in var_dict:
        # Exclude non-weight variables
        if "Momentum" in name or "global_step" in name:
            continue

        tf_tensor = reader.get_tensor(name)

        # Handle weights and biases separately
        if "weights" in name and tf_tensor.ndim == 4:  # Conv2D weights
            tf_tensor = np.transpose(tf_tensor, (3, 2, 0, 1))  # [H, W, In, Out] -> [Out, In, H, W]

        # Convert to PyTorch format
        param_name = name.replace("/", ".").replace("_", ".").lower()
        if param_name in state_dict:
            state_dict[param_name] = torch.tensor(tf_tensor, dtype=state_dict[param_name].dtype)

    pytorch_model.load_state_dict(state_dict, strict=False)
    print("Weights loaded successfully.")

if __name__ == "__main__":
    # Define the TensorFlow checkpoint path
    tf_checkpoint_path = r".\AlexNet\training_seed_10\model.ckpt_epoch89"

    # Instantiate the PyTorch model
    model = alexnet_v2_pytorch(num_classes=1000, dropout_keep_prob=0.5)

    # Load TensorFlow weights into the PyTorch model
    load_tf_checkpoint(tf_checkpoint_path, model)

    # Save the converted PyTorch model
    torch.save(model.state_dict(), "training_seed_10.pth")
    print("Model saved.")
