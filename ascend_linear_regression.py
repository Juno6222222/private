"""Example linear regression training script for Ascend NPUs."""
import torch
import torch.nn as nn
import torch.optim as optim

try:
    import torch_npu  # noqa: F401  # Import to initialize Ascend NPU support.
    NPU_AVAILABLE = torch.npu.is_available()
except ImportError:
    NPU_AVAILABLE = False

if NPU_AVAILABLE:
    device = torch.device("npu:0")
else:
    # Fallback to CUDA or CPU when an Ascend NPU is not present.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Construct simple linear data y = 3x + 2 with noise.
x = torch.randn(100, 1, device=device)
y = 3 * x + 2 + 0.1 * torch.randn(100, 1, device=device)

# Define linear model.
model = nn.Linear(1, 1).to(device)

# Define loss function and optimizer.
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Training loop.
for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(x)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/100], Loss: {loss.item():.4f}")

# Print final results.
print("Trained weight:", model.weight.item())
print("Trained bias:", model.bias.item())
