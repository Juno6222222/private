import argparse
import dataclasses
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.cuda import amp
from torchvision import datasets, transforms


@dataclasses.dataclass
class Metrics:
    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float


class SimpleCNN(nn.Module):
    """一个适用于MNIST手写数字的轻量级卷积模型。"""

    def __init__(self) -> None:
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        return self.classifier(x)


def accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    preds = output.argmax(dim=1)
    correct = (preds == target).sum().item()
    return correct / target.size(0)


def get_data_loaders(data_dir: Path, batch_size: int, num_workers: int) -> Tuple[data.DataLoader, data.DataLoader]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    train_set = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    val_set = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

    train_loader = data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = data.DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def train_one_epoch(
    model: nn.Module,
    dataloader: data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scaler: amp.GradScaler,
    device: torch.device,
    use_amp: bool,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    running_acc = 0.0

    for inputs, targets in dataloader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with amp.autocast(enabled=use_amp):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * inputs.size(0)
        running_acc += accuracy(outputs, targets) * inputs.size(0)

    dataset_size = len(dataloader.dataset)
    epoch_loss = running_loss / dataset_size
    epoch_acc = running_acc / dataset_size

    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    running_acc = 0.0

    for inputs, targets in dataloader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        running_loss += loss.item() * inputs.size(0)
        running_acc += accuracy(outputs, targets) * inputs.size(0)

    dataset_size = len(dataloader.dataset)
    epoch_loss = running_loss / dataset_size
    epoch_acc = running_acc / dataset_size

    return epoch_loss, epoch_acc


def save_checkpoint(state: Dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / "best_model.pt"
    torch.save(state, ckpt_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="在CUDA环境下训练MNIST分类模型")
    parser.add_argument("--data-dir", type=Path, default=Path("./data"), help="MNIST数据存储目录")
    parser.add_argument("--output-dir", type=Path, default=Path("./outputs"), help="模型与日志输出目录")
    parser.add_argument("--epochs", type=int, default=5, help="训练轮次")
    parser.add_argument("--batch-size", type=int, default=128, help="批大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader工作线程数")
    parser.add_argument("--use-amp", action="store_true", help="开启混合精度训练")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise EnvironmentError("当前环境不支持CUDA，请确认已经配置GPU驱动和CUDA。")

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    set_seed(args.seed)

    train_loader, val_loader = get_data_loaders(args.data_dir, args.batch_size, args.num_workers)

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scaler = amp.GradScaler(enabled=args.use_amp)

    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            use_amp=args.use_amp,
        )

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        metrics = Metrics(
            epoch=epoch,
            train_loss=train_loss,
            train_accuracy=train_acc,
            val_loss=val_loss,
            val_accuracy=val_acc,
        )

        print(
            f"Epoch {metrics.epoch:02d}: "
            f"train_loss={metrics.train_loss:.4f}, train_acc={metrics.train_accuracy:.4f}, "
            f"val_loss={metrics.val_loss:.4f}, val_acc={metrics.val_accuracy:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_accuracy": val_acc,
                    "epoch": epoch,
                    "args": dataclasses.asdict(args),
                },
                args.output_dir,
            )

    print(f"训练完成，最佳验证准确率: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
