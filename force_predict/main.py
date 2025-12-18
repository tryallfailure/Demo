#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Real-time prediction printer and force curve plot.
Usage:
    python main.py --data test/binarydata
    python main.py --data rabbit/binarydata
"""
import argparse, glob, os, pandas as pd
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch, matplotlib.pyplot as plt
from torchvision import transforms
from utils.model import CustomResNet18


class ForceImageFolder(Dataset):
    def __init__(self, img_dir, transform):
        self.transform = transform
        self.paths = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        img = self.transform(img)
        return img, Path(self.paths[idx]).stem


def predict_and_plot(img_dir: Path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    loader = DataLoader(ForceImageFolder(str(img_dir), transform),
                        batch_size=1, shuffle=False)

    model_path = Path(__file__).resolve().parent / 'pth' / 'model.pth'
    model = CustomResNet18(num_classes=3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    preds, frames = [], []
    with torch.no_grad():
        for img, (name,) in loader:
            pred = model(img.to(device)).cpu().squeeze().tolist()
            preds.append(pred)
            frames.append(name)
            print(f'{name}  Pred_X: {pred[0]:.3f}  '
                  f'Pred_Y: {pred[1]:.3f}  Pred_Z: {pred[2]:.3f}')

    # load ground-truth if exists
    true_csv = (img_dir.parent / 'data' / 'data.csv').resolve()
    has_true = true_csv.exists()
    true_df = pd.DataFrame()
    if has_true:
        true_df = pd.read_csv(true_csv)
        true_df = true_df.rename(columns=lambda c: c.strip().lower())
        true_df = true_df.rename(columns={'x': 'True_X', 'y': 'True_Y', 'z': 'True_Z'})
        true_df['image'] = true_df['image'].astype(str).str.strip()
        true_df = true_df[true_df['image'].isin(frames)]
        order_map = {name: idx for idx, name in enumerate(frames)}
        true_df['order'] = true_df['image'].map(order_map)
        true_df = true_df.sort_values('order').reset_index(drop=True)
        has_true = not true_df.empty

    # plot
    plt.figure(figsize=(8, 5))
    x = range(len(frames))
    for axis, color in zip(['X', 'Y', 'Z'], ['r', 'g', 'b']):
        plt.plot(x, [p['XYZ'.index(axis)] for p in preds],
                 label=f'Pred {axis}', color=color, ls='--', lw=2)
        if has_true:
            plt.plot(x, true_df[f'True_{axis}'],
                     label=f'True {axis}', color=color, ls='-', lw=2)

    plt.title('True vs Predicted Force' if has_true else 'Predicted Force')
    plt.xlabel('Index')
    plt.ylabel('Force / N')
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='sub-folder under repo root')
    args = parser.parse_args()
    img_dir = Path(__file__).resolve().parent / args.data.replace('\\', '/')
    if not img_dir.exists():
        raise FileNotFoundError(f'Folder not found: {img_dir}')
    predict_and_plot(img_dir)