import argparse
import os
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
IMPORT_DIRS = [
    SCRIPT_DIR,
    os.getcwd(),
    '/content',
    '/content/drive/MyDrive/AB COMVIS/code',
    '/content/drive/MyDrive/Comvis/code',
]
for import_dir in IMPORT_DIRS:
    if os.path.isdir(import_dir) and import_dir not in sys.path:
        sys.path.insert(0, import_dir)

if __package__:
    from .arsitektur_yolo_depth import NYUDepthV2Dataset, YOLO26nDepth
else:
    try:
        from arsitektur_yolo_depth import NYUDepthV2Dataset, YOLO26nDepth
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            'File arsitektur_yolo_depth.py belum ada di folder Colab. '
            'Upload/salin file tersebut ke folder yang sama dengan train_nyu_colab.py.'
        ) from error


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        print('PERINGATAN: GPU tidak aktif. Runtime Colab sebaiknya disetel ke T4/A100 GPU.')

    with h5py.File(args.nyu_path, 'r') as dataset_file:
        total_samples = dataset_file['images'].shape[0]

    sample_count = total_samples if args.max_samples <= 0 else min(args.max_samples, total_samples)
    indices = np.arange(sample_count)
    rng = np.random.default_rng(42)
    rng.shuffle(indices)
    split = max(1, int(sample_count * 0.9))

    train_set = NYUDepthV2Dataset(
        args.nyu_path,
        indices[:split],
        image_size=(args.input_width, args.input_height),
    )
    val_set = NYUDepthV2Dataset(
        args.nyu_path,
        indices[split:],
        image_size=(args.input_width, args.input_height),
    )

    model = YOLO26nDepth(width_mult=0.25, depth_mult=0.50).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.SmoothL1Loss(reduction='none')
    best_val = float('inf')
    train_history = []
    val_history = []

    print(f'Device: {device}')
    print(f'Dataset: {sample_count}/{total_samples} sampel')
    print(f'Train: {len(train_set)} | Validasi: {len(val_set)}')
    print(f'Epoch: {args.epochs} | Input: {args.input_width}x{args.input_height}')

    try:
        for epoch in range(args.epochs):
            model.train()
            train_total = 0.0
            train_order = rng.permutation(len(train_set))

            for position in train_order:
                image, target = train_set[int(position)]
                image = image.unsqueeze(0).to(device, non_blocking=True)
                target = target.unsqueeze(0).to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                prediction = model(image)['depth']
                valid = target > 0
                loss = criterion(prediction, target)[valid].mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_total += loss.item()

            model.eval()
            val_total = 0.0
            with torch.no_grad():
                for position in range(len(val_set)):
                    image, target = val_set[position]
                    prediction = model(image.unsqueeze(0).to(device))['depth']
                    target = target.unsqueeze(0).to(device)
                    valid = target > 0
                    val_total += criterion(prediction, target)[valid].mean().item()

            train_loss = train_total / max(1, len(train_set))
            val_loss = val_total / max(1, len(val_set))
            train_history.append(train_loss)
            val_history.append(val_loss)
            scheduler.step()

            print(
                f'Epoch [{epoch + 1:02d}/{args.epochs}] '
                f'Train: {train_loss:.6f} | Val: {val_loss:.6f} '
                f'LR: {scheduler.get_last_lr()[0]:.8f}'
            )

            if val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(), args.output)
                print(f'  checkpoint terbaik disimpan: {args.output}')

    finally:
        train_set.close()
        val_set.close()

    plt.figure(figsize=(8, 5))
    plt.plot(train_history, label='train')
    plt.plot(val_history, label='validation')
    plt.xlabel('Epoch')
    plt.ylabel('SmoothL1 loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.loss_plot, dpi=150)
    print(f'Grafik loss disimpan: {args.loss_plot}')
    print(f'Checkpoint final terbaik: {args.output}')
    print(f'Best validation loss: {best_val:.6f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train YOLO26n-depth on NYU Depth V2 in Google Colab')
    default_nyu_paths = [
        '/content/drive/MyDrive/AB COMVIS/code/data/nyu_depth_v2_labeled.mat',
        '/content/drive/MyDrive/Comvis/code/data/nyu_depth_v2_labeled.mat',
        './data/nyu_depth_v2_labeled.mat',
    ]
    default_nyu_path = next((path for path in default_nyu_paths if os.path.exists(path)), default_nyu_paths[0])
    parser.add_argument('--nyu-path', default=default_nyu_path)
    parser.add_argument('--output', default='yolo26n_depth_nyu_30epoch.pt')
    parser.add_argument('--loss-plot', default='nyu_loss_30epoch.png')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--max-samples', type=int, default=0)
    parser.add_argument('--input-width', type=int, default=320)
    parser.add_argument('--input-height', type=int, default=240)
    args, _ = parser.parse_known_args()
    if not os.path.exists(args.nyu_path):
        parser.error(
            f'File NYU tidak ditemukan: {args.nyu_path}. '
            'Gunakan --nyu-path dengan lokasi nyu_depth_v2_labeled.mat.'
        )
    train(args)
