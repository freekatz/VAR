import os
import argparse
import glob
import multiprocessing
from pathlib import Path

from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode


def resize_image(input_path, output_path, size=(512, 512)):
    try:
        transform = transforms.Compose([
            transforms.Resize(size, interpolation=InterpolationMode.LANCZOS)
        ])
        with Image.open(input_path) as img:
            resized_img = transform(img)
            resized_img.save(output_path)
            print(f"Processed: {output_path}")
    except Exception as e:
        print(f"Failed to process {input_path}: {e}")


def process_image_paths(image_paths, output_dir, size=(512, 512)):
    for input_path in image_paths:
        output_path = os.path.join(output_dir, os.path.basename(input_path))
        resize_image(input_path, output_path, size)


def resize_ffhq(input_dir, output_dir, size=(512, 512), num_workers=16):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = glob.glob(os.path.join(input_dir, "*.png"))
    assert len(image_paths) > 0
    print(f"Found {len(image_paths)} images to process.")

    chunk_size = len(image_paths) // num_workers + 1
    with multiprocessing.Pool(num_workers) as pool:
        pool.starmap(
            process_image_paths,
            [(image_paths[i:i+chunk_size], output_dir, size) for i in range(0, len(image_paths), chunk_size)]
        )

    print(f"All images have been resized and saved to {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=str)
    parser.add_argument('-o', type=str)
    parser.add_argument('--size', type=str, default='512x512')
    parser.add_argument('--num_workers', type=int, default=16)
    args = parser.parse_args()

    size = args.size.split('x')
    assert 0 < len(size) <= 2
    if len(size) == 1:
        h = w = int(size[0])
    else:
        h, w = (int(s) for s in size)

    assert args.i != args.o
    resize_ffhq(
        input_dir=args.i,
        output_dir=args.o,
        size=(h, w)
    )
