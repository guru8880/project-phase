import os, sys
from pathlib import Path
from PIL import Image

def resize_images(src_root='data/chest_xray', dst_root='data_preprocessed', size=(150,150)):
    src_root = Path(src_root)
    dst_root = Path(dst_root)
    for split in ['train','val','test']:
        for cls in (src_root/split).iterdir():
            if not cls.is_dir(): continue
            out_dir = dst_root / split / cls.name
            out_dir.mkdir(parents=True, exist_ok=True)
            for imgf in cls.glob('*'):
                try:
                    img = Image.open(imgf).convert('RGB').resize(size)
                    img.save(out_dir / imgf.name)
                except Exception as e:
                    print('skip', imgf, e)

if __name__ == '__main__':
    resize_images()
    print('done preprocessing images')