import json
import os
from collections import Counter

base = r'C:\Users\User\projectTA\aku-nak-jugak-1'

print('=== STRUKTUR FOLDER ===')
for root, dirs, files in os.walk(base):
    level = root.replace(base, '').count(os.sep)
    indent = ' ' * 2 * level
    print(f'{indent}{os.path.basename(root)}/')
    subindent = ' ' * 2 * (level + 1)
    for f in files:
        fpath = os.path.join(root, f)
        size = os.path.getsize(fpath)
        print(f'{subindent}{f} ({size} bytes)')

print()

# Baca setiap split
splits = ['train', 'valid', 'test']
for split in splits:
    ann_path = os.path.join(base, split, '_annotations.coco.json')
    if not os.path.exists(ann_path):
        print(f'[{split}] annotation file tidak ditemukan!')
        continue

    with open(ann_path, encoding='utf-8') as f:
        data = json.load(f)

    print(f'=== SPLIT: {split.upper()} ===')
    print(f'  Jumlah gambar   : {len(data["images"])}')
    print(f'  Jumlah anotasi  : {len(data["annotations"])}')

    if split == 'train':
        print()
        print('  --- CATEGORIES (KELAS) ---')
        for cat in data['categories']:
            print(f'    ID {cat["id"]}: {cat["name"]}')

        print()
        print('  --- DISTRIBUSI ANOTASI ---')
        cat_map = {cat['id']: cat['name'] for cat in data['categories']}
        class_counts = Counter(ann['category_id'] for ann in data['annotations'])
        for cat_id, count in sorted(class_counts.items()):
            name = cat_map.get(cat_id, f'ID_{cat_id}')
            print(f'    {name}: {count} objek')
    print()
