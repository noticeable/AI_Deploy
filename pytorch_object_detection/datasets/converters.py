import json
import pathlib
import random
import xml.etree.ElementTree as ET

from PIL import Image


def load_yolo_classes(classes_path):
    path = pathlib.Path(classes_path)
    return [line.strip() for line in path.read_text(encoding='utf-8').splitlines()
            if line.strip()]


def build_coco_categories(class_names, start_id=0):
    return [{
        'id': start_id + index,
        'name': name,
        'supercategory': 'object',
    } for index, name in enumerate(class_names)]


def discover_images(images_dir, suffixes=None):
    images_dir = pathlib.Path(images_dir)
    suffixes = tuple(s.lower() for s in (suffixes or ('.jpg', '.jpeg', '.png', '.bmp', '.webp')))
    return sorted(path for path in images_dir.rglob('*')
                  if path.is_file() and path.suffix.lower() in suffixes)


def split_paths(image_paths, train_ratio=0.8, val_ratio=0.1, seed=0):
    image_paths = list(image_paths)
    if not image_paths:
        return [], [], []
    if train_ratio < 0 or val_ratio < 0 or train_ratio + val_ratio > 1:
        raise ValueError('train_ratio and val_ratio must satisfy 0 <= train_ratio + val_ratio <= 1.')
    generator = random.Random(seed)
    generator.shuffle(image_paths)
    train_end = int(len(image_paths) * train_ratio)
    val_end = train_end + int(len(image_paths) * val_ratio)
    return image_paths[:train_end], image_paths[train_end:val_end], image_paths[val_end:]


def split_paths_from_files(root_dir, train_file=None, val_file=None, test_file=None):
    root_dir = pathlib.Path(root_dir)

    def _read_names(file_path):
        if file_path is None:
            return []
        names = []
        for line in pathlib.Path(file_path).read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if not line:
                continue
            names.append(pathlib.Path(line).name)
        return names

    train_names = set(_read_names(train_file or root_dir / 'train.txt'))
    val_names = set(_read_names(val_file or root_dir / 'val.txt'))
    test_names = set(_read_names(test_file or root_dir / 'test.txt'))
    return train_names, val_names, test_names


def convert_yolo_to_coco(root_dir,
                         output_path,
                         classes_path=None,
                         image_paths=None,
                         class_id_offset=0):
    root_dir = pathlib.Path(root_dir)
    images_dir = root_dir / 'images'
    labels_dir = root_dir / 'labels'
    classes = load_yolo_classes(classes_path or root_dir / 'classes.txt')
    categories = build_coco_categories(classes, start_id=class_id_offset)
    image_paths = list(image_paths) if image_paths is not None else discover_images(images_dir)

    dataset = {
        'images': [],
        'annotations': [],
        'categories': categories,
    }
    annotation_id = 0

    for image_id, image_path in enumerate(image_paths):
        image_path = pathlib.Path(image_path)
        with Image.open(image_path) as image:
            width, height = image.size
        relative = image_path.relative_to(images_dir)
        dataset['images'].append({
            'id': image_id,
            'file_name': relative.as_posix(),
            'width': width,
            'height': height,
        })
        label_path = labels_dir / relative.with_suffix('.txt')
        if not label_path.exists():
            continue
        for raw_line in label_path.read_text(encoding='utf-8').splitlines():
            fields = raw_line.strip().split()
            if len(fields) != 5:
                continue
            class_id, cx, cy, bw, bh = map(float, fields)
            x1 = (cx - bw / 2.0) * width
            y1 = (cy - bh / 2.0) * height
            box_w = bw * width
            box_h = bh * height
            dataset['annotations'].append({
                'id': annotation_id,
                'image_id': image_id,
                'category_id': int(class_id) + class_id_offset,
                'bbox': [x1, y1, box_w, box_h],
                'area': box_w * box_h,
                'iscrowd': 0,
                'segmentation': [],
            })
            annotation_id += 1

    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding='utf-8')
    return output_path


def _get_and_check(root, name, length):
    values = root.findall(name)
    if len(values) == 0:
        raise NotImplementedError(f'Can not find {name} in {root.tag}.')
    if length > 0 and len(values) != length:
        raise NotImplementedError(f'The size of {name} is supposed to be {length}, but is {len(values)}.')
    if length == 1:
        return values[0]
    return values


def discover_voc_classes(xml_dir):
    classes = []
    for xml_path in sorted(pathlib.Path(xml_dir).glob('*.xml')):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for obj in root.findall('object'):
            name = _get_and_check(obj, 'name', 1).text
            if name not in classes:
                classes.append(name)
    return classes


def convert_voc_to_coco(xml_paths,
                        output_path,
                        class_names,
                        image_suffix='jpg',
                        class_id_offset=1,
                        skip_unknown=False):
    categories = {name: class_id_offset + index for index, name in enumerate(class_names)}
    dataset = {
        'info': ['none'],
        'license': ['none'],
        'images': [],
        'annotations': [],
        'categories': [{
            'id': category_id,
            'name': name,
            'supercategory': 'none',
        } for name, category_id in categories.items()],
    }

    annotation_id = 1
    for image_id, xml_path in enumerate(xml_paths):
        xml_path = pathlib.Path(xml_path)
        tree = ET.parse(xml_path)
        root = tree.getroot()
        size = _get_and_check(root, 'size', 1)
        width = int(_get_and_check(size, 'width', 1).text)
        height = int(_get_and_check(size, 'height', 1).text)
        dataset['images'].append({
            'id': image_id,
            'file_name': f'{xml_path.stem}.{image_suffix}',
            'width': width,
            'height': height,
        })
        for obj in root.findall('object'):
            name = _get_and_check(obj, 'name', 1).text
            if name not in categories:
                if skip_unknown:
                    continue
                categories[name] = max(categories.values(), default=class_id_offset - 1) + 1
                dataset['categories'].append({
                    'id': categories[name],
                    'name': name,
                    'supercategory': 'none',
                })
            bndbox = _get_and_check(obj, 'bndbox', 1)
            xmin = int(float(_get_and_check(bndbox, 'xmin', 1).text))
            ymin = int(float(_get_and_check(bndbox, 'ymin', 1).text))
            xmax = int(float(_get_and_check(bndbox, 'xmax', 1).text))
            ymax = int(float(_get_and_check(bndbox, 'ymax', 1).text))
            box_w = abs(xmax - xmin)
            box_h = abs(ymax - ymin)
            dataset['annotations'].append({
                'id': annotation_id,
                'image_id': image_id,
                'category_id': categories[name],
                'bbox': [xmin, ymin, box_w, box_h],
                'area': box_w * box_h,
                'iscrowd': 0,
                'ignore': 0,
                'segmentation': [],
            })
            annotation_id += 1

    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding='utf-8')
    return output_path
