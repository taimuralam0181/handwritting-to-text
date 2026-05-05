import csv
import json
import random
from pathlib import Path

from datasets import load_dataset
from django.conf import settings


def get_handwriting_dataset_profiles():
    """Return configured handwriting dataset profiles."""

    return settings.HANDWRITING_DATASET_PROFILES


def resolve_handwriting_dataset_config(
    dataset_profile: str | None = None,
    dataset_id: str | None = None,
    output_dir: str | None = None,
):
    """Resolve a handwriting dataset profile into a concrete config dict."""

    profiles = get_handwriting_dataset_profiles()
    profile_name = dataset_profile or 'iam_line'

    if dataset_profile and dataset_profile not in profiles:
        available = ", ".join(sorted(profiles))
        raise ValueError(f"Unknown dataset profile '{dataset_profile}'. Available profiles: {available}")

    profile = profiles.get(profile_name, {})
    return {
        'profile_name': profile_name,
        'dataset_id': dataset_id or profile.get('dataset_id') or settings.HANDWRITING_DATASET_ID,
        'output_dir': output_dir or profile.get('output_dir') or settings.HANDWRITING_DATASET_DIR,
        'image_field': profile.get('image_field', 'image'),
        'text_field': profile.get('text_field'),
        'label_field': profile.get('label_field'),
        'label_offset': profile.get('label_offset', 0),
        'alphabet': profile.get('alphabet', ''),
        'validation_split_ratio': profile.get('validation_split_ratio'),
        'test_split_ratio': profile.get('test_split_ratio'),
        'csv_path': profile.get('csv_path'),
        'image_name_field': profile.get('image_name_field', 'image_name'),
        'description': profile.get('description', ''),
    }


def _write_prepared_split(split_records, split_dir: Path, split_name: str):
    """Write images and metadata for one prepared dataset split."""

    images_dir = split_dir / 'images'
    images_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = split_dir / 'metadata.jsonl'

    row_count = 0
    with metadata_path.open('w', encoding='utf-8') as metadata_file:
        for index, record in enumerate(split_records):
            image_source = Path(record['image_path'])
            text = (record.get('text') or '').strip()
            if not image_source.exists() or not text:
                continue

            image_filename = f'{split_name}_{index:05d}{image_source.suffix.lower() or ".png"}'
            image_target = images_dir / image_filename
            image_target.write_bytes(image_source.read_bytes())

            metadata_file.write(
                json.dumps(
                    {
                        'image_path': str(image_target.resolve()),
                        'text': text,
                        'split': split_name,
                    },
                    ensure_ascii=True,
                ) + '\n'
            )
            row_count += 1

    return {
        'rows': row_count,
        'metadata_file': str(metadata_path.resolve()),
        'images_dir': str(images_dir.resolve()),
    }


def _prepare_custom_csv_dataset(config, output_path: Path, sample_limit: int | None = None):
    """Prepare a local custom handwriting dataset from a CSV file."""

    csv_path = Path(config['csv_path'])
    if not csv_path.exists():
        raise FileNotFoundError(f'Custom dataset CSV not found: {csv_path}')

    dataset_root = csv_path.parent
    image_name_field = config.get('image_name_field', 'image_name')
    text_field = config.get('text_field', 'text')

    records = []
    with csv_path.open('r', encoding='utf-8-sig', newline='') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            image_name = (row.get(image_name_field) or '').strip()
            text = (row.get(text_field) or '').strip()
            if not image_name or not text:
                continue

            image_path = dataset_root / image_name
            if not image_path.exists():
                image_path = dataset_root / 'images' / image_name

            records.append(
                {
                    'image_path': str(image_path.resolve()),
                    'text': text,
                }
            )

    if not records:
        raise ValueError('No valid custom CSV rows found. Expected image_name,text entries.')

    missing_images = [record['image_path'] for record in records if not Path(record['image_path']).exists()]
    if missing_images:
        preview = missing_images[:5]
        raise FileNotFoundError(
            'Some custom dataset images are missing. '
            f'Examples: {preview}. Keep images beside the CSV file or inside datasets/images/.'
        )

    random.Random(42).shuffle(records)
    if sample_limit is not None:
        records = records[:sample_limit]

    total_count = len(records)
    validation_ratio = float(config.get('validation_split_ratio') or 0.2)
    test_ratio = float(config.get('test_split_ratio') or 0.1)

    test_count = max(1, int(total_count * test_ratio)) if total_count >= 3 else 0
    validation_count = max(1, int(total_count * validation_ratio)) if total_count >= 3 else 0
    if total_count - test_count - validation_count <= 0:
        validation_count = 1 if total_count >= 2 else 0
        test_count = 1 if total_count >= 3 else 0

    train_count = max(total_count - validation_count - test_count, 1)
    train_records = records[:train_count]
    validation_records = records[train_count:train_count + validation_count]
    test_records = records[train_count + validation_count:]

    if not validation_records:
        validation_records = train_records[:1]
    if not test_records and len(train_records) > 1:
        test_records = train_records[:1]

    summary = {
        'dataset_profile': config['profile_name'],
        'dataset_id': config['dataset_id'],
        'output_dir': str(output_path),
        'description': config.get('description', ''),
        'csv_path': str(csv_path.resolve()),
        'splits': {},
    }

    split_map = {
        'train': train_records,
        'validation': validation_records,
        'test': test_records,
    }
    for split_name, split_records in split_map.items():
        split_dir = output_path / split_name
        summary['splits'][split_name] = _write_prepared_split(split_records, split_dir, split_name)

    return summary


def _derive_text_label(item, split_dataset, config):
    """Convert a dataset row into the OCR target text."""

    text_field = config.get('text_field')
    if text_field:
        return (item.get(text_field) or '').strip()

    label_field = config.get('label_field')
    if not label_field:
        return ''

    label_value = item.get(label_field)
    if label_value is None:
        return ''

    label_names = getattr(split_dataset.features.get(label_field), 'names', None)
    if label_names and 0 <= int(label_value) < len(label_names):
        return str(label_names[int(label_value)]).strip()

    alphabet = config.get('alphabet', '')
    label_index = int(label_value) - int(config.get('label_offset', 0))
    if alphabet and 0 <= label_index < len(alphabet):
        return alphabet[label_index]

    return str(label_value).strip()


def _split_train_validation_if_needed(dataset, config):
    """Create validation from train when the source dataset does not provide one."""

    if 'validation' in dataset or 'train' not in dataset:
        return dataset

    validation_ratio = config.get('validation_split_ratio')
    if not validation_ratio:
        return dataset

    split_dataset = dataset['train'].train_test_split(test_size=validation_ratio, seed=42)
    dataset['train'] = split_dataset['train']
    dataset['validation'] = split_dataset['test']
    return dataset


def _prepare_huggingface_dataset(config, output_path: Path, sample_limit: int | None = None):
    """Prepare a Hugging Face handwriting dataset into local metadata and images."""

    dataset = load_dataset(config['dataset_id'])
    dataset = _split_train_validation_if_needed(dataset, config)

    split_name_map = {
        'train': 'train',
        'validation': 'validation',
        'test': 'test',
    }
    summary = {
        'dataset_profile': config['profile_name'],
        'dataset_id': config['dataset_id'],
        'output_dir': str(output_path),
        'description': config.get('description', ''),
        'splits': {},
    }

    image_field = config['image_field']
    for original_split, target_split in split_name_map.items():
        if original_split not in dataset:
            continue

        split_dataset = dataset[original_split]
        if sample_limit is not None:
            split_dataset = split_dataset.select(range(min(sample_limit, len(split_dataset))))

        split_dir = output_path / target_split
        images_dir = split_dir / 'images'
        images_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = split_dir / 'metadata.jsonl'

        row_count = 0
        with metadata_path.open('w', encoding='utf-8') as metadata_file:
            for index, item in enumerate(split_dataset):
                text = _derive_text_label(item, split_dataset, config)
                image = item.get(image_field)
                if not text or image is None:
                    continue

                image_filename = f'{target_split}_{index:05d}.png'
                image_path = images_dir / image_filename
                image.save(image_path)
                metadata_file.write(
                    json.dumps(
                        {
                            'image_path': str(image_path.resolve()),
                            'text': text,
                            'split': target_split,
                        },
                        ensure_ascii=True,
                    ) + '\n'
                )
                row_count += 1

        summary['splits'][target_split] = {
            'rows': row_count,
            'metadata_file': str(metadata_path.resolve()),
            'images_dir': str(images_dir.resolve()),
        }

    return summary


def prepare_handwriting_dataset(
    dataset_profile: str | None = None,
    dataset_id: str | None = None,
    output_dir: str | None = None,
    sample_limit: int | None = None,
):
    """
    Prepare a handwriting dataset into local train/validation/test metadata.

    Supported profiles:
    - custom_csv: your own local image_name,text CSV
    - iam_line: line-level handwriting OCR dataset
    - emnist_letters: isolated letter handwriting dataset
    """

    config = resolve_handwriting_dataset_config(
        dataset_profile=dataset_profile,
        dataset_id=dataset_id,
        output_dir=output_dir,
    )
    output_path = Path(config['output_dir'])
    output_path.mkdir(parents=True, exist_ok=True)

    if config['profile_name'] == 'custom_csv':
        summary = _prepare_custom_csv_dataset(config, output_path, sample_limit=sample_limit)
    else:
        summary = _prepare_huggingface_dataset(config, output_path, sample_limit=sample_limit)

    summary_path = output_path / 'dataset_summary.json'
    with summary_path.open('w', encoding='utf-8') as summary_file:
        json.dump(summary, summary_file, indent=2)

    return summary
