import json
from pathlib import Path
from threading import Lock

from django.conf import settings


STATUS_PATH = Path(settings.MEDIA_ROOT) / 'training_status.json'
_STATUS_LOCK = Lock()


def read_training_status():
    """Read the current training status from disk."""

    if not STATUS_PATH.exists():
        return {
            'status': 'idle',
            'progress': 0,
            'message': 'No training started yet.',
        }

    try:
        with STATUS_PATH.open('r', encoding='utf-8') as status_file:
            return json.load(status_file)
    except (OSError, json.JSONDecodeError):
        return {
            'status': 'idle',
            'progress': 0,
            'message': 'No training started yet.',
        }


def write_training_status(**updates):
    """Persist the current training status to disk for browser polling."""

    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _STATUS_LOCK:
        current_status = read_training_status()
        current_status.update(updates)
        with STATUS_PATH.open('w', encoding='utf-8') as status_file:
            json.dump(current_status, status_file, indent=2)
    return current_status
