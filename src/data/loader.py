import random
from datasets import load_dataset
import logging

logger = logging.getLogger(__name__)

def get_subset(dataset, limit: int, seed: int):
    """
    Returns a deterministic subset of the dataset.
    CRITICAL: Must correspond to the same indices across different runs.
    """
    if limit is None or limit >= len(dataset):
        return dataset
    
    # Create a stable random generator
    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    
    selected_indices = indices[:limit]
    # HF Datasets .select() reorders based on input list, so we sort for safety
    return dataset.select(sorted(selected_indices))

def load_task_data(task_config, limit=None, seed=42):
    """
    Loads and preprocesses data for a specific task.
    """
    logger.info(f"Loading task: {task_config.name}")

    if task_config.name == "math":
        ds = load_dataset(task_config.dataset, split=task_config.split)

        # Handle level filtering
        level_filter = task_config.get('level_filter', None)
        if level_filter is not None:
            if isinstance(level_filter, int):
                # Single level (e.g., 5)
                logger.info(f"Filtering for Level {level_filter}")
                ds = ds.filter(lambda x: x['level'] == level_filter)
            elif isinstance(level_filter, (list, tuple)) and len(level_filter) == 2:
                # Range of levels (e.g., [3, 5])
                min_level, max_level = level_filter
                logger.info(f"Filtering for Levels {min_level}-{max_level}")
                ds = ds.filter(lambda x: min_level <= x['level'] <= max_level)
            elif isinstance(level_filter, (list, tuple)):
                # Specific levels (e.g., [1, 3, 5])
                logger.info(f"Filtering for Levels {level_filter}")
                ds = ds.filter(lambda x: x['level'] in level_filter)
        else:
            logger.info("Using all levels")

        # Filter by subject if specified
        subject_filter = task_config.get('subject_filter', None)
        if subject_filter:
            if isinstance(subject_filter, str):
                subject_filter = [subject_filter]
            logger.info(f"Filtering for subjects: {subject_filter}")
            ds = ds.filter(lambda x: x['subject'] in subject_filter)

    elif task_config.name == "gpqa":
        ds = load_dataset(task_config.dataset, task_config.subset_name, split=task_config.split)
    elif task_config.name == "code":
        ds = load_dataset(task_config.dataset, split=task_config.split)
    else:
        raise ValueError(f"Unknown task: {task_config.name}")

    return get_subset(ds, limit, seed)
