import ast
import csv
import os
import time
from itertools import cycle
from pathlib import Path
from typing import Optional

import torch

from config.parser import Args

PROJECT_HOME_DIR = Path.home() / 'documents/projects'
DATA_DIR = str(PROJECT_HOME_DIR / 'data/knowledge-tracing')
OUTPUT_DIR = str(PROJECT_HOME_DIR / 'outputs/knowledge-tracing')


class KTArgs(Args):
    # Program arguments
    data_root: str = DATA_DIR
    output_root: str = OUTPUT_DIR
    dataset_root: str = 'lt-json'
    dataset_name: str = 'new_mini_09'
    model: str = 'DKT'
    cross_validation: bool = False
    n_splits: int = 5
    load: bool = False
    train: bool = True

    # Model specific arguments
    input_dim: int = 64  # unavailable for HAN
    hidden_dim: int = 64
    num_layers: int = 1

    lr: float = 1e-3
    dropout: float = 0.

    batch_size: int = 16
    seq_size: int = 200
    early_stop_patience: int = 10
    warm_up_step_count: int = 4000

    alpha: float = 0.05  # DHKT hyper parameter for loss

    exercise_count: int
    student_count: int
    concept_count: int
    max_concepts: int

    # trainer arguments
    default_root_dir: str
    cuda: bool = True
    gpu: int = 0
    gpus: Optional[list[int]]
    max_epochs: int = 100
    enable_model_summary: bool = False
    num_sanity_val_steps: int = 2
    determine = True

    random_seed: int = 20
    random_seeds: str = ""
    _random_seed_iter: cycle

    def process_args(self) -> None:
        self.weight_dir.mkdir(parents=True, exist_ok=True)
        self.default_root_dir = str(self.weight_dir)
        stat_path = Path(self.data_dir).parent / 'stat.csv'
        if stat_path.exists():
            self._read_stat(stat_path)

        if not self.cross_validation:
            self.n_splits = 1

        self.gpus = [self.gpu] if self.cuda and torch.cuda.is_available() else None

        if self.random_seeds and self.cross_validation:
            random_seed_list = ast.literal_eval(self.random_seeds)
        else:
            random_seed_list = [self.random_seed]
        self._random_seed_iter = cycle(random_seed_list)

        if hasattr(self, 'han_num_head'):
            self.han_num_heads = [self.han_num_head]

    def _read_stat(self, stat_path: Path) -> None:
        with open(stat_path) as f:
            for row in csv.DictReader(f):
                if self.dataset_name == row['dataset']:
                    self.exercise_count = int(row['exercise_count'])
                    self.student_count = int(row['student_count'])
                    self.concept_count = int(row['concept_count'])
                    self.max_concepts = int(row['max_concepts'])

    @property
    def num_workers(self) -> int:
        return min(os.cpu_count(), 32) if os.name != 'nt' else 0

    @property
    def device(self) -> torch.device:
        return torch.device('cuda' if self.gpus else 'cpu', self.gpu)

    @property
    def data_dir(self) -> Path:
        return Path(self.data_root) / self.dataset_root / self.dataset_name

    @property
    def weight_dir(self) -> Path:
        return Path(self.output_root) / 'weight' / f'{self.dataset_name}-{self.model}'

    @property
    def log_path(self) -> Path:
        return self.weight_dir / f'log-{time.strftime("%m%d_%H%M%S")}.txt'

    @property
    def ckpt_path(self) -> str:
        return str(max(self.weight_dir.glob('*.ckpt'), key=lambda f: f.stat().st_ctime)) if self.load else None

    @property
    def next_seed(self) -> int:
        return next(self._random_seed_iter)
