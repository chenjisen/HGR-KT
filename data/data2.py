import json

import dgl
import torch
from dgl import DGLHeteroGraph
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, TensorDataset

from config.KTArgs import KTArgs
from data.Record import Record
from data.json_seq import load_relation


class KTDataModule2(LightningDataModule):
    """
        N: number of students
            junyi: 3501
            ednet: 3501
            ass09: 2380
            nm09 : 20
        S: sequence size = 200
        M: max_concept_num
            junyi: 1
            ednet: 7
            ass09: 4
    """
    train: TensorDataset
    val: TensorDataset
    test: TensorDataset

    exercise_min_index = 1

    def __init__(self, args: KTArgs) -> None:
        super().__init__()

        self.fold = 0
        self.seq_size = args.seq_size
        self.num_workers = args.num_workers
        self.batch_size = args.batch_size
        self.show_len = 100
        self.exercise_count = args.exercise_count

        self.train = TensorDataset(*Record.load(args.data_dir / 'train.json', self.seq_size))
        self.test = TensorDataset(*Record.load(args.data_dir / 'test.json', self.seq_size))
        self.val = self.test

        if 'HAN' in args.model:
            self.c_e_relation = load_relation(args.data_dir / 'relation.json')
            with open(args.data_dir / args.han_c_graph_file) as f:
                self.c_c_relation = json.load(f)

    @property
    def heterograph(self) -> DGLHeteroGraph:
        r = Record(*self.train.tensors)
        s_e = {'s': torch.repeat_interleave(torch.arange(len(self.train)), r.length),
               'e': r.exercise[r.exercise > 0],
               'ce': r.combined_exercise[r.combined_exercise > 0]}

        c_e = {'e': [], 'ce': [], 'c': []}
        for e, c_list in self.c_e_relation.items():
            for c in c_list:
                for label in 0, 1:
                    if e < self.exercise_count:
                        c_e['e'].append(e)
                        c_e['c'].append(c)
                        c_e['ce'].append(Record.get_combined_exercise(e, label))
        for k in c_e:
            c_e[k] = torch.tensor(c_e[k])

        c_c = {'c1': [], 'c2': []}
        for c1, c2 in self.c_c_relation:
            c_c['c1'].append(c1)
            c_c['c2'].append(c2)

        a_e = {'e': [], 'ce': [], 'a': []}
        for e in range(1, self.exercise_count + 1):
            for a in 0, 1:
                a_e['e'].append(e)
                a_e['ce'].append(Record.get_combined_exercise(e, a))
                a_e['a'].append(a)
        a_e['ae'] = a_e['ce']
        for k in a_e:
            a_e[k] = torch.tensor(a_e[k])

        node_dict = {'e': torch.arange(1, self.exercise_count + 1),
                     'ce': torch.arange(2, 2 * self.exercise_count + 2)}

        # assert torch.equal(node_dict['e'], torch.unique(a_e['e']))
        # assert torch.equal(node_dict['ce'], a_e['ce'])

        graph_data = get_dict(s_e) | get_dict(c_e) | get_dict(a_e)
        graph_data['c', 'c-c', 'c'] = c_c['c1'], c_c['c2']
        graph_data['c', 'c-c@r', 'c'] = c_c['c2'], c_c['c1']
        for k, v in node_dict.items():
            graph_data |= get_dict({k: v})
        return dgl.heterograph(graph_data)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train, self.batch_size, True, num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val, self.batch_size, False, num_workers=self.num_workers)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test, self.batch_size, False, num_workers=self.num_workers)


def get_dict(d: dict) -> dict[tuple, tuple]:
    return {(x, x + '-' + y, y): (d[x], d[y]) for x in d for y in d}
