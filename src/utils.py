"""
    Utility classes / functions in use.
    
    ---

    Written & Maintained by: 
        Astromsoc
    Last Updated at:
        Apr 8, 2023
"""


import re
import pickle

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence



class ParamsObject(object):
    """
        Convert yaml dictionary into object for cleaner calls 
    """
    def __init__(self, cfg_dict: dict):
        self.__dict__.update(cfg_dict)
        for k, v in self.__dict__.items():
            if isinstance(v, dict): 
                self.__dict__.update({k: ParamsObject(v)})



class GPDatasetWithLabels(Dataset):

    CHRLINE_REGEX = re.compile(r"([a-z \.'<>-]+)\t([\d]+)")
    GPLINE_REGEX = re.compile(r"\(([a-z \.'<>_-]+), ([A-Z/ ]+)\)\t([\d]+)")

    """
        Dataset for (grapheme, phoneme) predictions, with labels
    """
    def __init__(self, 
                 dataset_pkl_filepath: str, 
                 gp2idx_txt_filepath: str,
                 chr2idx_txt_filepath: str,
                 test_coherence: bool=False):
        super().__init__()
        # archiving
        self.filepaths = {
            'dataset': dataset_pkl_filepath,
            'gp2idx': gp2idx_txt_filepath,
            'chr2idx': chr2idx_txt_filepath
        }
        # load dictionaries
        for l in open(self.filepaths['gp2idx'], 'r'):
            self.parse_gp2idx_line(l)
        self.gp2idx = {(g, p): idx for l in open(self.filepaths['gp2idx'], 'r') if l
                                   for (g, p, idx) in [self.parse_gp2idx_line(l)]}
        self.chr2idx = {c: idx for l in open(self.filepaths['chr2idx'], 'r') if l
                                   for (c, idx) in [self.parse_chr2idx_line(l)]}
        # add beginning, ending & padding symbols
        self.gp2idx['<sos>'] = len(self.gp2idx)
        self.gp2idx['<eos>'] = len(self.gp2idx)
        self.gp2idx['<pad>'] = len(self.gp2idx)
        self.chr2idx['<pad>'] = len(self.chr2idx)
        # reverse dict (using list)
        self.idx2gp = list(self.gp2idx.keys())
        self.idx2chr = list(self.chr2idx.keys())
        # load datasets
        wgps = pickle.load(open(self.filepaths['dataset'], 'rb'))
        # coherence test
        if test_coherence:
            for u in wgps:
                rec = [self.idx2gp[i] for i in u['gp_idx']]
                if u['gps'] != rec: print("[{}](golden) is different from [{}].".format(u['gps'], rec))
        # what really matters
        self.inputs = [torch.tensor(u['word_idx']) for u in wgps]
        self.outputs = [torch.tensor([self.gp2idx['<sos>']] + u['gp_idx'] + [self.gp2idx['<eos>']]) 
                        for u in wgps]


    def __getitem__(self, index):
        return self.inputs[index], self.outputs[index]

    def __len__(self):
        return len(self.inputs)

    def parse_gp2idx_line(self, gpidxline: str):
        m = re.match(self.GPLINE_REGEX, gpidxline)
        return m.group(1), m.group(2), int(m.group(3))

    def parse_chr2idx_line(self, chridxline: str):
        m = re.match(self.CHRLINE_REGEX, chridxline)
        return m.group(1), int(m.group(2))
    
    def collate_fn(self, batch):
        inputs, outputs = zip(*batch)
        # obtain original lengths
        inlens = [len(u) for u in inputs]
        outlens = [len(u) for u in outputs]
        # pad both inputs & outputss
        inputs = pad_sequence(inputs, batch_first=True, padding_value=self.chr2idx['<pad>'])
        outputs = pad_sequence(outputs, batch_first=True, padding_value=self.gp2idx['<pad>'])
        # in order: X, y, X_lens, y_lens
        return inputs, outputs, torch.tensor(inlens), torch.tensor(outlens)


