"""
    Utility classes / functions in use.
    
    ---

    Written & Maintained by: 
        Astromsoc
    Last Updated at:
        Sep 12, 2023
"""


import re
import torch
import pickle

from torch.utils.data       import Dataset
from torch.nn.utils.rnn     import pad_sequence



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

    CHRLINE_REGEX   = re.compile(r"([a-z \.'<>-]+)\t([\d]+)")
    GPPLINE_REGEX   = re.compile(r"\(([a-z \.'<>_-]+), ([A-Z/ ]+)\)\t([\d]+)")

    """
        Dataset for (grapheme, phoneme) predictions, with labels
    """
    def __init__(self, 
                 dataset_pkl_filepath: str, 
                 gp2idx_txt_filepath: str,
                 chr2idx_txt_filepath: str,
                 test_coherence: bool=False,
                 pos2idx_txt_filepath: str=None):
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
        self.gp2idx  = {(g, p): idx for l in open(self.filepaths['gp2idx'], 'r') if l
                                    for (g, p, idx) in [self.parse_gp2idx_line(l)]}
        self.chr2idx = {c: idx for l in open(self.filepaths['chr2idx'], 'r') if l
                                    for (c, idx) in [self.parse_chr2idx_line(l)]}
        # add beginning, ending & padding symbols
        self.gp2idx['<sos>']  = len(self.gp2idx)
        self.gp2idx['<eos>']  = len(self.gp2idx)
        self.chr2idx['<pad>'] = len(self.chr2idx)
        # reverse dict (using list)
        self.idx2gp  = list(self.gp2idx.keys())
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
        m = re.match(self.GPPLINE_REGEX, gpidxline)
        return m.group(1), m.group(2), int(m.group(3))

    def parse_chr2idx_line(self, chridxline: str):
        m = re.match(self.CHRLINE_REGEX, chridxline)
        return m.group(1), int(m.group(2))
    
    def collate_fn(self, batch):
        inputs, outputs = zip(*batch)
        # obtain original lengths
        inlens  = [len(u) for u in inputs]
        outlens = [len(u) for u in outputs]
        # pad both inputs & outputss
        inputs = pad_sequence(inputs, batch_first=True, padding_value=self.chr2idx['<pad>'])
        outputs = pad_sequence(outputs, batch_first=True, padding_value=self.gp2idx['<eos>'])
        # in order: X, y, X_lens, y_lens
        return (inputs, outputs, 
                torch.tensor(inlens, dtype=torch.long), 
                torch.tensor(outlens, dtype=torch.long))




class GPPOSDatasetWithLabels(Dataset):

    GPPLINE_REGEX   = re.compile(r"\(([a-z \.'<>_-]+), ([A-Z/ ]+)\)\t([\d]+)")
    CHRLINE_REGEX   = re.compile(r"([a-z \.'<>-]+)\t([\d]+)")
    POSLINE_REGEX   = re.compile(r"(.*)[(, ?)\t]([\d]+)")

    """
        Dataset for (grapheme, phoneme) predictions, with labels & POS tags
    """

    def __init__(self, 
                 dataset_pkl_filepath: str, 
                 gp2idx_txt_filepath: str,
                 chr2idx_txt_filepath: str,
                 pos2idx_txt_filepath: str,
                 test_coherence: bool=False):
        super().__init__()
        # archiving
        self.filepaths = {
            'dataset': dataset_pkl_filepath,
            'gp2idx': gp2idx_txt_filepath,
            'chr2idx': chr2idx_txt_filepath,
            'pos2idx': pos2idx_txt_filepath
        }
        # load dictionaries
        for l in open(self.filepaths['gp2idx'], 'r'): 
            self.parse_gp2idx_line(l)
        self.gp2idx  = {(g, p): idx for l in open(self.filepaths['gp2idx'], 'r') if l
                                    for (g, p, idx) in [self.parse_gp2idx_line(l)]}
        self.chr2idx = {c: idx for l in open(self.filepaths['chr2idx'], 'r') if l
                                    for (c, idx) in [self.parse_chr2idx_line(l)]}
        self.pos2idx = {p: idx for l in open(self.filepaths['pos2idx'], 'r') if l and not l.startswith('#')
                                    for (p, idx) in [self.parse_pos2idx_line(l)]}
        # add beginning, ending & padding symbols
        self.gp2idx['<sos>']  = len(self.gp2idx)
        self.gp2idx['<eos>']  = len(self.gp2idx)
        self.chr2idx['<pad>'] = len(self.chr2idx)
        # reverse dict (using list)
        self.idx2gp  = list(self.gp2idx.keys())
        self.idx2chr = list(self.chr2idx.keys())
        self.idx2pos = list(self.pos2idx.keys())
        # load datasets
        wgps = pickle.load(open(self.filepaths['dataset'], 'rb'))
        # # coherence test
        # if test_coherence:
        #     for u in wgps:
        #         rec = [self.idx2gp[i] for i in u['gp_idx']]
        #         if u['gps'] != rec: print("[{}](golden) is different from [{}].".format(u['gps'], rec))
        # what really matters
        self.inputs_chr = [torch.tensor(u['word_idx']) for u in wgps]
        self.inputs_pos = [torch.tensor(u['pos']) for u in wgps]
        self.outputs = [torch.tensor([self.gp2idx['<sos>']] + u['gp_idx'] + [self.gp2idx['<eos>']]) for u in wgps]


    def __getitem__(self, index):
        return self.inputs_chr[index], self.inputs_pos[index], self.outputs[index]


    def __len__(self):
        return len(self.inputs_chr)


    def parse_gp2idx_line(self, gpidxline: str):
        m = re.match(self.GPPLINE_REGEX, gpidxline)
        return m.group(1), m.group(2), int(m.group(3))


    def parse_chr2idx_line(self, chridxline: str):
        m = re.match(self.CHRLINE_REGEX, chridxline)
        return m.group(1), int(m.group(2))


    def parse_pos2idx_line(self, posidxline: str):
        m = re.match(self.POSLINE_REGEX, posidxline)
        return m.group(1), int(m.group(2))
    

    def collate_fn(self, batch):
        inputs_chr, inputs_pos, outputs = zip(*batch)
        # obtain original lengths
        inlens  = [len(u) for u in inputs_chr]
        outlens = [len(u) for u in outputs]
        # pad both inputs & outputss
        inputs_chr = pad_sequence(inputs_chr, batch_first=True, padding_value=self.chr2idx['<pad>'])
        outputs = pad_sequence(outputs, batch_first=True, padding_value=self.gp2idx['<eos>'])
        # in order: X, y, X_lens, y_lens
        return ((inputs_chr, torch.tensor(inputs_pos, dtype=torch.long)), outputs, 
                torch.tensor(inlens, dtype=torch.long), 
                torch.tensor(outlens, dtype=torch.long))
