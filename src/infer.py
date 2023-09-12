"""
    Script for inferrence.
    
    ---

    Written & Maintained by: 
        Astromsoc
    Last Updated at:
        Sep 12, 2023
"""


import os
import torch
import argparse

from src.utils      import *
from src.models     import *
from ruamel.yaml    import YAML
yaml = YAML(typ='safe')



class Inferer:

    GPPLINE_REGEX   = re.compile(r"\(([a-z \.'<>_-]+), ([A-Z/ ]+)\)\t([\d]+)")

    def __init__(self, 
                 ckpt: str, 
                 device: str='cuda', 
                 gpp2idx_filepath: str='data/gpp2idx.txt',
                 pos2idx_filepath: str='data/pos2idx.txt',
                 restricted: bool=False):
        # archiving
        self.ckpt               = ckpt
        self.device             = device
        self.restricted         = restricted
        self.gpp2idx_filepath   = gpp2idx_filepath
        self.pos2idx_filepath   = pos2idx_filepath
        # build model instance
        self.expfolder          = os.path.dirname(self.ckpt)
        self.expconfigs         = ParamsObject(yaml.load(open(f"{self.expfolder}/configs.yaml", 'r')))
        # reload checkpoint
        self.load_model()
        # build restricted search pools
        self.build_pyramids()
    

    def load_model(self):
        """
            Load a model checkpoint from specified filepath.
        """
        assert os.path.exists(self.ckpt), f"\n[** FILE NOT EXISTED **] Can't load from [{self.ckpt}].\n"
        loaded = torch.load(self.ckpt, map_location=torch.device(self.device))
        print(f"\n[** MODEL LOADED **] Successfully loaded checkpoint from [{self.ckpt}]\n")

        # load configs
        self.model_configs  = loaded['configs']['model']
        # TODO: ADD MULTI-MODEL-TYPE SELECTION
        self.model          = TorchL2ROneLayerEncDecGruSeqPredWithPOS(configs=self.model_configs)
        # other state dicts / saved attributes
        self.model.load_state_dict(loaded['model_state_dict'])
        idx_dicts           = loaded['idx_dicts']
        self.idx2gp         = list(idx_dicts['gp'].keys())
        self.chr2idx        = idx_dicts['chr']
        # add special marks
        self.chr2idx['<']   = len(self.chr2idx)
        self.chr2idx['>']   = len(self.chr2idx)
        self.chr2idx['_']   = len(self.chr2idx)
        # take model to device
        self.model.to(self.device)
        # build restricted search pyramids
        self.build_pyramids()
        # load POS to idx mapping
        self.pos2idx = {pos: int(idx) for l in open(self.pos2idx_filepath, 'r') if l and not l.startswith('#')
                                      for pos, idx in [l.strip().split(',')]}
        # load trivial constants for restricted search
        self.vowel_gpidlists = [[self.chr2idx[v]] for v in 'aeiou']
        self.silent_e_gpids = [self.idx2gp.index(('e', '//')), self.idx2gp.index(('e>', '//'))]
        self.silent_e_suffix = [self.chr2idx['_'], self.chr2idx['e']]


    def build_pyramids(self):
        """
            Build pyramids for restricted search.
        """
        self.chr2gps        = dict()
        self.gpidx2glen     = dict()
        self.silent_e_gpids = list()
        self.max_chr_len    = 0
        for idx, gp in enumerate(self.idx2gp):
            if not isinstance(gp, tuple) or gp[0] == '<unk>': continue
            chr_ids = tuple([self.chr2idx[c] for c in gp[0]])
            if chr_ids not in self.chr2gps: 
                self.chr2gps[chr_ids] = list()
            self.chr2gps[chr_ids].append(idx)
            self.gpidx2glen[idx] = (1                   if gp[0].endswith('_e') else
                                    len(gp[0]) - 1      if (gp[0].startswith('<') or gp[0].endswith('>')) else
                                    len(gp[0]))
            self.max_chr_len = max(self.max_chr_len, len(chr_ids))
    

    def encode_inputs(self, word: str):
        """
            Convert word string into indices.
        """
        return torch.tensor([self.chr2idx.get(c, self.chr2idx['<unk>']) for c in word],
                            device=self.device).unsqueeze(0)
    

    def decode_outputs(self, gpids: torch.tensor or list):
        outputs = list()
        # convert output type
        if not isinstance(gpids, list):
            gpids = gpids.squeeze(0).tolist()
        # decode
        for idx in gpids:
            t = self.idx2gp[idx]
            if t == '<eos>': break
            outputs.append(t)
        return outputs


    def infer(self, word: str, pos: str=None):
        # obtain output gp
        word_ids, word_len = self.encode_inputs(word), torch.tensor([len(word)]).to(self.device)
        if self.restricted:
            pred_logits = self.model.restricted_infer_single(
                input_chr_ids=word_ids,
                input_pos=self.pos2idx.get(pos, 0),
                c2gpidx=self.chr2gps,
                gpidx2glen=self.gpidx2glen,
                max_chr_len=self.max_chr_len,
                left_border_idx=self.chr2idx['<'],
                right_border_idx=self.chr2idx['>'],
                silent_e_suffix=self.silent_e_suffix,
                silent_e_gpids=self.silent_e_gpids,
                vowel_gpidlists=self.vowel_gpidlists
            )
        else:
            pred_logits = self.model(
                input_ids=word_ids, 
                input_lens=word_len
            ).argmax(dim=-1)
        # obtain argmax indices & decode
        return self.decode_outputs(pred_logits)


    @staticmethod
    def gps_to_gpstr(gps: list):
        return ', '.join(["({}, {})".format(*gp) for gp in gps])




def main(args):

    # obtain device information
    device = ('cuda' if torch.cuda.is_available() else
              'mps' if torch.backends.mps.is_available() else 
              'cpu')

    # build inferer
    inferer = Inferer(ckpt=args.ckpt, restricted=args.restricted)

    # set model to inference model
    inferer.model.eval()
    with torch.inference_mode():

        while True:
            # accept input word in lower case
            word = input("\n\nPlease input a word: \n").strip()
            if args.restricted:
                pos = input("\n\nPlease input its POS tag: \n").strip()
                gps = inferer.infer(word=word.lower(), pos=pos.upper())
                print(f"The (grapheme, phoneme) breakdown of [{word}](POS=[{pos}]) is:\n\t{gps}")
            else:
                gps = inferer.infer(word.lower())
                print(f"The (grapheme, phoneme) breakdown of [{word}] is:\n\t{gps}")
    



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Inferring the (grapheme, phoneme) breakdown of input words.")
    parser.add_argument(
        '--ckpt', '-c', type=str,
        help='(str) Filepath to the model checkpoint under experiment folder.'
    )
    parser.add_argument(
        '--restricted', '-r', action='store_true',
        help='(bool) Whether to use restricted search'
    )
    args = parser.parse_args()

    main(args)