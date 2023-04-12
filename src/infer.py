"""
    Script for inferrence.
    
    ---

    Written & Maintained by: 
        Astromsoc
    Last Updated at:
        Apr 11, 2023
"""


import os
import torch
import argparse
from ruamel.yaml import YAML
yaml = YAML(typ='safe')


from src.utils import *
from src.model import *



class Inferer:

    def __init__(self, ckpt: str, device: str='cuda'):
        # archiving
        self.ckpt = ckpt
        self.device = device
        # build model instance
        self.expfolder = os.path.dirname(self.ckpt)
        self.expconfigs = ParamsObject(yaml.load(open(f"{self.expfolder}/configs.yaml", 'r')))
        # reload checkpoint
        self.load_model()
    

    def load_model(self):
        """
            Load a model checkpoint from specified filepath.
        """
        assert os.path.exists(self.ckpt), f"\n[** FILE NOT EXISTED **] Can't load from [{self.ckpt}].\n"
        loaded = torch.load(self.ckpt, map_location=torch.device(self.device))
        print(f"\n[** MODEL LOADED **] Successfully loaded checkpoint from [{self.ckpt}]\n")

        # load configs
        self.model_configs = loaded['configs']['model']
        self.model = TorchL2ROneLayerEncDecGruSeqPred(configs=self.model_configs)
        # other state dicts / saved attributes
        self.model.load_state_dict(loaded['model_state_dict'])
        idx_dicts = loaded['idx_dicts']
        self.idx2gp = list(idx_dicts['gp'].keys())
        self.chr2idx = idx_dicts['chr']
        # take model to device
        self.model.to(self.device)
    

    def encode_inputs(self, word: str):
        """
            Convert word string into indices.
        """
        return torch.tensor([self.chr2idx.get(c, self.chr2idx['<unk>']) for c in word],
                            device=self.device).unsqueeze(0)
    

    def decode_outputs(self, gpids: torch.tensor):
        outputs = list()
        for idx in gpids.squeeze(0).tolist():
            t = self.idx2gp[idx]
            if t == '<eos>': break
            outputs.append(t)
        return outputs


    def infer(self, word: str):
        # obtain output gp
        word_ids, word_len = self.encode_inputs(word), torch.tensor([len(word)]).to(self.device)
        pred_logits = self.model(input_ids=word_ids, input_lens=word_len)
        # obtain argmax indices & decode
        return self.decode_outputs(pred_logits.argmax(dim=-1))


    @staticmethod
    def gps_to_gpstr(gps: list):
        return ', '.join(["({}, {})".format(*gp) for gp in gps])




def main(args):

    # obtain device information
    device = ('cuda' if torch.cuda.is_available() else
              'mps' if torch.backends.mps.is_available() else 
              'cpu')

    # build inferer
    inferer = Inferer(args.ckpt)

    # set model to inference model
    inferer.model.eval()
    with torch.inference_mode():

        while True:
            # accept input word in lower case
            word = input("\n\nPlease input a word: \n")
            # convert
            gps = inferer.infer(word.lower())
            print(f"The (grapheme, phoneme) breakdown of [{word}] is:\n\t{gps}")
            


    




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Inferring the (grapheme, phoneme) breakdown of input words.")
    parser.add_argument(
        '--ckpt', '-c', type=str,
        help='(str) Filepath to the model checkpoint under experiment folder.'
    )
    args = parser.parse_args()

    main(args)