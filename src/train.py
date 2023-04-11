"""
    Script to train models.
    
    ---

    Written & Maintained by: 
        Astromsoc
    Last Updated at:
        Apr 11, 2023
"""



import os
import wandb
import argparse
from tqdm import tqdm
from ruamel.yaml import YAML
yaml = YAML(typ='safe')

import torch
import torch.nn
from torchsummaryX import summary
from torch.utils.data import DataLoader

from src.model import *
from src.utils import *




class Trainer:
    def __init__(self,
                 cfgs: ParamsObject,
                 model: nn.Module,
                 trn_loader: DataLoader,
                 val_loader: DataLoader,
                 gp2idx: dict,
                 chr2idx: dict,
                 device: str='cuda'):
        self.cfgs = cfgs
        self.model = model
        self.trn_loader = trn_loader
        self.val_loader = val_loader
        # take out for easier reference
        self.tf_rate = ParamsObject({'val': self.cfgs.init_tf_rate,
                                     'nie': self.cfgs.tf_scheduler.min_init_epochs,
                                     'int': self.cfgs.tf_scheduler.interval,
                                     'min': self.cfgs.tf_scheduler.min_tf_rate,
                                     'luepoch': -1, 'luloss': float('inf')})
        self.idx_dicts = {'gp': gp2idx, 'chr2idx': chr2idx}
        self.dec_num_cls = len(gp2idx)
        self.device = device
        # other records
        self.bests = {'loss': float('inf'), 'epoch': -1}
        self.train_losses = list()
        self.train_gradnorms = list()
        self.val_losses = list()
        self.epoch = 1
        # init from configs
        self.init_from_configs()
    

    def init_from_configs(self):
        # criterion
        self.criterion = nn.CrossEntropyLoss(**self.cfgs.criterion.__dict__)
        # optimizer 
        self.optimizer = torch.optim.AdamW(self.model.parameters(), 
                                           **self.cfgs.optimizer.__dict__)
        # scaler
        self.scaler = torch.cuda.amp.GradScaler() if self.cfgs.scaler.use else None
    

    def update_tf_rate(self):
        if self.cfgs.tf_scheduler.use:
            if (self.epoch > self.tf_rate.nie and
                self.tf_rate.val - self.tf_rate.int >= self.tf_rate.min):
                if self.val_losses[-1] <= self.tf_rate.luloss:
                    self.tf_rate.luepoch = self.epoch
                    self.tf_rate.luloss = self.val_losses[-1]
                    self.tf_rate.val -= self.tf_rate.int
    

    def build_decout_masks(self, dec_lens: torch.tensor):
        # output dims
        bs, maxlen = dec_lens.size(0), dec_lens.max()
        dec_masks = torch.arange(0, maxlen).expand(bs, maxlen)
        dec_masks = ((dec_masks < dec_masks.new(dec_lens).unsqueeze(-1))
                     .to(self.device).flatten().to(torch.int))
        # return flattened masks (batch_size * maxlen, ) and total number of valid timesteps
        return dec_masks, dec_masks.sum()


    def train_epoch(self):
        # build tqdm progress bar
        tqdm_bar = tqdm(total=len(self.trn_loader), leave=False, dynamic_ncols=True,
                        desc=f"training epoch [{self.epoch:<3}]")
        train_loss_this_epoch = np.zeros((len(self.trn_loader),))
        grad_norm = np.zeros((len(self.trn_loader),))
        # alternate tf rate
        self.update_tf_rate()
        # switch to training mode
        self.model.train()
        # iterate through batches
        for i, (in_ids, out_ids, in_lens, out_lens) in enumerate(self.trn_loader):
            # take to device
            in_ids, out_ids = in_ids.to(self.device), out_ids.to(self.device)
            in_lens, out_lens = in_lens.to(self.device), out_lens.to(self.device)
            # obtain logits per timestep
            out_logits = self.model(in_ids, out_ids, self.tf_rate.val)
            # build decoder output masks
            dec_masks, valid_sum = self.build_decout_masks(out_lens)
            # compute loss: (batch_size * dec_max_len)
            loss = (self.criterion(out_logits.view(-1, self.dec_num_cls), 
                                   out_ids.view(-1, self.dec_num_cls)) * dec_masks) / valid_sum
            train_loss_this_epoch[i] = loss.item()
            # backprop & update
            if self.scaler:
                with torch.cuda.amp.autocast():
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            # compute gradient norm
            grad_norm[i] = sum([p.grad.data.detach().norm(2) 
                                for p in self.model.parameters() if p.grad is not None]) ** 0.5
            # clear grad
            self.optimizer.zero_grad()
            # update batch bar
            tqdm_bar.set_postfix(
                train_loss=f"{train_loss_this_epoch[i]:.6f}",
                grad_norm=f"{grad_norm[i]:6f}",
                lr=self.optimizer.param_groups[0]['lr']
            )
            tqdm_bar.update()
            # push to wandb
            if self.cfgs.wandb.use:
                wandb.log({"train_loss_per_batch": train_loss_this_epoch[i],
                           "grad_norm_per_batch": grad_norm[i]})
        # clear
        del in_ids, out_ids, in_lens, out_lens
        torch.cuda.empty_cache()
        tqdm_bar.close()

        return train_loss_this_epoch.mean(), grad_norm.mean()



    def eval_epoch(self):
        # build tqdm progress bar
        tqdm_bar = tqdm(total=len(self.val_loader), leave=False, dynamic_ncols=True,
                        desc=f"evaluating epoch [{self.epoch:<3}]")
        val_loss_this_epoch = np.zeros((len(self.val_loader),))
        # switch to training mode
        self.model.eval()
        with torch.inference_mode():
            # iterate through batches
            for i, (in_ids, out_ids, in_lens, out_lens) in enumerate(self.val_loader):
                # take to device
                in_ids, out_ids = in_ids.to(self.device), out_ids.to(self.device)
                in_lens, out_lens = in_lens.to(self.device), out_lens.to(self.device)
                # obtain logits per timestep
                out_logits = self.model(in_ids, None, self.tf_rate.val)
                # build decoder output masks
                dec_masks, valid_sum = self.build_decout_masks(out_lens)
                # compute loss: (batch_size * dec_max_len)
                loss = (self.criterion(out_logits.view(-1, self.dec_num_cls), 
                                       out_ids.view(-1, self.dec_num_cls)) * dec_masks) / valid_sum
                val_loss_this_epoch[i] = loss.item()
                # update batch bar
                tqdm_bar.set_postfix(val_loss=f"{val_loss_this_epoch[i]:.6f}")
                tqdm_bar.update()
                # push to wandb
                if self.cfgs.wandb.use:
                    wandb.log({"val_loss_per_batch": val_loss_this_epoch[i]})
        # clear
        del in_ids, out_ids, in_lens, out_lens
        torch.cuda.empty_cache()
        tqdm_bar.close()

        return val_loss_this_epoch.mean()
    
    

    def train(self, expcfgs: dict):
        """
            Train the current model for what's been specified in input experiment configs.
        """
        # if finetuning: load checkpoint
        if expcfgs.finetune.use: self.load_model(expcfgs.finetune.ckpt)

        while self.epoch <= expcfgs.epoch:
            train_avg_loss, train_avg_grad_norm = self.train_epoch()
            val_avg_loss = self.eval_epoch()
            # record
            self.train_losses.append(train_avg_loss)
            self.train_gradnorms.append(train_avg_grad_norm)
            self.val_losses.append(val_avg_loss)
            # update learning rate
            if self.scheduler: self.scheduler.step(self.val_losses[-1])
            # push to wandb
            if self.cfgs.wandb.use:
                wandb.log({'train_loss_per_epoch': self.train_losses[-1],
                           'train_gradnorm_per_epoch': self.train_gradnorms[-1],
                           'val_loss_per_epoch': self.val_losses[-1],
                           'lr_per_epoch': self.optimizer.param_groups[0]['lr']})
            # save model
            self.save_model(expcfgs)
            # increment epoch by 1
            self.epoch += 1


    def save_model(self, expcfgs: str):
        """
            Save a model checkpoint to specified experiment folder.
        """
        # check if a lower val MSE is reached or the bests are not reached or it's the last epoch
        if (self.val_losses[-1] < self.bests['loss'] 
            or len(self.best_fps) < self.cfgs.max_saved_ckpts
            or self.epoch == expcfgs.epoch):
            # update best model stats
            if self.val_losses[-1] < self.bests['loss']:
                self.bests = {'loss': self.val_losses[-1], 'epoch': self.epoch}
            # sort the saved checkpoints (before reaching maximum storage)
            if len(self.best_fps) < self.cfgs.max_saved_ckpts:
                self.best_fps = [self.best_fps[i] for i in sorted(list(range(len(self.best_fps))), 
                                                                  key=lambda i: -self.val_losses[i])]
            # save checkpoint
            if len(self.best_fps) == self.cfgs.max_saved_ckpts:
                # delete the oldest checkpoint
                os.remove(self.best_fps.pop(0))
            # create folder if not existed
            if not os.path.exists(expcfgs.folder):
                os.makedirs(expcfgs.folder, exist_ok=True)
            # add new filepath
            if (self.epoch != expcfgs.epoch or self.bests['loss'] == self.val_losses[-1]):
                self.best_fps.append(os.path.join(expcfgs.folder, f"epoch-{self.epoch}.pt"))
            output_filepath = (self.best_fps[-1] if self.epoch != expcfgs.epoch 
                               else os.path.join(expcfgs.folder, f"epoch-{self.epoch}.pt"))
            # save model checkpoint
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optim_state_dict': self.optimizer.state_dict(),
                'epoch': self.epoch,
                'best_fps': self.best_fps,
                'bests': self.bests,
                'train_losses': self.train_losses,
                'train_gradnorms': self.train_gradnorms,
                'val_losses': self.val_losses,
                'idx_dicts': self.idx_dicts, 
                'configs': {'trainer': self.cfgs, 
                            'exp': expcfgs}
            }, output_filepath)
            print(f"\n[** MODEL SAVED **] Successfully saved checkpoint to [{self.best_fps[-1]}]\n")
    

    def load_model(self, ckpt_filepath: str):
        """
            Load a model checkpoint from specified filepath.
        """
        assert os.path.exists(ckpt_filepath), f"\n[** FILE NOT EXISTED **] Can't load from [{ckpt_filepath}].\n"
        loaded = torch.load(ckpt_filepath, map_location=torch.device(self.device))
        print(f"\n[** MODEL LOADED **] Successfully loaded checkpoint from [{ckpt_filepath}]\n")

        # load configs
        self.cfgs = loaded['configs']['trainer']
        # init from configs
        self.init_from_cfgs()
        # other state dicts / saved attributes
        self.model.load_state_dict(loaded['model_state_dict'])
        self.optimizer.load_state_dict(loaded['optim_state_dict'])
        self.epoch = loaded['epoch'] + 1
        self.best_fps = loaded['best_fps']
        self.bests = loaded['bests']
        self.train_losses = loaded['train_losses']
        self.train_gradnorms = loaded['train_gradnorms']
        self.val_losses = loaded['val_losses']
        self.idx_dicts = loaded['idx_dicts']




def main(args):
    # obtain device info
    device = ('cuda' if torch.cuda.is_available() else 
              'mps' if torch.backends.mps.is_available() else 
              'cpu')
    # device = 'cpu'
    print(f"\nNow running on device: [{device}]\n")

    # build parameters class
    configs = ParamsObject(yaml.load(open(args.cfgfp, 'r')))

    # load datasets & build loaders
    train_dataset = GPDatasetWithLabels(dataset_pkl_filepath=configs.train_filepath,
                                        gp2idx_txt_filepath=configs.gp2idx_filepath,
                                        chr2idx_txt_filepath=configs.chr2idx_filepath)
    val_dataset = GPDatasetWithLabels(dataset_pkl_filepath=configs.val_filepath,
                                      gp2idx_txt_filepath=configs.gp2idx_filepath,
                                      chr2idx_txt_filepath=configs.chr2idx_filepath)
    train_loader = DataLoader(dataset=train_dataset,
                              collate_fn=train_dataset.collate_fn,
                              **configs.train_loader.__dict__)
    val_loader = DataLoader(dataset=val_dataset,
                            collate_fn=val_dataset.collate_fn,
                            **configs.val_loader.__dict__)

    # add attributes to model configs
    configs.torch_model.init_dec_idx = train_dataset.gp2idx['<sos>']
    configs.torch_model.num_inp = len(train_dataset.chr2idx)
    configs.torch_model.num_cls = len(train_dataset.gp2idx)
    configs.torch_model.enc_pad_idx = train_dataset.chr2idx['<pad>']
    configs.torch_model.dec_pad_idx = train_dataset.gp2idx['<pad>']
    # initiate model
    model = TorchL2ROneLayerEncDecGruSeqPred(configs=configs.torch_model)
    
    # print model summary
    with torch.inference_mode():
        x, y, _, _ = next(iter(train_loader))
        print(summary(model, x, y, 1.0))

    # # build trainer class
    # trainer = Trainer(cfgs=configs.trainer,  model=model, trn_loader=train_loader,
    #                   gp2idx=train_dataset.gp2idx, chr2idx=train_dataset.chr2idx,
    #                   val_loader=val_loader, device=device)
    



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Training model w/ PyTorch.")

    parser.add_argument(
        '--cfgfp', '-f', type=str, default="cfgs/sample-gppred.yaml",
        help="(str) Filepath to the training configurations."
    )

    args = parser.parse_args()

    main(args)