"""
    Script to train models.
    
    ---

    Written & Maintained by: 
        Astromsoc
    Last Updated at:
        Sep 12, 2023
"""



import os
import wandb
import torch
import torch.nn
import argparse
import Levenshtein
import numpy            as np

from src.utils          import *
from src.models         import *
from tqdm               import tqdm
from torch.utils.data   import DataLoader
from ruamel.yaml        import YAML
yaml = YAML(typ='safe')



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
                                     'luepoch': -1, 'luld': float('inf')})
        self.idx_dicts = {'gp': gp2idx, 'chr': chr2idx}
        # take out <eos> for faster reference
        self.dec_eos_idx = gp2idx['<eos>']
        self.dec_num_cls = len(gp2idx)
        self.device = device
        # other records
        self.bests = {'ld': float('inf'), 'epoch': -1}
        self.best_fps = list()
        self.train_losses = list()
        self.train_avg_losses = list()
        self.train_gradnorms = list()
        self.val_losses = list()
        self.train_lds = list()
        self.val_lds = list()
        self.epoch = 1
        self.use_wandb = False
        # init from configs
        self.init_from_configs()
        # take model to device
        self.model.to(self.device)
    

    def init_from_configs(self):
        # criterion
        self.criterion = nn.CrossEntropyLoss(**self.cfgs.criterion.__dict__)
        # optimizer 
        self.optimizer = torch.optim.AdamW(self.model.parameters(), 
                                           **self.cfgs.optimizer.__dict__)
        # scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=self.optimizer, **self.cfgs.lr_scheduler.configs.__dict__
        ) if self.cfgs.lr_scheduler.use else None
        # scaler
        self.scaler = torch.cuda.amp.GradScaler() if self.cfgs.scaler.use else None
    

    def update_tf_rate(self):
        if self.cfgs.tf_scheduler.use:
            if self.epoch > self.tf_rate.nie:
                if self.val_lds[-1] <= self.tf_rate.luld:
                    self.tf_rate.luepoch = self.epoch
                    self.tf_rate.luld = self.val_lds[-1]
                    self.tf_rate.val = max(self.tf_rate.val - self.tf_rate.int, self.tf_rate.min)
    

    def build_decout_masks(self, dec_lens: torch.tensor):
        # output dims
        bs, maxlen = dec_lens.size(0), dec_lens.max()
        dec_masks = torch.arange(0, maxlen).expand(bs, maxlen)
        # using [1:] to skip the uniform initial <sos> symbol 
        dec_masks = ((dec_masks < dec_masks.new(dec_lens).unsqueeze(-1))
                     .to(self.device)[:, 1:].flatten().to(torch.int))
        # return flattened masks (batch_size * (maxlen - 1), ) and total number of valid timesteps
        return dec_masks, dec_masks.sum()
    

    def compute_ld(self, hyps, refs):
        """
            REQUIRING both hypothesis and reference inputs don't have <sos> tags.
        """
        batch_size, total_ld = len(refs), 0
        for i in range(batch_size):
            # truncate before the first <eos> tag
            hyp, ref = list(), list()
            for t in hyps[i].tolist():
                if t == self.dec_eos_idx: break
                hyp.append(t)
            for t in refs[i].tolist():
                if t == self.dec_eos_idx: break
                ref.append(t)
            total_ld += Levenshtein.distance(hyp, ref)
        return total_ld / batch_size


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
            if isinstance(in_ids, tuple):
                in_ids  = (in_ids[0].to(self.device), in_ids[1].to(self.device))
                out_ids = out_ids.to(self.device)
            else:
                in_ids, out_ids = in_ids.to(self.device), out_ids.to(self.device)
            in_lens = in_lens.to(self.device)
            # obtain logits per timestep
            out_logits = self.model(in_ids, in_lens, out_ids, self.tf_rate.val)
            # build decoder output masks
            dec_masks, valid_sum = self.build_decout_masks(out_lens)
            # compute loss: (batch_size * dec_max_len)
            if self.scaler:
                with torch.cuda.amp.autocast():
                    loss = (self.criterion(out_logits.view(-1, self.dec_num_cls), 
                                           out_ids[:, 1:].flatten()) 
                                           * dec_masks).sum() / valid_sum
            else:
                loss = (self.criterion(out_logits.view(-1, self.dec_num_cls), 
                                       out_ids[:, 1:].flatten()) 
                                       * dec_masks).sum() / valid_sum
            # compute metrics
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
                lr=self.optimizer.param_groups[0]['lr'],
                tf_rate=f"{self.tf_rate.val:.2f}"
            )
            tqdm_bar.update()
            # push to wandb
            if self.use_wandb:
                wandb.log({"train_loss_per_batch": train_loss_this_epoch[i],
                           "grad_norm_per_batch": grad_norm[i]})
        tqdm_bar.close()
        # clear
        del in_ids, out_ids, in_lens, out_lens
        torch.cuda.empty_cache()

        return train_loss_this_epoch.mean(), grad_norm.mean()


    def eval_some_loader(self, loader, annotation: str=""):
        # build tqdm progress bar
        tqdm_bar = tqdm(total=len(loader), leave=False, dynamic_ncols=True,
                        desc=f"evaluating epoch [{self.epoch:<3}] on [{annotation}]...")
        loss_this_epoch = np.zeros((len(loader),))
        ld_this_epoch = np.zeros((len(loader),))
        # switch to evaluation mode
        self.model.eval()
        with torch.inference_mode():
            # iterate through batches
            for i, (in_ids, out_ids, in_lens, out_lens) in enumerate(loader):
                # take to device
                if isinstance(in_ids, tuple):
                    in_ids  = (in_ids[0].to(self.device), in_ids[1].to(self.device))
                    out_ids = out_ids.to(self.device)
                else:
                    in_ids, out_ids = in_ids.to(self.device), out_ids.to(self.device)
                in_lens = in_lens.to(self.device)
                # eliminating the first <sos> token
                dec_max_len = out_ids.size(1) - 1 
                # build decoder output masks
                dec_masks, valid_sum = self.build_decout_masks(out_lens)
                # mixed precision
                if self.scaler:
                    with torch.cuda.amp.autocast():
                        # obtain logits per timestep
                        out_logits = self.model(in_ids, in_lens, None, self.tf_rate.val)
                        # compute loss: (batch_size * dec_max_len)
                        loss = (self.criterion(out_logits[:, :dec_max_len, :].reshape(-1, self.dec_num_cls), 
                                               out_ids[:, 1:].flatten()) 
                                               * dec_masks).sum() / valid_sum
                else:
                    # obtain logits per timestep
                    out_logits = self.model(in_ids, in_lens, None, self.tf_rate.val)
                    # compute loss: (batch_size * dec_max_len)
                    loss = (self.criterion(out_logits[:, :dec_max_len, :].reshape(-1, self.dec_num_cls), 
                                           out_ids[:, 1:].flatten()) 
                                           * dec_masks).sum() / valid_sum
                # compute metrics
                loss_this_epoch[i] = loss.item()
                ld_this_epoch[i] = self.compute_ld(out_logits.argmax(dim=-1), 
                                                   out_ids[:, 1:])
                # update batch bar
                tqdm_bar.set_postfix(loss=f"{loss_this_epoch[i]:.6f}", 
                                     ld=f"{ld_this_epoch[i]:.2f}")
                tqdm_bar.update()

        tqdm_bar.close()

        return loss_this_epoch.mean(), ld_this_epoch.mean()


    def eval_epoch(self):
        # evaluate model for this epoch on training set
        train_loss_this_epoch, train_ld_this_epoch = self.eval_some_loader(
            loader=self.trn_loader, annotation='train'
        )
        val_loss_this_epoch, val_ld_this_epoch = self.eval_some_loader(
            loader=self.val_loader, annotation='val'
        )
        # record
        self.train_losses.append(train_loss_this_epoch)
        self.train_lds.append(train_ld_this_epoch)
        self.val_losses.append(val_loss_this_epoch)
        self.val_lds.append(val_ld_this_epoch)
        # push to wandb
        if self.use_wandb:
            wandb.log({"train_loss_per_epoch": train_loss_this_epoch,
                       "train_ld_per_epoch": train_ld_this_epoch,
                       "val_loss_per_epoch": val_loss_this_epoch,
                       "val_ld_per_epoch": val_ld_this_epoch})

    

    def train(self, expcfgs: dict):
        """
            Train the current model for what's been specified in input experiment configs.
        """
        # if finetuning: load checkpoint
        if expcfgs.finetune.use: self.load_model(expcfgs.finetune.ckpt)
        # update wandb settings
        self.use_wandb = expcfgs.wandb.use

        while self.epoch <= expcfgs.epoch:
            # updating model
            train_avg_loss, train_avg_grad_norm = self.train_epoch()
            # record
            self.train_avg_losses.append(train_avg_loss)
            self.train_gradnorms.append(train_avg_grad_norm)
            # push to wandb
            if self.use_wandb:
                wandb.log({'train_avg_loss_per_epoch': self.train_avg_losses[-1],
                           'train_gradnorm_per_epoch': self.train_gradnorms[-1],
                           'lr_per_epoch': self.optimizer.param_groups[0]['lr'],
                           'tf_rate_per_epoch': self.tf_rate.val})
            # evaluate models
            self.eval_epoch()
            # update learning rate
            if self.scheduler: self.scheduler.step(self.val_lds[-1])
            # save model
            self.save_model(expcfgs)
            # increment epoch by 1
            self.epoch += 1


    def save_model(self, expcfgs: str):
        """
            Save a model checkpoint to specified experiment folder.
        """
        # check if a lower val MSE is reached or the bests are not reached or it's the last epoch
        if (self.val_lds[-1] < self.bests['ld'] 
            or len(self.best_fps) < self.cfgs.max_saved_ckpts
            or self.epoch == expcfgs.epoch):
            # update best model stats
            if self.val_lds[-1] < self.bests['ld']:
                self.bests = {'ld': self.val_lds[-1], 'epoch': self.epoch}
            # sort the saved checkpoints (before reaching maximum storage)
            if len(self.best_fps) < self.cfgs.max_saved_ckpts:
                self.best_fps = [self.best_fps[i] for i in sorted(list(range(len(self.best_fps))), 
                                                                  key=lambda i: -self.val_lds[i])]
            # delete the oldest checkpoint if exceeding max storage
            if len(self.best_fps) == self.cfgs.max_saved_ckpts:
                os.remove(self.best_fps.pop(0))
            # create folder if not existed
            if not os.path.exists(expcfgs.folder):
                os.makedirs(expcfgs.folder, exist_ok=True)
            # add new filepath
            if (self.epoch != expcfgs.epoch or self.bests['ld'] == self.val_lds[-1]):
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
                'train_avg_losses': self.train_avg_losses,
                'train_lds': self.train_lds,
                'train_gradnorms': self.train_gradnorms,
                'val_losses': self.val_losses,
                'val_lds': self.val_lds,
                'idx_dicts': self.idx_dicts, 
                'tf_rate': self.tf_rate,
                'configs': {'trainer': self.cfgs, 
                            'exp': expcfgs, 
                            'model': self.model.configs}
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
        self.init_from_configs()
        # other state dicts / saved attributes
        self.model.load_state_dict(loaded['model_state_dict'])
        self.optimizer.load_state_dict(loaded['optim_state_dict'])
        self.epoch = loaded['epoch'] + 1
        self.best_fps = loaded['best_fps']
        self.bests = loaded['bests']
        self.train_losses = loaded['train_losses']
        self.train_avg_losses = loaded['train_avg_losses']
        self.train_lds = loaded['train_lds']
        self.train_gradnorms = loaded['train_gradnorms']
        self.val_losses = loaded['val_losses']
        self.val_lds = loaded['val_lds']
        self.idx_dicts = loaded['idx_dicts']
        self.tf_rate = loaded['tf_rate']




def main(args):
    # obtain device info
    device = ('cuda' if torch.cuda.is_available() else 
              'mps' if torch.backends.mps.is_available() else 
              'cpu')
    print(f"\nNow running on device: [{device}]\n")

    # build parameters class
    configs = ParamsObject(yaml.load(open(args.cfgfp, 'r')))

    # fix random seeds
    np.random.seed(configs.SEED)
    torch.manual_seed(configs.SEED)

    # load datasets & build loaders
    dataset_class   = GPPOSDatasetWithLabels if args.use_pos else GPDatasetWithLabels
    pos_kwargs      = {'pos2idx_txt_filepath': configs.pos2idx_filepath if args.use_pos else None}
    train_dataset   = dataset_class(dataset_pkl_filepath=configs.train_filepath,
                                    gp2idx_txt_filepath=configs.gp2idx_filepath,
                                    chr2idx_txt_filepath=configs.chr2idx_filepath,
                                    test_coherence=True, 
                                    **pos_kwargs)
    val_dataset     = dataset_class(dataset_pkl_filepath=configs.val_filepath,
                                    gp2idx_txt_filepath=configs.gp2idx_filepath,
                                    chr2idx_txt_filepath=configs.chr2idx_filepath,
                                    **pos_kwargs)
    train_loader    = DataLoader(dataset=train_dataset,
                                 collate_fn=train_dataset.collate_fn,
                                 **configs.train_loader.__dict__)
    val_loader      = DataLoader(dataset=val_dataset,
                                 collate_fn=val_dataset.collate_fn,
                                 **configs.val_loader.__dict__)

    # add attributes to model configs
    configs.torch_model.init_dec_idx = train_dataset.gp2idx['<sos>']
    configs.torch_model.num_inp      = len(train_dataset.chr2idx)
    configs.torch_model.num_cls      = len(train_dataset.gp2idx)
    configs.torch_model.enc_pad_idx  = train_dataset.chr2idx['<pad>']
    configs.torch_model.dec_pad_idx  = train_dataset.gp2idx['<eos>']
    # initiate model
    model_class = TorchL2ROneLayerEncDecGruSeqPredWithPOS if args.use_pos else TorchL2ROneLayerEncDecGruSeqPred
    model = model_class(configs=configs.torch_model).to(device)
    
    # print model summary (using simple prints for concerns over repeated inferrence steps)
    print(model)

    # build trainer class
    trainer = Trainer(cfgs=configs.trainer, model=model, trn_loader=train_loader,
                      gp2idx=train_dataset.gp2idx, chr2idx=train_dataset.chr2idx,
                      val_loader=val_loader, device=device)

    # build wandb name if used
    if configs.exp.wandb.use:
        configs.exp.wandb.configs.name = (f"encembchr-{configs.torch_model.enc_emb_dim_chr}-"
                                          f"encembpos-{configs.torch_model.enc_emb_dim_pos}-"
                                          f"enchid-{configs.torch_model.encgru.hidden_size}-"
                                          f"decemb-{configs.torch_model.dec_emb_dim}-"
                                          f"dechid-{configs.torch_model.decgru.hidden_size}-"
                                          f"clslin-{configs.torch_model.cls_lin_dim}-"
                                          f"bs-{configs.train_loader.batch_size}-"
                                          f"initf-{configs.trainer.init_tf_rate}" +
                                         (f"-{configs.exp.annotation}" if configs.exp.annotation 
                                          else ""))
        wandb.init(config=configs, **configs.exp.wandb.configs.__dict__)
        # revise exp folder with the new ablation name
        configs.exp.folder = os.path.join(configs.exp.folder, configs.exp.wandb.configs.name)

    # archiving config files
    if not os.path.exists(configs.exp.folder):
        os.makedirs(configs.exp.folder, exist_ok=True)
    os.system(f"cp {args.cfgfp} {configs.exp.folder}/configs.yaml")

    # pretrain / finetune the model
    trainer.train(configs.exp)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Training model w/ PyTorch.")
    parser.add_argument(
        '--cfgfp', '-f', type=str, default="cfgs/sample-gppred-train.yaml",
        help="(str) Filepath to the training configurations."
    )
    parser.add_argument(
        '--use_pos', '-p', action="store_true",
        help="(bool) Whether to include POS tags as partial inputs."
    )
    args = parser.parse_args()

    main(args)
    