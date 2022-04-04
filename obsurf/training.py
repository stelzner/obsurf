import torch
import numpy as np
from collections import defaultdict
from tqdm import tqdm


class BaseTrainer(object):
    ''' Base trainer class.
    '''

    def evaluate(self, val_loader, **kwargs):
        ''' Performs an evaluation.
        Args:
            val_loader (dataloader): pytorch dataloader
        '''
        eval_list = defaultdict(list)

        for data in tqdm(val_loader):
            eval_step_dict = self.eval_step(data, **kwargs)

            for k, v in eval_step_dict.items():
                eval_list[k].append(v)

        eval_dict = {k: torch.cat(v, 0).mean().item() for k, v in eval_list.items()}
        print('Evaluation results:')
        print(eval_dict)
        return eval_dict

    def train_step(self, *args, **kwargs):
        ''' Performs a training step.
        '''
        raise NotImplementedError

    def eval_step(self, *args, **kwargs):
        ''' Performs an evaluation step.
        '''
        raise NotImplementedError

    def visualize(self, *args, **kwargs):
        ''' Performs  visualization.
        '''
        raise NotImplementedError
