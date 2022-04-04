import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
import numpy as np

import os
import argparse
import time

from obsurf import config, data
from obsurf.checkpoints import CheckpointIO
import obsurf.model.config as model_config


class LrScheduler():
    """ Implements a learning rate schedule with warum up and decay """
    def __init__(self, peak_lr=4e-4, peak_it=10000, decay_rate=0.5, decay_it=100000):
        self.peak_lr = peak_lr
        self.peak_it = peak_it
        self.decay_rate = decay_rate
        self.decay_it = decay_it

    def get_cur_lr(self, it):
        if it < self.peak_it:
            return self.peak_lr * (it / self.peak_it)
        it_since_peak = it - self.peak_it
        return self.peak_lr * (self.decay_rate ** (it_since_peak / self.decay_it))

    def update_every(self, it):
        if it <= self.peak_it:
            return 1
        return self.decay_it // 100


if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser(
        description='Train a 3D reconstruction model.'
    )
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
    parser.add_argument('--exit-after', type=int, default=1000000,
                        help='Checkpoint and exit after specified number of seconds'
                             'with exit code 2.')
    parser.add_argument('--dummy', action='store_true', help='Use small dummy dataset for fast debugging')
    parser.add_argument('--testnow', action='store_true', help='Run eval on startup')
    parser.add_argument('--visnow', action='store_true', help='Run visualization on startup')
    parser.add_argument('--maxlen', type=int, help='Limits the size of the validation set')
    parser.add_argument('--full_scale', action='store_true', help='Eval on full resolution')
    parser.add_argument('--rtpt', type=str, help='Use rtpt to set process name with given initials')

    args = parser.parse_args()
    testnow = args.testnow
    visnow = args.visnow
    cfg = config.load_config(args.config, 'configs/default.yaml')
    is_cuda = (torch.cuda.is_available() and not args.no_cuda)
    device = torch.device("cuda" if is_cuda else "cpu")
    gpu_count = torch.cuda.device_count() 

    if 'max_it' in cfg['training']:
        max_it = min(cfg['training']['max_it'], args.exit_after)
    else:
        max_it = args.exit_after

    exp_name = os.path.basename(os.path.dirname(args.config))
    
    if args.rtpt is not None:
        from rtpt import RTPT
        rtpt = RTPT(name_initials=args.rtpt, experiment_name=exp_name, max_iterations=max_it)


    # Shorthands
    out_dir = cfg['training']['out_dir']
    if out_dir == 'HERE!':
        out_dir = os.path.dirname(args.config)
        cfg['training']['out_dir'] = out_dir

    batch_size = cfg['training']['batch_size']
    backup_every = cfg['training']['backup_every']

    model_selection_metric = cfg['training']['model_selection_metric']
    if cfg['training']['model_selection_mode'] == 'maximize':
        model_selection_sign = 1
    elif cfg['training']['model_selection_mode'] == 'minimize':
        model_selection_sign = -1
    else:
        raise ValueError('model_selection_mode must be '
                         'either maximize or minimize.')

    # Output directory
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Dataset
    print('Loading training set...')
    train_dataset = config.get_dataset('train', cfg, dummy=args.dummy)
    print('Loading validation set...')
    val_dataset = config.get_dataset('val', cfg, dummy=args.dummy, max_len=args.maxlen)

    num_workers = cfg['training']['num_workers'] if 'num_workers' in cfg['training'] else 4
    num_workers *= gpu_count
    print(f'Using {num_workers} workers...')
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=False,
        worker_init_fn=data.worker_init_fn, persistent_workers=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=max(1, batch_size // 8), num_workers=num_workers, shuffle=False,
        pin_memory=False, worker_init_fn=data.worker_init_fn, persistent_workers=True)

    # For visualizations
    vis_loader_val = torch.utils.data.DataLoader(
        val_dataset, batch_size=12, pin_memory=True,
        worker_init_fn=data.worker_init_fn)
    vis_loader_train = torch.utils.data.DataLoader(
        train_dataset, batch_size=12, shuffle=True, pin_memory=True,
        worker_init_fn=data.worker_init_fn)

    print('Workers initialized')
    data_vis_val = next(iter(vis_loader_val))  # Validation set data for visualization

    train_dataset.mode = 'val'  # Get validation info from training set just this once
    data_vis_train = next(iter(vis_loader_train))  # Training set data for visualization
    train_dataset.mode = 'train'

    print('Visualization data loaded')

    model = model_config.get_model(cfg, device=device, dataset=train_dataset)

    print('Model created')

    if gpu_count > 1:
        print(f'Parallelizing model across {gpu_count} gpus')
        model.encoder = torch.nn.DataParallel(model.encoder)
        model.decoder = torch.nn.DataParallel(model.decoder)

    if 'lr_warmup' in cfg['training']:
        peak_it = cfg['training']['lr_warmup']
    else:
        peak_it = 10000
    decay_it = cfg['training']['decay_it'] if 'decay_it' in cfg['training'] else 100000

    lr_scheduler = LrScheduler(peak_it=peak_it, decay_it=decay_it)

    # Intialize training
    optimizer = optim.Adam(model.parameters(), lr=lr_scheduler.get_cur_lr(0))
    trainer = model_config.get_trainer(model, optimizer, cfg, device=device)
    checkpoint_io = CheckpointIO(out_dir, model=model, optimizer=optimizer)

    try:
        if os.path.exists(os.path.join(out_dir, f'model_{max_it}.pt')):
            load_dict = checkpoint_io.load(f'model_{max_it}.pt')
        else:
            load_dict = checkpoint_io.load('model.pt')
    except FileExistsError:
        load_dict = dict()

    epoch_it = load_dict.get('epoch_it', -1)
    it = load_dict.get('it', -1)
    time_elapsed = load_dict.get('t', 0.)
    metric_val_best = load_dict.get(
        'loss_val_best', -model_selection_sign * np.inf)

    print('Current best validation metric (%s): %.8f'
          % (model_selection_metric, metric_val_best))

    logger = SummaryWriter(os.path.join(out_dir, 'logs'))

    # Shorthands
    print_every = cfg['training']['print_every']
    checkpoint_every = cfg['training']['checkpoint_every']
    validate_every = cfg['training']['validate_every']
    visualize_every = cfg['training']['visualize_every']

    # Print model
    nparameters = sum(p.numel() for p in model.parameters())
    print(model)
    print('Total number of parameters: %d' % nparameters)

    if args.rtpt is not None:
        rtpt.start()

    while True:
        epoch_it += 1

        for batch in train_loader:
            it += 1

            # Save checkpoint
            if (checkpoint_every > 0 and (it % checkpoint_every) == 0 and
                it > 0 and trainer.bad_training_steps == 0):
                print('Saving checkpoint')
                checkpoint_io.save('model.pt', epoch_it=epoch_it, it=it,
                                   loss_val_best=metric_val_best, t=time_elapsed)

            # Backup if necessary
            if (backup_every > 0 and (it % backup_every) == 0):
                print('Backup checkpoint')
                checkpoint_io.save('model_%d.pt' % it, epoch_it=epoch_it, it=it,
                                   loss_val_best=metric_val_best, t=time_elapsed)

            # Visualize output
            if visnow or (it > 0 and visualize_every > 0 and (it % visualize_every) == 0):
                print('Visualizing')
                trainer.visualize(data_vis_val, mode='val')
                trainer.visualize(data_vis_train, mode='train')

            # Run validation
            if testnow or (it > 0 and validate_every > 0 and (it % validate_every) == 0):
                eval_dict = trainer.evaluate(val_loader, full_scale=args.full_scale)
                metric_val = eval_dict[model_selection_metric]
                print('Validation metric (%s): %.4f'
                      % (model_selection_metric, metric_val))

                for k, v in eval_dict.items():
                    logger.add_scalar('val/%s' % k, v, it)

                if model_selection_sign * (metric_val - metric_val_best) > 0:
                    metric_val_best = metric_val
                    print('New best model (loss %.4f)' % metric_val_best)
                    checkpoint_io.save('model_best.pt', epoch_it=epoch_it, it=it,
                                       loss_val_best=metric_val_best, t=time_elapsed)

            # Run Training step
            t0 = time.time()
            loss = trainer.train_step(batch, it)
            time_elapsed += time.time() - t0
            logger.add_scalar('train/loss', loss, it)

            # Print output
            if print_every > 0 and (it % print_every) == 0:
                print(out_dir, '[Epoch %02d] it=%03d, loss=%.4f'
                      % (epoch_it, it, loss))


            if it % lr_scheduler.update_every(it) == 0:
                new_lr = lr_scheduler.get_cur_lr(it)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr

            if args.rtpt is not None:
                rtpt.step(subtitle=f'score={metric_val_best:2.2f}')
            testnow = False
            visnow = False

            if it >= max_it:
                print('Iteration limit reached. Exiting.')
                checkpoint_io.save('model.pt', epoch_it=epoch_it, it=it,
                                   loss_val_best=metric_val_best, t=time_elapsed)
                exit(0)

