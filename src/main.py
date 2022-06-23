from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os

import torch
import torch.utils.data

from opts import opts
from model.model import create_model, freeze_layers, load_model, save_model, merge_models
from logger import Logger
from dataset.dataset_factory import get_dataset
from trainer import Trainer
import json
from utils.eval_frustum import EvalFrustum


def get_optimizer(opt, model):
  if opt.optim == 'adam':
    print('Using adam.')
    optimizer = torch.optim.Adam(model.parameters(), opt.lr)
  elif opt.optim == 'sgd':
    print('Using SGD')
    optimizer = torch.optim.SGD(
      model.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay) 
  else:
    raise ValueError("Optimizer not implemented yet.")
  return optimizer

def main(opt):
  torch.manual_seed(opt.seed)
  torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.eval
  Dataset = get_dataset(opt.dataset)
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
  print('-'*100, '\nOPTIONS:')
  print(opt, '\n', '-'*100)
  if not opt.not_set_cuda_env:
    # Not training on GPU cluster
    print('Setting CUDA_VISIBLE_DEVICES.')
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str   # e.g. "0,1"
  opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
  logger = Logger(opt)

  # Get and print information about current GPU setup
  if torch.cuda.is_available():
    print('CUDA is available.') 
    num_gpus = torch.cuda.device_count()
    print(f'{num_gpus} GPU(s) were found.')
    vis_dev = os.environ.get('CUDA_VISIBLE_DEVICES')
    print(f'CUDA_VISIBLE_DEVICES is set to: {vis_dev}')
    for idx in range(num_gpus):
      print(f'Properties of GPU with index {idx}:\n{torch.cuda.get_device_properties(idx)}')
  else:
    print('CUDA is NOT available.')

  if opt.merge_models:
    template_model = create_model(opt.arch, opt.heads, opt.head_conv, opt=opt)
    model, epoch = merge_models(template_model, opt.load_model, opt.load_model2, opt)
    optimizer = get_optimizer(opt, model)
    print('Merged model is saved in:', os.path.join(opt.log_dir, f'model_{epoch}.pth'))
    save_model(os.path.join(opt.log_dir, f'model_{epoch}.pth'), 
                  epoch, model, optimizer)
  else:
    print('Creating model...')
    model = create_model(opt.arch, opt.heads, opt.head_conv, opt=opt)
    optimizer = get_optimizer(opt, model)
    start_epoch = 0
    lr = opt.lr

    if opt.load_model != '':
      model, optimizer, start_epoch = load_model(
        model, opt.load_model, opt, optimizer)
    
    if opt.freeze_layers:
      model = freeze_layers(model, opt)

    if opt.set_eval_layers:
      for layer in opt.layers_to_eval:
          print('Layer set to eval()-mode in training: ', layer)

    trainer = Trainer(opt, model, optimizer)
    trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)

    # Create object of class EvalFrustum if required
    if opt.eval_frustum > 0:
      eval_frustum = EvalFrustum(opt)
    else:
      eval_frustum = None

    if opt.val_intervals <= opt.num_epochs or opt.eval:
      print('Setting up validation data...')
      # Set the batchsize to the number of GPUs used
      val_loader = torch.utils.data.DataLoader(
        Dataset(opt=opt, split=opt.val_split, eval_frustum=eval_frustum), batch_size=1, shuffle=False, 
                num_workers=opt.num_workers, pin_memory=True)

      if opt.eval:
        # If eval is active then just validate and not train
        _, preds = trainer.val(0, val_loader)
        val_loader.dataset.run_eval(preds, opt.log_dir, n_plots=opt.eval_n_plots, 
                                    render_curves=opt.eval_render_curves)
        return

    print('Setting up training data...')
    train_loader = torch.utils.data.DataLoader(
        Dataset(opt=opt, split=opt.train_split, eval_frustum=eval_frustum), batch_size=opt.batch_size, 
          shuffle=opt.shuffle_train, num_workers=opt.num_workers, 
          pin_memory=True, drop_last=True
    )

    print('Starting training...')
    
    # Loop over epochs
    for epoch in range(start_epoch + 1, opt.num_epochs + 1):
      mark = epoch if opt.save_all else 'last'

      # log learning rate
      for param_group in optimizer.param_groups:
        lr = param_group['lr']
        logger.scalar_summary('LR', lr, epoch)
        break
      
      # train one epoch
      log_dict_train, _ = trainer.train(epoch, train_loader, eval_frustum=eval_frustum)
      logger.write('epoch: {} |'.format(epoch))
      
      # log train results
      for k, v in log_dict_train.items():
        logger.scalar_summary('train_{}'.format(k), v, epoch)
        logger.write('{} {:8f} | '.format(k, v))
      
      # Save model as "model_last.pth or model_epoch when opt.save_all"
      save_model(os.path.join(opt.log_dir, f'model_{mark}.pth'), 
                  epoch, model, optimizer)

      # evaluate
      if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
        # Validate
        with torch.no_grad():
          # Validate current epoch, create validation summary as .json
          log_dict_val, preds = trainer.val(epoch, val_loader)
          
          # evaluate val set using dataset-specific evaluator
          if opt.run_dataset_eval:
            out_dir = val_loader.dataset.run_eval(preds, opt.log_dir, 
                                                  n_plots=opt.eval_n_plots, 
                                                  render_curves=opt.eval_render_curves
                                                  )

            # log dataset-specific evaluation metrics
            with open('{}metrics_summary.json'.format(out_dir), 'r') as f:
              metrics = json.load(f)
            logger.scalar_summary('AP/overall', metrics['mean_ap']*100.0, epoch)
            for k,v in metrics['mean_dist_aps'].items():
              logger.scalar_summary('AP/{}'.format(k), v*100.0, epoch)
            for k,v in metrics['tp_errors'].items():
              logger.scalar_summary('Scores/{}'.format(k), v, epoch)
            logger.scalar_summary('Scores/NDS', metrics['nd_score'], epoch)
            
        # log eval results
        for k, v in log_dict_val.items():
          logger.scalar_summary('val_{}'.format(k), v, epoch)
          logger.write('{} {:8f} | '.format(k, v))
      
      logger.write('\n')
      # Save the model if wanted
      if epoch in opt.save_point:
        save_model(os.path.join(opt.log_dir, f'model_{epoch}.pth'), 
                  epoch, model, optimizer)
      
      # update learning rate
      if epoch in opt.lr_step:
        # Drop lr by a factor of opt.lr_step_factor at every epoch listed in list lr_step
        lr = opt.lr * (opt.lr_step_factor ** (opt.lr_step.index(epoch) + 1)) 
        print('Drop LR to', lr)
        # save lr into optimizer
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

      # If frustum evaluation is required, save/plt the histograms
      if opt.eval_frustum > 0:
        eval_frustum.print()
        if opt.eval_frustum < 4:
          eval_frustum.plot_histograms(save_plots=False)
        if opt.eval_frustum == 4:
          eval_frustum.plot_histograms(save_plots=True)
        if opt.eval_frustum == 5:
          # Dump snapshot evaluation using pickle
          eval_frustum.dump_snapshot_eval()

    logger.close()

if __name__ == '__main__':
  opt = opts().parse()
  torch.cuda.empty_cache()
  main(opt)
