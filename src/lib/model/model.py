from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from multiprocessing.sharedctypes import Value

import torchvision.models as models
import torch
import torch.nn as nn
import os

from .networks.dla import DLASeg
from .networks.resdcn import PoseResDCN
from .networks.resnet import PoseResNet
from .networks.dlav0 import DLASegv0
from .networks.generic_network import GenericNetwork

_network_factory = {
  'resdcn': PoseResDCN,
  'dla': DLASeg,
  'res': PoseResNet,
  'dlav0': DLASegv0,
  'generic': GenericNetwork
}

def create_model(arch, head, head_conv, opt=None):
  # Deconstruct architecture notation with layers e.g. dla_34 -> network DLA with 34 Layers
  num_layers = int(arch[arch.find('_') + 1:]) if '_' in arch else 0
  arch = arch[:arch.find('_')] if '_' in arch else arch
  # Get import corresponding to the chosen architecture
  model_class = _network_factory[arch]
  # Create model
  model = model_class(num_layers, heads=head, head_convs=head_conv, opt=opt)  # init the class of architecture
  return model

def load_model(model, model_path, opt, optimizer=None):
  with torch.no_grad():
    start_epoch = 0
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    print('Loaded in model: {}, starting at epoch {}'.format(model_path, checkpoint['epoch']))
    state_dict_ = checkpoint['state_dict']
    state_dict = {}
    
    # convert data_parallel to model
    for k in state_dict_:
      if k.startswith('module') and not k.startswith('module_list'):
        state_dict[k[7:]] = state_dict_[k] # remove not important stuff
      else:
        state_dict[k] = state_dict_[k]
    model_state_dict = model.state_dict() # create default parameters for chosen network architecture (get all number and shape of all needed parameters)

    # check loaded parameters and created model parameters
    for k in state_dict:
      if k in model_state_dict:
        # Weights have shape [in_channels, out_channels, kernel_dim_1, kernel_dim_2]
        # Biases have shape [in_channels]
        if (state_dict[k].shape != model_state_dict[k].shape) or \
          (opt.reset_hm and k.startswith('hm') and (state_dict[k].shape[0] in [80, 1])):
          if opt.reuse_hm:
            print('Reusing parameter {}, required shape{}, '\
                  'loaded shape{}.'.format(
              k, model_state_dict[k].shape, state_dict[k].shape))
            if state_dict[k].shape[0] < model_state_dict[k].shape[0]:   # there are less parameters (in_channels) loaded in than required for this state
              model_state_dict[k][:state_dict[k].shape[0]] = state_dict[k] 
            else:   # there are more parameters (in_channels) loaded in than required for this state
              model_state_dict[k] = state_dict[k][:model_state_dict[k].shape[0]]
            state_dict[k] = model_state_dict[k] # overwrite loaded model
          elif opt.warm_start_weights:
            try:
              print('Partially loading parameter {}, required shape{}, '\
                    'loaded shape{}.'.format(
                k, model_state_dict[k].shape, state_dict[k].shape))
              if state_dict[k].shape[1] < model_state_dict[k].shape[1]: # there are less parameters (out_channels) loaded in than required for this state
                model_state_dict[k][:,:state_dict[k].shape[1]] = state_dict[k]
              else:   # there are more parameters (out_channels) loaded in than required for this state
                model_state_dict[k] = state_dict[k][:,:model_state_dict[k].shape[1]]
              state_dict[k] = model_state_dict[k] # overwrite loaded model
            except:
              print('Skip (!) loading parameter {}, required shape{}, '\
                  'loaded shape{}.'.format(
                  k, model_state_dict[k].shape, state_dict[k].shape))
              state_dict[k] = model_state_dict[k] # overwrite loaded model
    
          else:
            print('Skip (!) loading parameter {}, required shape{}, '\
                  'loaded shape{}.'.format(
              k, model_state_dict[k].shape, state_dict[k].shape))
            state_dict[k] = model_state_dict[k] # overwrite loaded model
        
        else:
          # If dims match don't need to correct anything
          continue # line not needed only for comprehensiveness
      
      else:
        # State loaded in but not needed in chosen network architecture
        print('Drop parameter {}.'.format(k))

    for k in model_state_dict:
      if not (k in state_dict):
        # State not loaded in but needed in chosen network architecture
        print('No param {}.'.format(k))
        state_dict[k] = model_state_dict[k] # add default values to loaded model
    model.load_state_dict(state_dict, strict=False) # use updated loaded model to set all available and usable states and its parameters

  # resume optimizer parameters
  if optimizer is not None and opt.resume:
    if 'optimizer' in checkpoint:
      start_epoch = checkpoint['epoch']
      start_lr = opt.lr
      # Drop lr by a factor of 1e-1 at every epoch x,y,z listed in lr_step [x,y,z]
      for step in opt.lr_step:
        if start_epoch >= step:
          start_lr *= opt.lr_step_factor
      # save into optimizer
      for param_group in optimizer.param_groups:
        param_group['lr'] = start_lr
      print('Resumed optimizer with start lr: ', start_lr)
    else:
      print('No optimizer parameters in checkpoint.')
      
  if optimizer is not None:
    return model, optimizer, start_epoch
  else:
    return model

def freeze_layers(model, opt):
  """ 
  Freeze layers in network specified in opts.
  """
  for (name, module) in model.named_children():
    if name in opt.layers_to_freeze:
      print('Layerblock frozen: ', name)
      for (name, layer) in module.named_children():
        for param in layer.parameters():
          param.requires_grad = False
  return model

def eval_layers(model, opt):
  """ 
  Set layers to eval mode specified in opts.
  """
  for (name, module) in model.named_children():
    if name in opt.layers_to_eval:
      module.eval()
  return model

def save_model(path, epoch, model, optimizer=None):
  if isinstance(model, torch.nn.DataParallel):
    state_dict = model.module.state_dict()
  else:
    state_dict = model.state_dict()
  data = {'epoch': epoch,
          'state_dict': state_dict}
  if not (optimizer is None):
    data['optimizer'] = optimizer.state_dict()
  torch.save(data, path)


def compare_models(model1, model2):
  ## Compare models
  for (name_module, module) in model1.named_children():
    for (name_layer, layer) in module.named_children():
      for (name_param, param) in layer.named_parameters():
        try:
          param2 = model2.__getattr__(name_module).__getattr__(name_layer).__getattr__(name_param)
        except:
          try:
            param2 = model2.__getattr__(name_module).__getattr__(name_layer).__getattr__(name_param.split(".")[0]).__getattr__(name_param.split(".")[1]).__getattr__(name_param.split(".")[2]).__getattr__(name_param.split(".")[3])
          except:
            try:
              param2 = model2.__getattr__(name_module).__getattr__(name_layer).__getattr__(name_param.split(".")[0]).__getattr__(name_param.split(".")[1]).__getattr__(name_param.split(".")[2])
            except:
              param2 = model2.__getattr__(name_module).__getattr__(name_layer).__getattr__(name_param.split(".")[0]).__getattr__(name_param.split(".")[1])
        if param.equal(param2):
          print("Checked:", name_param)
          continue
        elif not name_module.__contains__("lfa"):
          raise ValueError(f"{name_module} of model1 doesnt match model2 parameter.")


def merge_models(merged_model, model1, model2, opt):
  """
  Merge models 
  - model1: Backbone + prim heads + sec heads
  - model2: LFANet
  """
  import copy
  with torch.no_grad():
    checkpoint1 = torch.load(model1, map_location=lambda storage, loc: storage)
    print('Loaded in model1: {}, starting at epoch {}'.format(model1, checkpoint1['epoch']))
    state_dict_1 = checkpoint1['state_dict']
    state_dict1 = {}

    # convert data_parallel to model
    for k in state_dict_1:
      if k.startswith('module') and not k.startswith('module_list'):
        state_dict1[k[7:]] = state_dict_1[k] # remove not important stuff
      else:
        state_dict1[k] = state_dict_1[k]
    
    checkpoint2 = torch.load(model2, map_location=lambda storage, loc: storage)
    print('Loaded in model2: {}, starting at epoch {}'.format(model2, checkpoint2['epoch']))
    state_dict_2 = checkpoint2['state_dict']
    state_dict2 = {}

    # convert data_parallel to model
    for k in state_dict_2:
      if k.startswith('module') and not k.startswith('module_list'):
        state_dict2[k[7:]] = state_dict_2[k] # remove not important stuff
      else:
        state_dict2[k] = state_dict_2[k]
    
    
    model_state_dict = merged_model.state_dict()
    # Check that model1 has the same structure as template model except of lfa stuff
    # ONLY CHECKING FOR LEARNABLE PARAMETERS!
    for (name_module, module) in merged_model.named_children():
      for (name_layer, layer) in module.named_children():
        name = name_module + '.' + name_layer
        if len(list(layer.named_children())) == 0:
          tmp = copy.deepcopy(layer._parameters)
          if layer._buffers:
            tmp.update(layer._buffers)
          for p in tmp:
            name_p = name + '.' + p
            print(f'{name_p} checked')
            if name_p not in state_dict1 and not name_p.__contains__("lfa"):
              if name_p in ['ida_up.up_1.bias','ida_up.up_2.bias']:
                print(f'{name_p} is not in model1! But this is an exception')
              else:
                raise ValueError(f"Parameter {name_p} not in model1!")
        else:
          for (name_param, param) in layer.named_children():
            name_Param = name + '.' + name_param
            if len(list(param.named_children())) == 0:
              tmp = copy.deepcopy(param._parameters)
              if param._buffers:
                tmp.update(param._buffers)
              for p in tmp:
                name_p = name_Param + '.' + p
                print(f'{name_p} checked')
                if name_p not in state_dict1 and not name_p.__contains__("lfa"):
                  if (name_p.__contains__('base') or name_p.__contains__('dla_up')) and name_p.__contains__('bias'):
                    print(f'{name_p} is not in model1! But this is an exception')
                  else:
                     raise ValueError(f"Parameter {name_p} not in model1!")
            else:
              for (name_attr, attr) in param.named_children():
                name_Atrr = name_Param + '.' + name_attr
                if len(list(attr.named_children())) == 0:
                  tmp = copy.deepcopy(attr._parameters)
                  if attr._buffers:
                    tmp.update(attr._buffers)
                  for p in tmp:
                    name_p = name_Atrr + '.' + p
                    print(f'{name_p} checked')
                    if name_p not in state_dict1 and not name_p.__contains__("lfa"):
                      if name_p.__contains__('base') and name_p.__contains__('bias'):
                        print(f'{name_p} is not in model1! But this is an exception')
                      else:
                        raise ValueError(f"Parameter {name_p} not in model1!")
                else:
                  for (name_smth, smth) in attr.named_children():
                    name_Smth = name_Atrr + '.' + name_smth
                    if len(list(smth.named_children())) == 0:
                      tmp = copy.deepcopy(smth._parameters)
                      if smth._buffers:
                        tmp.update(smth._buffers)
                      for p in tmp:
                        name_p = name_Smth + '.' + p
                        print(f'{name_p} checked')
                        if name_p not in state_dict1 and not name_p.__contains__("lfa"):
                          if name_p.__contains__('base') and name_p.__contains__('bias'):
                            print(f'{name_p} is not in model1! But this is an exception')
                          else:
                            raise ValueError(f"Parameter {name_p} not in model1!")
                    else:
                      for (name_smth2, smth2) in smth.named_children():
                        name_Smth2 = name_Smth + '.' + name_smth2
                        if len(list(smth2.named_children())) == 0:
                          tmp = copy.deepcopy(smth2._parameters)
                          if smth2._buffers:
                            tmp.update(smth2._buffers)
                          for p in tmp:
                            name_p = name_Smth2 + '.' + p
                            print(f'{name_p} checked')
                            if name_p not in state_dict1 and not name_p.__contains__("lfa"):
                              raise ValueError(f"Parameter {name_p} not in model1!")
                  
    print('='*50)
    _ = load_model(merged_model, model1,opt)
    _ = load_model(merged_model, model2,opt)
    print('='*50)
    # Idea: set template model in a way that it fits backbone, prim, sec heads from model1 and lfanet from model2
    for k in state_dict1:
      if k not in model_state_dict and not k.__contains__('lfa'):
        # raise ValueError(f'Model {k} not in template model!')
        print(f'Model {k} not in template model!')
      if k in model_state_dict and not k.__contains__('lfa'):
        print(f'Merged model state {k} is overwritten by model1')
        model_state_dict[k] = state_dict1[k]
      
    print('-'*50)
    for k in state_dict2:
      if k not in model_state_dict and k.__contains__("lfa"):
        raise ValueError(f'Model {k} not in template model!')
      if k.__contains__("lfa"):
        print(f'Merged model state {k} is overwritten by model2')
        model_state_dict[k] = state_dict2[k]
      
    for k in model_state_dict:
      if not (k in state_dict1 or k in state_dict2):
        raise ValueError(f'Template model has extra state {k}') 

    merged_model.load_state_dict(model_state_dict, strict=True)
    merged_epochs = checkpoint1['epoch'] + checkpoint2['epoch']
    print(f"Merged model is:\n{merged_model}")
    print(f"Combined epoch is: {merged_epochs}")

    return merged_model, merged_epochs

