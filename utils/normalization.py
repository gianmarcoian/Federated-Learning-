import torch.nn as nn
import torch
import functools


def get_normalization(norm,num_classes, conditional=False):
  if conditional:
    if norm == 'InstanceNorm++':
      return functools.partial(ConditionalInstanceNorm2dPlus, num_classes=num_classes)
    else:
      raise NotImplementedError(f'{norm} not implemented yet.')
  else:
    if norm == 'GroupNorm':
      return nn.GroupNorm
    else:
      raise ValueError('Unknown normalization: %s' % norm)
  
  
class ConditionalInstanceNorm2dPlus(nn.Module):
  def __init__(self, num_features, num_classes, bias=True):
    super().__init__()
    self.num_features = num_features
    self.bias = bias
    self.instance_norm = nn.InstanceNorm2d(num_features, affine=False, track_running_stats=False)
    if bias:
      self.embed = nn.Embedding(num_classes, num_features * 3)
      self.embed.weight.data[:, :2 * num_features].normal_(1, 0.02)  # Initialise scale at N(1, 0.02)
      self.embed.weight.data[:, 2 * num_features:].zero_()  # Initialise bias at 0
    else:
      self.embed = nn.Embedding(num_classes, 2 * num_features)
      self.embed.weight.data.normal_(1, 0.02)

  def forward(self, x, y):
    means = torch.mean(x, dim=(2, 3))
    m = torch.mean(means, dim=-1, keepdim=True)
    v = torch.var(means, dim=-1, keepdim=True)
    means = (means - m) / (torch.sqrt(v + 1e-5))
    h = self.instance_norm(x)

    if self.bias:
      gamma, alpha, beta = self.embed(y).chunk(3, dim=-1)
      h = h + means[..., None, None] * alpha[..., None, None]
      out = gamma.view(-1, self.num_features, 1, 1) * h + beta.view(-1, self.num_features, 1, 1)
    else:
      gamma, alpha = self.embed(y).chunk(2, dim=-1)
      h = h + means[..., None, None] * alpha[..., None, None]
      out = gamma.view(-1, self.num_features, 1, 1) * h
    return out
