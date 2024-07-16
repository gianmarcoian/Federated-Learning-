import abc
import torch
import numpy as np


class SDE(abc.ABC):

  def __init__(self, N):

    super().__init__()
    self.N = N

  @property
  @abc.abstractmethod
  def T(self):
    pass

  @abc.abstractmethod
  def sde(self, x, t):
    pass

  @abc.abstractmethod
  def marginal_prob(self, x, t):
    pass

  @abc.abstractmethod
  def prior_sampling(self, shape):
    pass

  @abc.abstractmethod
  def prior_logp(self, z):

    pass

  def discretize(self, x, t):

    dt = 1 / self.N
    drift, diffusion = self.sde(x, t)
    f = drift * dt
    G = diffusion * torch.sqrt(torch.tensor(dt, device=t.device))
    return f, G

  def reverse(self, score_fn, probability_flow=False):

    N = self.N
    T = self.T
    sde_fn = self.sde
    discretize_fn = self.discretize
