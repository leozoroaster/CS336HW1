import torch
import math
from collections.abc import Callable
from typing import Optional

def cross_entropy(o, target):
    m, _ = torch.max(o, dim=-1, keepdim=True)
    o = o - m
    new_o = torch.exp(o)
    s = torch.sum(new_o, dim=-1, keepdim=True)
    log_softmax = o - torch.log(s)

    return -log_softmax.gather(-1, target.unsqueeze(-1)).squeeze(-1)

def cosine_annealing_schedule(t,a_max,a_min,t_w,t_c):
  if t<t_w:
    return t*a_max/t_w
  elif t_w<=t<=t_c:
    return a_min+0.5*(a_max-a_min)*(1+math.cos(math.pi*(t-t_w)/(t_c-t_w)))
  else:
    return a_min

class AdamW(torch.optim.Optimizer):
  def __init__(self, params, decay, lr=1e-3, beta_1=0.9, beta_2=0.95, eps=1e-8):
    if lr < 0:
      raise ValueError(f"Invalid learning rate: {lr}")
    defaults = {"lr": lr, "decay": decay, "beta_1": beta_1, "beta_2": beta_2, "eps": eps}
    super().__init__(params, defaults)

  def step(self, closure: Optional[Callable] = None, scheduler=None):
    loss = None if closure is None else closure()
    for group in self.param_groups:
      lr = group["lr"] # Get the learning rate
      new_lr = lr
      if scheduler is not None:
          epoch, a_max, a_min, t_w, t_c = scheduler
          new_lr = cosine_annealing_schedule(epoch, a_max, a_min, t_w, t_c)
      decay=group["decay"]
      beta_1=group["beta_1"]
      beta_2=group["beta_2"]
      eps=group["eps"]
      for p in group["params"]:
        if p.grad is None:
          continue
        state = self.state[p] # Get state associated with p.
        if len(state) == 0:
          state["t"] = 0
          state["m"] = torch.zeros_like(p)
          state["v"] = torch.zeros_like(p)

        grad = p.grad # Get the gradient of loss with respect to p.
        t=state["t"]
        m=state["m"]
        v=state["v"]
        m.mul_(beta_1).add_(grad, alpha=1 - beta_1)
        v.mul_(beta_2).addcmul_(grad, grad, value=1 - beta_2)
        denominator = v.sqrt().add_(eps)
        state["t"] = t + 1

        adjusted_lr=new_lr*math.sqrt(1-beta_2**(t+1))/(1-beta_1**(t+1))
        with torch.no_grad():
          p.addcdiv_(m, denominator, value=-adjusted_lr)
          p.add_(p, alpha=-new_lr * decay)
    return loss

def grad_clip(g,m,eps=1e-6):
  l2_norm=torch.sqrt(torch.sum(g**2))
  if l2_norm<=m:
    return g
  else:
    return g*m/(l2_norm+eps)