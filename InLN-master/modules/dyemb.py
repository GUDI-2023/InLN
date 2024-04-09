import torch
from torch import nn

from collections import defaultdict
from copy import deepcopy


class Dyemb(nn.Module):

  def __init__(self, n_nodes, Dyemb_dimension, input_dimension, raw_feature, message_dimension=None,
               device="cpu", combination_method='sum'):
    super(Dyemb, self).__init__()
    self.n_nodes = n_nodes # item 的数量
    self.Dyemb_dimension = Dyemb_dimension
    self.input_dimension = input_dimension
    self.message_dimension = message_dimension
    self.device = device
    self.raw_feature=raw_feature
    self.Dyemb = raw_feature
    print(self.Dyemb.size())
    self.combination_method = combination_method

    self.__init_Dyemb__()

  def __init_Dyemb__(self):
    """
    Initializes the Dyemb to all zeros. It should be called at the start of each epoch.
    """
    # Treat Dyemb as parameter so that it is saved and loaded together with the model
    # self.Dyemb = nn.Parameter(torch.zeros((self.n_nodes, self.Dyemb_dimension)).to(self.device),
    #                            requires_grad=False)
    self.last_update = nn.Parameter(torch.zeros(self.n_nodes).to(self.device),
                                    requires_grad=False)

    self.messages = defaultdict(list)


  def get_Dyemb(self, node_idxs):
    return self.Dyemb[node_idxs, :]

  def set_Dyemb(self, node_idxs, values):
    self.Dyemb[node_idxs, :] = values
  def reset_Dyemb(self):
    self.Dyemb = self.raw_feature


  def get_last_update(self, node_idxs):
    return self.last_update[node_idxs]

  def backup_Dyemb(self):
    messages_clone = {}
    for k, v in self.messages.items():
      messages_clone[k] = [(x[0].clone(), x[1].clone()) for x in v]

    return self.Dyemb.data.clone(), self.last_update.data.clone(), messages_clone

  def restore_Dyemb(self, Dyemb_backup):
    self.Dyemb.data, self.last_update.data = Dyemb_backup[0].clone(), Dyemb_backup[1].clone()

    self.messages = defaultdict(list)
    for k, v in Dyemb_backup[2].items():
      self.messages[k] = [(x[0].clone(), x[1].clone()) for x in v]
