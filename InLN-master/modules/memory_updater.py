from torch import nn
import torch


class MemoryUpdater(nn.Module):
  def update_memory(self, unique_node_ids, unique_messages, timestamps):
    pass


class SequenceMemoryUpdater(MemoryUpdater):
  def __init__(self, memory, message_dimension, memory_dimension, device):
    super(SequenceMemoryUpdater, self).__init__()
    self.memory = memory
    self.layer_norm = torch.nn.LayerNorm(memory_dimension)
    self.message_dimension = message_dimension
    self.device = device

  def update_memory(self, unique_node_ids, unique_messages, timestamps):
    if len(unique_node_ids) <= 0:
      return

    # assert (self.memory.get_last_update(unique_node_ids) <= timestamps).all().item(), "Trying to " \
    #                                                                                  "update memory to time in the past"

    memory = self.memory.get_memory(unique_node_ids)
    self.memory.last_update[unique_node_ids] = timestamps

    #print(len(unique_node_ids),memory.shape) #138   torch.Size([138, 172])


    # gru
    #updated_memory = self.memory_updater(unique_messages, memory)
    # LINEAR
    updated_memory = self.memory_updater(unique_messages)

    #print(unique_messages.shape,updated_memory.shape)

    self.memory.set_memory(unique_node_ids, updated_memory)

  def get_updated_memory(self, unique_node_ids, unique_messages, timestamps):
    if len(unique_node_ids) <= 0:
      return self.memory.memory.data.clone(), self.memory.last_update.data.clone()

    # assert (self.memory.get_last_update(unique_node_ids) <= timestamps).all().item(), "Trying to " \
    #                                                                              "update memory to time in the past"

    #910 torch.Size([910, 518])  torch.Size([910])

    updated_memory = self.memory.memory.data.clone()

    #print(unique_messages.shape) 这个shape是可变的, torch.Size([993, 518])torch.Size([994, 518])

    #GRU
    #updated_memory[unique_node_ids] = self.memory_updater(unique_messages, updated_memory[unique_node_ids])
    # LINEAR
    updated_memory[unique_node_ids] = self.memory_updater(unique_messages)

    # 第一个参数 shape是 [batch,feature_len]，第二个是 [batch, hidden length]

    updated_last_update = self.memory.last_update.data.clone()
    updated_last_update[unique_node_ids] = timestamps

    #print(updated_memory.shape, unique_messages.shape)
    #torch.Size([1981, 172]) torch.Size([1981]) 始终是一样的


    return updated_memory, updated_last_update


class GRUMemoryUpdater(SequenceMemoryUpdater):
  def __init__(self, memory, message_dimension, memory_dimension, device):
    super(GRUMemoryUpdater, self).__init__(memory, message_dimension, memory_dimension, device)

    self.memory_updater = nn.GRUCell(input_size=message_dimension,
                                     hidden_size=memory_dimension)


class RNNMemoryUpdater(SequenceMemoryUpdater):
  def __init__(self, memory, message_dimension, memory_dimension, device):
    super(RNNMemoryUpdater, self).__init__(memory, message_dimension, memory_dimension, device)

    self.memory_updater = nn.RNNCell(input_size=message_dimension,
                                     hidden_size=memory_dimension)


class LinearMemoryUpdater(SequenceMemoryUpdater):
  def __init__(self, memory, message_dimension, memory_dimension, device):
    super(LinearMemoryUpdater, self).__init__(memory, message_dimension, memory_dimension, device)

    self.memory_updater = nn.Linear(message_dimension, memory_dimension)

def get_memory_updater(module_type, memory, message_dimension, memory_dimension, device):
  if module_type == "gru":
    return GRUMemoryUpdater(memory, message_dimension, memory_dimension, device)
  elif module_type == "rnn":
    return RNNMemoryUpdater(memory, message_dimension, memory_dimension, device)
  elif module_type == "linear":
    return LinearMemoryUpdater(memory, message_dimension, memory_dimension, device)
