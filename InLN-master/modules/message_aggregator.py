from collections import defaultdict
import torch
import numpy as np
from modules.ats.models_tgn import TransformerSeq

class MessageAggregator(torch.nn.Module):
  """
  Abstract class for the message aggregator module, which given a batch of node ids and
  corresponding messages, aggregates messages with the same node id.
  """
  def __init__(self, device):
    super(MessageAggregator, self).__init__()
    self.device = device

  def aggregate(self, node_ids, messages):
    """
    Given a list of node ids, and a list of messages of the same length, aggregate different
    messages for the same id using one of the possible strategies.
    :param node_ids: A list of node ids of length batch_size
    :param messages: A tensor of shape [batch_size, message_length]
    :param timestamps A tensor of shape [batch_size]
    :return: A tensor of shape [n_unique_node_ids, message_length] with the aggregated messages
    """

  def group_by_id(self, node_ids, messages, timestamps):
    node_id_to_messages = defaultdict(list)

    for i, node_id in enumerate(node_ids):
      node_id_to_messages[node_id].append((messages[i], timestamps[i]))

    return node_id_to_messages


class LastMessageAggregator(MessageAggregator):
  def __init__(self, device):
    super(LastMessageAggregator, self).__init__(device)

  def aggregate(self, node_ids, messages):
    """Only keep the last message for each node"""

    #print('----------')
    unique_node_ids = np.unique(node_ids)

    #print('unique_node_ids', len(unique_node_ids))

    unique_messages = []
    unique_timestamps = []

    to_update_node_ids = []

    for node_id in unique_node_ids:
      if len(messages[node_id]) > 0:
        to_update_node_ids.append(node_id)
        unique_messages.append(messages[node_id][-1][0])
        unique_timestamps.append(messages[node_id][-1][1])

    #print('ddd', len(unique_messages))

    unique_messages = torch.stack(unique_messages) if len(to_update_node_ids) > 0 else []

    unique_timestamps = torch.stack(unique_timestamps) if len(to_update_node_ids) > 0 else []

    # if len(to_update_node_ids) > 0:
    #   print(unique_messages.size(), unique_timestamps.size())

    return to_update_node_ids, unique_messages, unique_timestamps


class AttnMessageAggregator(MessageAggregator):
  def __init__(self, device):
    super(AttnMessageAggregator, self).__init__(device)

    #初始化一个Transformer
    adapt_io_params= {'adapt_io_enabled':False, 'adapt_io_tied':False, 'adapt_io_divval':4,'adapt_io_cutoffs':[20000, 40000, 200000]}
    adapt_span_params={'adapt_span_enabled':True,'adapt_span_loss':0,'adapt_span_ramp':32,'adapt_span_init':0,'adapt_span_cache':False}
    pers_mem_params={'pers_mem_size':0}
    self.device=device

    self.message_aggregator = TransformerSeq(
      vocab_size=10, # 输入序列长度
      hidden_size=400,
      nb_heads=2,
      nb_layers=1,
      attn_span= 10, # 最大注意力收集的范围
      emb_dropout=0,
      dropout=0.2,
      inner_hidden_size=512,
      adapt_io_params=adapt_io_params,
      adapt_span_params=adapt_span_params,
      pers_mem_params=pers_mem_params,
      ).to(device)

  def aggregate(self, node_ids, messages):
    """Only keep the last message for each node"""
    #print('----------')
    #print('node_ids',node_ids)
    #print('messages', messages[1125])
    # 一个message是一个长518的tensor
    vocab_size=10

    unique_node_ids = np.unique(node_ids) # 去掉重复id
    #print('unique_node_ids', len(unique_node_ids))
    unique_messages = []
    unique_timestamps = []

    to_update_node_ids = []


    X=[]
    pad_mask=[]
    message_len=[]

    for node_id in unique_node_ids:

      if len(messages[node_id]) > 0:

        to_update_node_ids.append(node_id)

        # messages[node_id,message embedding,timestamp]
        # 每一个节点的所有message: [m[0] for m in messages[node_id]]
        # batch数据预处理，生成长度相等的batch 和 响应的 pad mask 输入到message aggregator 中

        #temp=[m[0].cpu().numpy() for m in messages[node_id]]
        temp = [m[0] for m in messages[node_id]]
        #temp_mask = np.zeros(vocab_size,dtype=np.float32)
        temp_mask = torch.zeros(vocab_size).to(self.device)

        #补齐空位
        if len(temp)<vocab_size:
          message_len.append(len(temp) - 1)
          temp_mask[len(temp)-vocab_size:] = 1
          #temp = temp + [np.zeros(400)] * (vocab_size - len(temp)) #400是一个message的长度
          temp = temp + [torch.zeros(400).to(self.device)] * (vocab_size - len(temp))
        else:
          message_len.append(vocab_size - 1)

        #裁剪多出的message
        if len(temp)>vocab_size:
          X.append(torch.stack(temp[-vocab_size:]).unsqueeze(dim=0))
          #X.append(np.expand_dims(np.stack(temp[-vocab_size:]),0))

        else:
          X.append(torch.stack(temp).unsqueeze(dim=0))

          #X.append(np.expand_dims(np.stack(temp),0))

        pad_mask.append(temp_mask)
        unique_timestamps.append(messages[node_id][-1][1])

    if len(to_update_node_ids) > 0:
      X = torch.cat(X,dim=0).to(self.device)
      #X = np.concatenate(X,axis=0).astype(np.float32)
      #X = torch.from_numpy(X).to(self.device)

      pad_mask = torch.stack(pad_mask).unsqueeze(1).expand(-1,vocab_size,-1).to(self.device)
      #pad_mask = np.expand_dims(np.stack(pad_mask),1)
      #pad_mask = torch.from_numpy(pad_mask).expand(-1, vocab_size, -1).to(self.device)
      #进入aggregator
      #print(X.size(),pad_mask.size())
      attn_messages = self.message_aggregator(X, pad_mask)

      m=[]
      for i, ind  in enumerate(message_len):
        m.append(attn_messages[i,ind,:])

    #if len(to_update_node_ids) > 0:
      #print('ddd', len(m))
    unique_messages=torch.stack(m) if len(to_update_node_ids) > 0 else []
    unique_timestamps = torch.stack(unique_timestamps) if len(to_update_node_ids) > 0 else []
    #if len(to_update_node_ids) > 0:
      #print(unique_messages.size(),unique_timestamps.size())
    return to_update_node_ids, unique_messages, unique_timestamps




class MeanMessageAggregator(MessageAggregator):
  def __init__(self, device):
    super(MeanMessageAggregator, self).__init__(device)

  def aggregate(self, node_ids, messages):
    """Only keep the last message for each node"""
    unique_node_ids = np.unique(node_ids)
    unique_messages = []
    unique_timestamps = []

    to_update_node_ids = []
    n_messages = 0

    for node_id in unique_node_ids:
      if len(messages[node_id]) > 0:
        n_messages += len(messages[node_id])
        to_update_node_ids.append(node_id)
        unique_messages.append(torch.mean(torch.stack([m[0] for m in messages[node_id]]), dim=0))

        #print(torch.stack([m[0] for m in messages[node_id]]).shape) torch.Size([4, 518]) 四个历史message, 每个518维
        unique_timestamps.append(messages[node_id][-1][1])

    unique_messages = torch.stack(unique_messages) if len(to_update_node_ids) > 0 else []
    unique_timestamps = torch.stack(unique_timestamps) if len(to_update_node_ids) > 0 else []


    return to_update_node_ids, unique_messages, unique_timestamps






def get_message_aggregator(aggregator_type, device):
  if aggregator_type == "last":
    return LastMessageAggregator(device=device)
  elif aggregator_type == "mean":
    return MeanMessageAggregator(device=device)
  elif aggregator_type == "attn":
    return AttnMessageAggregator(device=device)
  else:
    raise ValueError("Message aggregator {} not implemented".format(aggregator_type))
