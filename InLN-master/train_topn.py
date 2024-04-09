import math
import logging
import time
import sys

import argparse
import torch
import numpy as np
import pickle
from pathlib import Path
import pandas as pd
import os
from evaluation.evaluation import eval_edge_prediction
from evaluation.topn import eval_edge_prediction_2
from model.InLN import InLN
from utils.utils import EarlyStopMonitor, RandEdgeSampler, get_neighbor_finder
from utils.data_processing import get_data, compute_time_statistics, get_kg_data, get_incre_data,get_train_data

torch.manual_seed(0)
np.random.seed(0)

### Argument and global variables
parser = argparse.ArgumentParser('TGN self-supervised training')
parser.add_argument('-d', '--data', type=str, help='Dataset name (eg. wikipedia or reddit)',
                    default='wikipedia')
parser.add_argument('--bs', type=int, default=400, help='Batch_size') #文中说了200兼具速度与准确
parser.add_argument('--prefix', type=str, default='', help='Prefix to name the checkpoints')
parser.add_argument('--n_degree', type=int, default=10, help='Number of neighbors to sample') #most recent 或者uniform
parser.add_argument('--n_head', type=int, default=2, help='Number of heads used in attention layer')
parser.add_argument('--n_epoch', type=int, default=8, help='Number of epochs')
parser.add_argument('--n_layer', type=int, default=1, help='Number of network layers') # 1-hop
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--patience', type=int, default=9, help='Patience for early stopping')
parser.add_argument('--n_runs', type=int, default=5, help='Number of runs')
parser.add_argument('--drop_out', type=float, default=0.2, help='Dropout probability')
parser.add_argument('--gpu', type=int, default=0, help='Idx for the gpu to use')
parser.add_argument('--node_dim', type=int, default=100, help='Dimensions of the node embedding')
parser.add_argument('--time_dim', type=int, default=100, help='Dimensions of the time embedding')
parser.add_argument('--backprop_every', type=int, default=1, help='Every how many batches to '
                                                                  'backprop')
parser.add_argument('--use_memory', action='store_true',
                    help='Whether to augment the model with a node memory')
parser.add_argument('--embedding_module', type=str, default="graph_attention", choices=[
  "graph_attention", "graph_sum", "identity", "time"], help='Type of embedding module')
parser.add_argument('--message_function', type=str, default="identity", choices=[
  "mlp", "identity"], help='Type of message function')
parser.add_argument('--memory_updater', type=str, default="linear", choices=[
  "gru", "rnn","linear"], help='Type of memory updater')
parser.add_argument('--aggregator', type=str, default="attn", help='Type of message '
                                                                        'aggregator')
parser.add_argument('--memory_update_at_end', action='store_true',
                    help='Whether to update memory at the end or at the start of the batch')
parser.add_argument('--message_dim', type=int, default=100, help='Dimensions of the messages')
parser.add_argument('--memory_dim', type=int, default=100, help='Dimensions of the memory for '
                                                                'each user') # 172是memory的size
parser.add_argument('--different_new_nodes', action='store_true',
                    help='Whether to use disjoint set of new nodes for train and val')
parser.add_argument('--uniform', action='store_true',
                    help='take uniform sampling from temporal neighbors')
parser.add_argument('--randomize_features', action='store_true',
                    help='Whether to randomize node features')
parser.add_argument('--use_destination_embedding_in_message', action='store_true',
                    help='Whether to use the embedding of the destination node as part of the message')
parser.add_argument('--use_source_embedding_in_message', action='store_true',
                    help='Whether to use the embedding of the source node as part of the message')
parser.add_argument('--dyrep', action='store_true',
                    help='Whether to run the dyrep model')


try:
  args = parser.parse_args()
except:
  parser.print_help()
  sys.exit(0)

BATCH_SIZE = args.bs
NUM_NEIGHBORS = args.n_degree
#NUM_NEG = 1
NUM_EPOCH = args.n_epoch
NUM_HEADS = args.n_head
DROP_OUT = args.drop_out
GPU = args.gpu


DATA = args.data
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr
NODE_DIM = args.node_dim
TIME_DIM = args.time_dim
USE_MEMORY = args.use_memory
MESSAGE_DIM = args.message_dim
MEMORY_DIM = args.memory_dim

Path("./saved_models/").mkdir(parents=True, exist_ok=True)
Path("./saved_checkpoints/").mkdir(parents=True, exist_ok=True)
MODEL_SAVE_PATH = f'./saved_models/{args.prefix}-{args.data}.pth'
get_checkpoint_path = lambda \
    epoch: f'./saved_checkpoints/{args.prefix}-{args.data}-{epoch}.pth'

### set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
Path("log/").mkdir(parents=True, exist_ok=True)
fh = logging.FileHandler('log/{}.log'.format(str(time.time())))
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.WARN)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info(args)

### Extract data for training, validation and testing
node_features, edge_features, full_data, _, val_data, test_data, new_node_val_data, \
new_node_test_data,n_items,n_users = get_data(DATA,
                              different_new_nodes_between_val_and_test=args.different_new_nodes, randomize_features=args.randomize_features)


train_data= get_train_data(DATA)

## get inremental data
incre_data = get_incre_data(DATA)

#导入kg数据
thg = get_kg_data(DATA)

#排开kg 的 edgeindex 和 interation data的node index

# Initialize training neighbor finder to retrieve temporal graph

args.uniform = False
train_ngh_finder = get_neighbor_finder(train_data, thg, args.uniform)

#train_ngh_finder = get_neighbor_finder(train_data, train_data ,args.uniform)

# Initialize validation and test neighbor finder to retrieve temporal graph
full_ngh_finder = get_neighbor_finder(full_data, thg ,args.uniform)
#full_ngh_finder = get_neighbor_finder(full_data, full_data ,args.uniform)

print(" Finish TH graph building")

# Initialize negative samplers. Set seeds for validation and testing so negatives are the same
# across different runs
# NB: in the inductive setting, negatives are sampled only amongst other new nodes
train_rand_sampler = RandEdgeSampler(train_data.sources, train_data.destinations) #Train_data只有interaction数据
val_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=0)
nn_val_rand_sampler = RandEdgeSampler(new_node_val_data.sources, new_node_val_data.destinations,
                                      seed=1)
test_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=2)
nn_test_rand_sampler = RandEdgeSampler(new_node_test_data.sources,
                                       new_node_test_data.destinations,
                                       seed=3)



# Set device
MSELoss = torch.nn.MSELoss()
device_string = 'cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu'
device = torch.device(device_string)

# Compute time statistics
mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = \
  compute_time_statistics(full_data.sources, full_data.destinations, full_data.timestamps)

for i in range(args.n_runs):
  results_path = "results/{}_{}.pkl".format(args.prefix, i) if i > 0 else "results/{}.pkl".format(args.prefix)
  Path("results/").mkdir(parents=True, exist_ok=True)

  # Initialize Model
  InLN = InLN(neighbor_finder=train_ngh_finder, node_features=node_features,
            edge_features=edge_features, device=device,
            n_layers=NUM_LAYER,
            n_heads=NUM_HEADS, dropout=DROP_OUT, use_memory=USE_MEMORY,
            message_dimension=MESSAGE_DIM, memory_dimension=MEMORY_DIM,
            #memory_update_at_start=not args.memory_update_at_end,
            memory_update_at_start= True,
            embedding_module_type=args.embedding_module,
            message_function=args.message_function,
            aggregator_type=args.aggregator,
            #aggregator_type='last', #设置
            memory_updater_type=args.memory_updater,
            n_neighbors=NUM_NEIGHBORS,
            mean_time_shift_src=mean_time_shift_src, std_time_shift_src=std_time_shift_src,
            mean_time_shift_dst=mean_time_shift_dst, std_time_shift_dst=std_time_shift_dst,
            use_destination_embedding_in_message=args.use_destination_embedding_in_message,
            use_source_embedding_in_message=args.use_source_embedding_in_message,
            dyrep=args.dyrep, n_items=n_items,n_users=n_users)


  criterion = torch.nn.BCELoss()
  optimizer = torch.optim.Adam(tgn.parameters(), lr=LEARNING_RATE)
  InLN = InLN.to(device)

  num_instance = len(train_data.sources)
  num_batch = math.ceil(num_instance / BATCH_SIZE)

  logger.info('num of training instances: {}'.format(num_instance))
  logger.info('num of batches per epoch: {}'.format(num_batch))
  idx_list = np.arange(num_instance)

  new_nodes_val_aps = []
  val_aps = []
  epoch_times = []
  total_epoch_times = []
  train_losses = []
  #NUM_EPOCH =0

  early_stopper = EarlyStopMonitor(max_round=args.patience)
  for epoch in range(NUM_EPOCH):
    # print("现在是",epoch)
    start_epoch = time.time()
    ### Training

    # Reinitialize memory of the model at the start of each epoch
    if USE_MEMORY:
      tgn.memory.__init_memory__()

    # Train using only training graph
    tgn.set_neighbor_finder(train_ngh_finder)
    m_loss = []


    logger.info('start {} epoch'.format(epoch))
    train_ts = 0
    incre_ts = max(incre_data.timestamps[0:BATCH_SIZE])
    incre_batch_id = 0


    #num_batch=0
    for k in range(0, num_batch, args.backprop_every):

      loss = 0
      incre_flag = False
      optimizer.zero_grad()

      # Custom loop to allow to perform backpropagation only every a certain number of batches
      for j in range(args.backprop_every):
        batch_idx = k + j

        if batch_idx >= num_batch:
          continue

        #print(train_ts , incre_ts, batch_idx, incre_batch_id)

        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(num_instance, start_idx + BATCH_SIZE)


        sources_batch, destinations_batch, sou_neighbor, des_neighbor = train_data.sources[start_idx:end_idx], \
                                            train_data.destinations[start_idx:end_idx],train_data.u1[start_idx:end_idx],train_data.v1[start_idx:end_idx]
        edge_idxs_batch = train_data.edge_idxs[start_idx: end_idx]
        timestamps_batch = train_data.timestamps[start_idx:end_idx]

        size = len(sources_batch)
        train_ts=min(timestamps_batch)
        _, negatives_batch = train_rand_sampler.sample(size)

        with torch.no_grad():
            pos_label = torch.ones(size, dtype=torch.float, device=device)
            neg_label = torch.zeros(size, dtype=torch.float, device=device)

        InLN = InLN.train()

        pos_prob, neg_prob, sou_emb, des_emb = tgn.compute_edge_probabilities_topn(sources_batch, destinations_batch, negatives_batch,
                                                            timestamps_batch, edge_idxs_batch, sou_neighbor,
                                                            des_neighbor, NUM_NEIGHBORS)
        #loss += criterion(pos_prob.squeeze(), pos_label) + criterion(neg_prob.squeeze(), neg_label)
        loss += MSELoss(sou_emb, des_emb)*100+ criterion(pos_prob.squeeze(), pos_label) + criterion(neg_prob.squeeze(), neg_label)

        loss.backward()
        optimizer.step()
        m_loss.append(loss.item())
        if USE_MEMORY:
            tgn.memory.detach_memory()



        if train_ts > incre_ts :
        #if False:
        # 增加 extra batch for KG dynamical updating
            loss = 0
            optimizer.zero_grad()


            start_idx = incre_batch_id * BATCH_SIZE
            end_idx = min(num_instance, start_idx + BATCH_SIZE)
            sources_batch, destinations_batch = incre_data.sources[start_idx:end_idx], incre_data.destinations[start_idx:end_idx]
            sou_neighbor = 9999
            des_neighbor = 9999
            edge_idxs_batch = incre_data.edge_idxs[start_idx: end_idx]
            timestamps_batch = incre_data.timestamps[start_idx:end_idx]
            size = len(sources_batch)

            incre_ts=max(incre_data.timestamps[(incre_batch_id+1) * BATCH_SIZE : (incre_batch_id+2) * BATCH_SIZE],default=9999)
            if incre_ts == 9999:
                continue
            incre_batch_id+=1
            incre_flag=True

            if np.random.random() > 0.67:
                _, negatives_batch = train_rand_sampler.sample(size)

                with torch.no_grad():
                  pos_label = torch.ones(size, dtype=torch.float, device=device)
                  neg_label = torch.zeros(size, dtype=torch.float, device=device)

                InLN = InLN.train()

                pos_prob, neg_prob = tgn.compute_edge_probabilities(sources_batch, destinations_batch, negatives_batch,
                                                                    timestamps_batch, edge_idxs_batch, sou_neighbor, des_neighbor, NUM_NEIGHBORS)

                loss += criterion(pos_prob.squeeze(), pos_label) + criterion(neg_prob.squeeze(), neg_label)
                loss.backward()
                optimizer.step()
                m_loss.append(loss.item())
                if USE_MEMORY:
                    tgn.memory.detach_memory()
                    torch.cuda.empty_cache()



    epoch_time = time.time() - start_epoch
    epoch_times.append(epoch_time)


    ### Validation
    # Validation uses the full graph
    tgn.set_neighbor_finder(full_ngh_finder)

    if USE_MEMORY:
      # Backup memory at the end of training, so later we can restore it and use it for the
      # validation on unseen nodes
      train_memory_backup = tgn.memory.backup_memory()

    _,_, val_ranks = eval_edge_prediction_2(model = tgn, negative_edge_sampler = val_rand_sampler, train_data = train_data, test_data = val_data,
    train_rand_sampler = train_rand_sampler, train_bs = BATCH_SIZE, device = device,
    n_neighbors = NUM_NEIGHBORS)

    val_10 = sum(np.array(val_ranks) <= 10) * 1.0 / len(val_ranks)

    if USE_MEMORY:
      val_memory_backup = tgn.memory.backup_memory()
      # Restore memory we had at the end of training to be used when validating on new nodes.
      # Also backup memory after validation so it can be used for testing (since test edges are
      # strictly later in time than validation edges)
      tgn.memory.restore_memory(train_memory_backup)

    # Validate on unseen nodes
    _,_, val_nn_ranks = eval_edge_prediction_2(model = tgn, negative_edge_sampler = val_rand_sampler, train_data = train_data, test_data = new_node_val_data,
    train_rand_sampler = train_rand_sampler, train_bs = BATCH_SIZE, device = device,
    n_neighbors = NUM_NEIGHBORS)
    nn_val_10 = sum(np.array(val_nn_ranks) <= 10) * 1.0 / len(val_nn_ranks)

    if USE_MEMORY:
      # Restore memory we had at the end of validation
      tgn.memory.restore_memory(val_memory_backup)

    new_nodes_val_aps.append(nn_val_10)
    val_aps.append(val_10)
    train_losses.append(np.mean(m_loss))

    # Save temporary results to disk
    pickle.dump({
      "val_aps": val_aps,
      "new_nodes_val_aps": new_nodes_val_aps,
      "train_losses": train_losses,
      "epoch_times": epoch_times,
      "total_epoch_times": total_epoch_times
    }, open(results_path, "wb"))

    total_epoch_time = time.time() - start_epoch
    total_epoch_times.append(total_epoch_time)

    logger.info('epoch: {} took {:.2f}s'.format(epoch, total_epoch_time))
    logger.info('Epoch mean loss: {}'.format(np.mean(m_loss)))
    logger.info(
      'val: {}, new node val: {}'.format(val_10, nn_val_10))
    # logger.info(
    #   'val ap: {}, new node val ap: {}'.format(val_ap, nn_val_ap))

    # Early stopping
    if early_stopper.early_stop_check(val_10):
      logger.info('No improvement over {} epochs, stop training'.format(early_stopper.max_round))
      logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
      best_model_path = get_checkpoint_path(early_stopper.best_epoch)
      tgn.load_state_dict(torch.load(best_model_path))
      logger.info(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
      tgn.eval()
      break
    else:
      torch.save(tgn.state_dict(), get_checkpoint_path(epoch))

  # Training has finished, we have loaded the best model, and we want to backup its current
  # memory (which has seen validation edges) so that it can also be used when testing on unseen
  # nodes
  if USE_MEMORY:
    val_memory_backup = tgn.memory.backup_memory()

  ### Test
  tgn.embedding_module.neighbor_finder = full_ngh_finder

  #test_ap, test_auc = eval_edge_prediction(model=tgn,negative_edge_sampler=test_rand_sampler,data=test_data,n_neighbors=NUM_NEIGHBORS)
  _,_, old_ranks = eval_edge_prediction_2(model=tgn, negative_edge_sampler=test_rand_sampler, train_data=train_data, test_data=test_data,
                                             train_rand_sampler=train_rand_sampler, train_bs=BATCH_SIZE, device=device,
                                             n_neighbors=NUM_NEIGHBORS)

  if USE_MEMORY:
    tgn.memory.restore_memory(val_memory_backup)

  # Test on unseen nodes
  #nn_test_ap, nn_test_auc = eval_edge_prediction_2(model=tgn,negative_edge_sampler=nn_test_rand_sampler,data=new_node_test_data,n_neighbors=NUM_NEIGHBORS)
  _,_, ranks = eval_edge_prediction_2(model=tgn, negative_edge_sampler=nn_test_rand_sampler, train_data=train_data, test_data=new_node_test_data,
                                                   train_rand_sampler=train_rand_sampler, train_bs=BATCH_SIZE, device=device,
                                                   n_neighbors=NUM_NEIGHBORS)

  #print(ranks)
  #print(old_ranks)
  mrr_nn = 1/np.mean(ranks)
  rec10_nn = sum(np.array(ranks) <= 10) * 1.0 / len(ranks)
  mrr_ol = 1/np.mean(old_ranks)
  rec10_ol = sum(np.array(old_ranks) <= 10) * 1.0 / len(old_ranks)

  logger.info(
    'Test statistics: Old nodes -- mrr: {}, rec10: {}'.format(mrr_ol, rec10_ol))
  logger.info(
    'Test statistics: New nodes -- mrr: {}, rec10: {}'.format(mrr_nn, rec10_nn))



  logger.info('Saving TGN model')
  if USE_MEMORY:
    # Restore memory at the end of validation (save a model which is ready for testing)
    tgn.memory.restore_memory(val_memory_backup)
  torch.save(tgn.state_dict(), MODEL_SAVE_PATH)
  logger.info('TGN model saved')