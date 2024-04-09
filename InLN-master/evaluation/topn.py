import math

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


def eval_edge_prediction_2(model, negative_edge_sampler, train_data, test_data, n_neighbors,train_rand_sampler, train_bs, device, test_batch_size=128):
  # Ensures the random sampler uses a seed for evaluation (i.e. we sample always the same
  # negatives for validation / test set)
  assert negative_edge_sampler.seed is not None
  negative_edge_sampler.reset_random_state()


  with torch.no_grad():
    model = model.eval()
    num_instance = len(train_data.sources)
    num_batch = math.ceil(num_instance / train_bs)
    idx_list = np.arange(num_instance)
    rank=[]
    val_ap, val_auc = [], []
    test_batch_id = 0


    TEST_BATCH_SIZE = test_batch_size
    num_test_instance = len(test_data.sources)
    num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)
    for k in range(num_test_batch):
      batch_idx = k
      if batch_idx >= num_test_batch:
        continue

      s_idx = test_batch_id * TEST_BATCH_SIZE
      e_idx = min(num_test_instance, s_idx + TEST_BATCH_SIZE)
      sources_batch = test_data.sources[s_idx:e_idx]
      destinations_batch = test_data.destinations[s_idx:e_idx]
      timestamps_batch = test_data.timestamps[s_idx:e_idx]
      edge_idxs_batch = test_data.edge_idxs[s_idx: e_idx]
      sou_neighbor, des_neighbor = test_data.u1[s_idx: e_idx],test_data.v1[s_idx: e_idx]

      size = len(sources_batch)
      _, negative_samples = negative_edge_sampler.sample(size)
      pos_prob, neg_prob, batch_rank = model.compute_edge_probabilities_distance(sources_batch, destinations_batch,
                                                            negative_samples, timestamps_batch,
                                                            edge_idxs_batch, sou_neighbor, des_neighbor,n_neighbors)


      rank.extend(batch_rank)

      pred_score = np.concatenate([(pos_prob).cpu().numpy(), (neg_prob).cpu().numpy()])
      true_label = np.concatenate([np.ones(size), np.zeros(size)])

      val_ap.append(average_precision_score(true_label, pred_score))
      val_auc.append(roc_auc_score(true_label, pred_score))
  return pos_prob, neg_prob, rank


def eval_node_classification(tgn, decoder, data, edge_idxs, batch_size, n_neighbors):
  pred_prob = np.zeros(len(data.sources))
  num_instance = len(data.sources)
  num_batch = math.ceil(num_instance / batch_size)

  with torch.no_grad():
    decoder.eval()
    tgn.eval()
    for k in range(num_batch):
      s_idx = k * batch_size
      e_idx = min(num_instance, s_idx + batch_size)

      sources_batch = data.sources[s_idx: e_idx]
      destinations_batch = data.destinations[s_idx: e_idx]
      timestamps_batch = data.timestamps[s_idx:e_idx]
      edge_idxs_batch = edge_idxs[s_idx: e_idx]

      source_embedding, destination_embedding, _ = tgn.compute_temporal_embeddings(sources_batch,
                                                                                   destinations_batch,
                                                                                   destinations_batch,
                                                                                   timestamps_batch,
                                                                                   edge_idxs_batch,
                                                                                   n_neighbors)

      pred_prob_batch = decoder(source_embedding).sigmoid()
      pred_prob[s_idx: e_idx] = pred_prob_batch.cpu().numpy()

  auc_roc = roc_auc_score(data.labels, pred_prob)
  return auc_roc
