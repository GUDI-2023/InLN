import json
import numpy as np
import pandas as pd
from pathlib import Path
import argparse


def preprocess(data_name):
  u_list, i_list, ts_list, label_list = [], [], [], []
  feat_l = []
  idx_list = []

  with open(data_name) as f:
    s = next(f)
    for idx, line in enumerate(f):
      e = line.strip().split(',')
      u = int(e[0])
      i = int(e[1])

      ts = float(e[2])
      label = float(e[3])  # int(e[3])

      feat = np.array([float(x) for x in e[4:]])

      u_list.append(u)
      i_list.append(i)
      ts_list.append(ts)
      label_list.append(label)
      idx_list.append(idx)

      feat_l.append(feat)
  return pd.DataFrame({'e1': u_list,
                       'e2': i_list,
                       'timestamp': ts_list,
                       'label': label_list
                        ,'idx': idx_list
                       }), np.array(feat_l)



def reindex(df,kg, bipartite=True):
  new_kg = kg.copy()
  new_df = df.copy()

  if bipartite:
    #assert (df.u.max() - df.u.min() + 1 == len(df.u.unique()))
    #assert (df.i.max() - df.i.min() + 1 == len(df.i.unique()))

    upper_u = df.e1.max() + 1
    new_e1 = kg.e1 + upper_u
    new_e2 = kg.e2 + upper_u

    new_kg.e1 = new_e1 + 1
    new_kg.e2 = new_e2 + 1

    new_ie2 = df.e2 + upper_u
    new_df.e2 = new_ie2


    #out=pd.concat([new_kg, new_df], axis=0, ignore_index=True)
    #out.e1 += 1
    #out.e2 += 1

    new_df.e1+=1
    new_df.e2+=1
    new_df.idx += len(new_kg)+1
    new_df.columns =['u','i','ts','label','idx']


    new_kg['idx']=list(range(1,len(new_kg)+1))

  else:
    new_df.u += 1
    new_df.i += 1
    new_df.idx += 1

  return new_kg, new_df


def run(data_name, bipartite=True):
  Path("data/").mkdir(parents=True, exist_ok=True)
  PATH = './data/{}.csv'.format(data_name)
  OUT_DF = './data/ml_{}.csv'.format(data_name)
  OUT_FEAT = './data/ml_{}.npy'.format(data_name)
  OUT_NODE_FEAT = './data/ml_{}_node.npy'.format(data_name)
  OUT_KG = './data/kgpart_{}.csv'.format(data_name)

  df, feat_inter = preprocess(PATH)

  if feat_inter.shape[1] ==1 or 2:
    feat_inter=np.zeros((feat_inter.shape[0],100))

  print('feat_inter', feat_inter.shape)
  
  kg = pd.read_csv('./data/kg_{}.csv'.format(data_name))

  only_kg, only_interact = reindex(df,kg, bipartite)

  # print (only_kg.info())
  # print (len(only_kg))
  # print(only_kg[2557743:2557750])


  #生成对应的 edge embedding (先kg+后 inter)

  #读取 kg relation
  rel_emb = pd.DataFrame(np.load('./data/relation_pretrain_{}.npy'.format(data_name)))
  temp_kg= only_kg.merge(rel_emb, left_on=['r'], right_on=rel_emb.index,how='left')

  empty = np.zeros(100)[np.newaxis, :]
  edge_emb=np.array(temp_kg.iloc[:,5:])
  feat_kg = np.row_stack((empty,edge_emb))

  edge_feat = np.row_stack((feat_kg, feat_inter))

  print('edge_feat.shape',edge_feat.shape)



  max_user_idx = only_interact.u.max()
  #print(max_user_idx)
  zero_user_feat = np.zeros((max_user_idx + 1, 100))
  # 读取 pretrain kg item emb
  item_emb = np.load('./data/ent_pretrain_{}.npy'.format(data_name))
  node_feat = np.row_stack((zero_user_feat, item_emb))

  print("node_feat.shape",node_feat.shape)

  only_kg.to_csv(OUT_KG)
  only_interact.to_csv(OUT_DF)


  np.save(OUT_NODE_FEAT, node_feat) # node feat 是zero初始化的
  np.save(OUT_FEAT, edge_feat) #edge feature npy 存储

parser = argparse.ArgumentParser('Interface for TGN data preprocessing')
parser.add_argument('--data', type=str, help='Dataset name (eg. wikipedia or reddit)',
                    default='wikipedia')
parser.add_argument('--bipartite', action='store_true', help='Whether the graph is bipartite')

args = parser.parse_args()

args.bipartite = True

run(args.data, bipartite=args.bipartite)
