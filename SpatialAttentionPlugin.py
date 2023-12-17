import torch

def get_spatial_attn(att_mat):
    """
    Given a multihead attention map, output normalized attention.
    """
    att_mat = torch.stack(att_mat).squeeze(1)
    # Average the attention weights across all heads.
    att_mat = torch.mean(att_mat, dim=1).cpu()
    #att_mat = att_mat[-1]

    # To account for residual connections, we add an identity matrix to the
    # attention matrix and re-normalize the weights.
    residual_att = torch.eye(att_mat.size(1)).cpu()
    aug_att_mat = att_mat + residual_att
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)
    # Recursively multiply the weight matrices
    joint_attentions = torch.zeros(aug_att_mat.size())
    joint_attentions[0] = aug_att_mat[0]

    for n in range(1, aug_att_mat.size(0)):
        joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])

    # Attention from the output token to the input space.
    v = joint_attentions[-1]
    final_attn = v[0, 1:]/v[0, 1:].sum()

    # final_attn = final_attn/final_attn.max()
    # final_attn = final_attn.reshape([8,8])
    # final_attn
    return final_attn.detach().numpy()

import numpy as np

import PyIO
import PyPluMA
import pickle
class SpatialAttentionPlugin:
 def input(self, inputfile):
     self.parameters = PyIO.readParameters(inputfile)
 def run(self):
     pass
 def output(self, outputfile):
     features = PyIO.readSequential(PyPluMA.prefix()+"/"+self.parameters["features"])
     infile = open(PyPluMA.prefix()+"/"+self.parameters["attnfile"], "rb")
     all_attn = pickle.load(infile)
     all_spatial_attn_dict = dict()
     for feature in features:
         all_spatial_attn_dict[feature] = []
     # Initialize dictionary of all pixel values for each feature type
     for example_i in range(len(all_attn)):
      spatial_attn, feature_attn = all_attn[example_i]
      for feature_i in range(len(all_spatial_attn_dict.keys())):
        att_mat = spatial_attn[feature_i]
        final_attn = get_spatial_attn(att_mat)
        all_spatial_attn_dict[list(all_spatial_attn_dict.keys())[feature_i]] += list(final_attn)

     # Compute mean and SD for each feature:
     stat_spatial_attn_dict = dict()
     for feature_i in range(len(all_spatial_attn_dict.keys())):
      feature = list(all_spatial_attn_dict.keys())[feature_i]
      stat_spatial_attn_dict[feature]=dict()
      stat_spatial_attn_dict[feature]['mean'] = np.mean(all_spatial_attn_dict[feature])
      stat_spatial_attn_dict[feature]['std'] = np.std(all_spatial_attn_dict[feature])

     outfile = open(outputfile, "wb")
     pickle.dump(stat_spatial_attn_dict, outfile)
