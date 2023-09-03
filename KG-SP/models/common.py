import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from scipy.stats import hmean

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class MLP(nn.Module):
    '''
    Baseclass to create a simple MLP
    Inputs
        inp_dim: Int, Input dimension
        out-dim: Int, Output dimension
        num_layer: Number of hidden layers
        relu: Bool, Use non linear function at output
        bias: Bool, Use bias
    '''

    def __init__(self, inp_dim, out_dim, num_layers=1, relu=True, bias=True, dropout=False, norm=False, layers=[]):
        super(MLP, self).__init__()
        mod = []
        incoming = inp_dim
        for layer in range(num_layers - 1):
            if len(layers) == 0:
                outgoing = incoming
            else:
                outgoing = layers.pop(0)
            mod.append(nn.Linear(incoming, outgoing, bias=bias))

            incoming = outgoing
            if norm:
                mod.append(nn.LayerNorm(outgoing))
                # mod.append(nn.BatchNorm1d(outgoing))
            mod.append(nn.ReLU(inplace=True))
            # mod.append(nn.LeakyReLU(inplace=True, negative_slope=0.2))
            if dropout:
                mod.append(nn.Dropout(p=0.5))

        mod.append(nn.Linear(incoming, out_dim, bias=bias))
        #if norm:
        #    mod.append(nn.LayerNorm(out_dim))
        if relu:
            mod.append(nn.ReLU(inplace=True))
            # mod.append(nn.LeakyReLU(inplace=True, negative_slope=0.2))

        #if dropout:
        #    mod.append(nn.Dropout(p=0.5))
        self.mod = nn.Sequential(*mod)

    def forward(self, x):
        return self.mod(x)


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


def calculate_margines(domain_embedding, gt, margin_range=5):
    '''
    domain_embedding: pairs * feats
    gt: batch * feats
    '''
    batch_size, pairs, features = gt.shape[0], domain_embedding.shape[0], domain_embedding.shape[1]
    gt_expanded = gt[:, None, :].expand(-1, pairs, -1)
    domain_embedding_expanded = domain_embedding[None, :, :].expand(batch_size, -1, -1)
    margin = (gt_expanded - domain_embedding_expanded) ** 2
    margin = margin.sum(2)
    max_margin, _ = torch.max(margin, dim=0)
    margin /= max_margin
    margin *= margin_range
    return margin


def l2_all_batched(image_embedding, domain_embedding):
    '''
    Image Embedding: Tensor of Batch_size * pairs * Feature_dim
    domain_embedding: Tensor of pairs * Feature_dim
    '''
    pairs = image_embedding.shape[1]
    domain_embedding_extended = image_embedding[:, None, :].expand(-1, pairs, -1)
    l2_loss = (image_embedding - domain_embedding_extended) ** 2
    l2_loss = l2_loss.sum(2)
    l2_loss = l2_loss.sum() / l2_loss.numel()
    return l2_loss


def same_domain_triplet_loss(image_embedding, trip_images, gt, hard_k=None, margin=2):
    '''
    Image Embedding: Tensor of Batch_size * Feature_dim
    Triplet Images: Tensor of Batch_size * num_pairs * Feature_dim
    GT: Tensor of Batch_size
    '''
    batch_size, pairs, features = trip_images.shape
    batch_iterator = torch.arange(batch_size).to(device)
    image_embedding_expanded = image_embedding[:, None, :].expand(-1, pairs, -1)

    diff = (image_embedding_expanded - trip_images) ** 2
    diff = diff.sum(2)

    positive_anchor = diff[batch_iterator, gt][:, None]
    positive_anchor = positive_anchor.expand(-1, pairs)

    # Calculating triplet loss
    triplet_loss = positive_anchor - diff + margin

    # Setting positive anchor loss to 0
    triplet_loss[batch_iterator, gt] = 0

    # Removing easy triplets
    triplet_loss[triplet_loss < 0] = 0

    # If only mining hard triplets
    if hard_k:
        triplet_loss, _ = triplet_loss.topk(hard_k)

    # Counting number of valid pairs
    num_positive_triplets = triplet_loss[triplet_loss > 1e-16].size(0)

    # Calculating the final loss
    triplet_loss = triplet_loss.sum() / (num_positive_triplets + 1e-16)

    return triplet_loss


def cross_domain_triplet_loss(image_embedding, domain_embedding, gt, hard_k=None, margin=2):
    '''
    Image Embedding: Tensor of Batch_size * Feature_dim
    Domain Embedding: Tensor of Num_pairs * Feature_dim
    gt: Tensor of Batch_size with ground truth labels
    margin: Float of margin
    Returns:
        Triplet loss of all valid triplets
    '''
    batch_size, pairs, features = image_embedding.shape[0], domain_embedding.shape[0], domain_embedding.shape[1]
    batch_iterator = torch.arange(batch_size).to(device)
    # Now dimensions will be Batch_size * Num_pairs * Feature_dim
    image_embedding = image_embedding[:, None, :].expand(-1, pairs, -1)
    domain_embedding = domain_embedding[None, :, :].expand(batch_size, -1, -1)

    # Calculating difference
    diff = (image_embedding - domain_embedding) ** 2
    diff = diff.sum(2)

    # Getting the positive pair
    positive_anchor = diff[batch_iterator, gt][:, None]
    positive_anchor = positive_anchor.expand(-1, pairs)

    # Calculating triplet loss
    triplet_loss = positive_anchor - diff + margin

    # Setting positive anchor loss to 0
    triplet_loss[batch_iterator, gt] = 0

    # Removing easy triplets
    triplet_loss[triplet_loss < 0] = 0

    # If only mining hard triplets
    if hard_k:
        triplet_loss, _ = triplet_loss.topk(hard_k)

    # Counting number of valid pairs
    num_positive_triplets = triplet_loss[triplet_loss > 1e-16].size(0)

    # Calculating the final loss
    triplet_loss = triplet_loss.sum() / (num_positive_triplets + 1e-16)

    return triplet_loss


def same_domain_triplet_loss_old(image_embedding, positive_anchor, negative_anchor, margin=2):
    '''
    Image Embedding: Tensor of Batch_size * Feature_dim
    Positive anchor: Tensor of Batch_size * Feature_dim
    negative anchor: Tensor of Batch_size * negs *Feature_dim
    '''
    batch_size, negs, features = negative_anchor.shape
    dist_pos = (image_embedding - positive_anchor) ** 2
    dist_pos = dist_pos.sum(1)
    dist_pos = dist_pos[:, None].expand(-1, negs)

    image_embedding_expanded = image_embedding[:, None, :].expand(-1, negs, -1)
    dist_neg = (image_embedding_expanded - negative_anchor) ** 2
    dist_neg = dist_neg.sum(2)

    triplet_loss = dist_pos - dist_neg + margin
    triplet_loss[triplet_loss < 0] = 0
    num_positive_triplets = triplet_loss[triplet_loss > 1e-16].size(0)

    triplet_loss = triplet_loss.sum() / (num_positive_triplets + 1e-16)
    return triplet_loss


def pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x ** 2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    # Ensure diagonal is zero if x=y
    # if y is None:
    #     dist = dist - torch.diag(dist.diag)
    return torch.clamp(dist, 0.0, np.inf)


class Evaluator:

    def __init__(self, dset, model, fast=False):

        self.dset = dset

        # Convert text pairs to idx tensors: [('sliced', 'apple'), ('ripe', 'apple'), ...] --> torch.LongTensor([[0,1],[1,1], ...])
        pairs = [(dset.attr2idx[attr], dset.obj2idx[obj]) for attr, obj in dset.pairs]
        self.train_pairs = [(dset.attr2idx[attr], dset.obj2idx[obj]) for attr, obj in dset.train_pairs]
        self.pairs = torch.LongTensor(pairs)

        # Mask over pairs that occur in closed world
        # Select set based on phase
        if dset.phase == 'train':
            print('Evaluating with train pairs')
            test_pair_set = set(dset.train_pairs)
            test_pair_gt = set(dset.train_pairs)
        elif dset.phase == 'val':
            print('Evaluating with validation pairs')
            test_pair_set = set(dset.val_pairs + dset.train_pairs)  # todo: check if this makes sense
            test_pair_gt = set(dset.val_pairs)
        else:
            print('Evaluating with test pairs')
            test_pair_set = set(dset.test_pairs + dset.train_pairs)
            test_pair_gt = set(dset.test_pairs)
            
        print(dset.train_pairs)
        print(dset.val_pairs)
        if dset.open_world and not fast:
            masks = [1 for _ in dset.pairs]
        else:
            masks = [1 if pair in test_pair_set else 0 for pair in dset.pairs]

        self.closed_mask = torch.BoolTensor(masks)
        # Mask of seen concepts
        seen_pair_set = set(dset.train_pairs)
        mask = [1 if pair in seen_pair_set else 0 for pair in dset.pairs]
        self.seen_mask = torch.BoolTensor(mask)

        # Object specific mask over which pairs occur in the object oracle setting
        oracle_obj_mask = []
        for _obj in dset.objs:
            mask = [1 if _obj == obj else 0 for attr, obj in dset.pairs]
            oracle_obj_mask.append(torch.BoolTensor(mask))
        self.oracle_obj_mask = torch.stack(oracle_obj_mask, 0)

        # Decide if the model under evaluation is a manifold model or not
        self.score_model = self.score_manifold_model

    # Generate mask for each settings, mask scores, and get prediction labels
    def generate_predictions(self, scores, obj_truth, bias=0.0, topk=5):  # (Batch, #pairs)
        '''
        Inputs
            scores: Output scores
            obj_truth: Ground truth object
        Returns
            results: dict of results in 3 settings
        '''

        def get_pred_from_scores(_scores, topk):
            '''
            Given list of scores, returns top 10 attr and obj predictions
            Check later
            '''
            _, pair_pred = _scores.topk(topk, dim=1)  # sort returns indices of k largest values
            pair_pred = pair_pred.contiguous().view(-1)
            attr_pred, obj_pred = self.pairs[pair_pred][:, 0].view(-1, topk), \
                                  self.pairs[pair_pred][:, 1].view(-1, topk)
            return (attr_pred, obj_pred)

        results = {}
        orig_scores = scores.clone()
        mask = self.seen_mask.repeat(scores.shape[0], 1)  # Repeat mask along pairs dimension
        scores[~mask] += bias  # Add bias to test pairs

        # Unbiased setting

        # Open world setting --no mask, all pairs of the dataset
        results.update({'open': get_pred_from_scores(scores, topk)})
        results.update({'unbiased_open': get_pred_from_scores(orig_scores, topk)})
        # Closed world setting - set the score for all Non test pairs to -1e10,
        # this excludes the pairs from set not in evaluation
        mask = self.closed_mask.repeat(scores.shape[0], 1)
        closed_scores = scores.clone()
        closed_scores[~mask] = -1e10
        closed_orig_scores = orig_scores.clone()
        closed_orig_scores[~mask] = -1e10
        results.update({'closed': get_pred_from_scores(closed_scores, topk)})
        results.update({'unbiased_closed': get_pred_from_scores(closed_orig_scores, topk)})

        # Object_oracle setting - set the score to -1e10 for all pairs where the true object does Not participate, can also use the closed score
        mask = self.oracle_obj_mask[obj_truth]
        oracle_obj_scores = scores.clone()
        oracle_obj_scores[~mask] = -1e10
        oracle_obj_scores_unbiased = orig_scores.clone()
        oracle_obj_scores_unbiased[~mask] = -1e10
        results.update({'object_oracle': get_pred_from_scores(oracle_obj_scores, 1)})
        results.update({'object_oracle_unbiased': get_pred_from_scores(oracle_obj_scores_unbiased, 1)})

        return results

    def score_clf_model(self, scores, obj_truth, topk=5):
        '''
        Wrapper function to call generate_predictions for CLF models
        '''
        attr_pred, obj_pred = scores

        # Go to CPU
        attr_pred, obj_pred, obj_truth = attr_pred.to('cpu'), obj_pred.to('cpu'), obj_truth.to('cpu')

        # Gather scores (P(a), P(o)) for all relevant (a,o) pairs
        # Multiply P(a) * P(o) to get P(pair)
        attr_subset = attr_pred.index_select(1, self.pairs[:, 0])  # Return only attributes that are in our pairs
        obj_subset = obj_pred.index_select(1, self.pairs[:, 1])
        scores = (attr_subset * obj_subset)  # (Batch, #pairs)

        results = self.generate_predictions(scores, obj_truth)
        results['biased_scores'] = scores  # todo: Check if needed

        return results

    def score_manifold_model(self, scores, obj_truth, bias=0.0, topk=5):
        '''
        Wrapper function to call generate_predictions for manifold models
        '''
        # Go to CPU
        scores = {k: v.to('cpu') for k, v in scores.items()}
        obj_truth = obj_truth.to(device)

        # Gather scores for all relevant (a,o) pairs
        scores = torch.stack(
            [scores[(attr, obj)] for attr, obj in self.dset.pairs], 1
        )  # (Batch, #pairs)
        orig_scores = scores.clone()
        results = self.generate_predictions(scores, obj_truth, bias, topk)
        results['scores'] = orig_scores
        return results

    def score_fast_model(self, scores, obj_truth, bias=0.0, topk=5):
        '''
        Wrapper function to call generate_predictions for manifold models
        '''

        results = {}
        mask = self.seen_mask.repeat(scores.shape[0], 1)  # Repeat mask along pairs dimension
        scores[~mask] += bias  # Add bias to test pairs

        mask = self.closed_mask.repeat(scores.shape[0], 1)
        closed_scores = scores.clone()
        closed_scores[~mask] = -1e10

        _, pair_pred = closed_scores.topk(topk, dim=1)  # sort returns indices of k largest values
        pair_pred = pair_pred.contiguous().view(-1)
        attr_pred, obj_pred = self.pairs[pair_pred][:, 0].view(-1, topk), \
                              self.pairs[pair_pred][:, 1].view(-1, topk)

        results.update({'closed': (attr_pred, obj_pred)})
        return results

    def evaluate_predictions(self, predictions, attr_truth, obj_truth, pair_truth, allpred, topk = 1):
        # Go to CPU
        attr_truth, obj_truth, pair_truth = attr_truth.to('cpu'), obj_truth.to('cpu'), pair_truth.to('cpu')

        pairs = list(
            zip(list(attr_truth.numpy()), list(obj_truth.numpy())))
        

        seen_ind, unseen_ind = [], []
        #print(len())
        for i in range(len(attr_truth)):
            if pairs[i] in self.train_pairs:
                seen_ind.append(i)
            else:
                unseen_ind.append(i)

        
        seen_ind, unseen_ind = torch.LongTensor(seen_ind), torch.LongTensor(unseen_ind)
        def _process(_scores):
            # Top k pair accuracy
            # Attribute, object and pair
            attr_match = (attr_truth.unsqueeze(1).repeat(1, topk) == _scores[0][:, :topk])
            obj_match = (obj_truth.unsqueeze(1).repeat(1, topk) == _scores[1][:, :topk])
          #  print(attr_match[:20])
            # Match of object pair
            match = (attr_match * obj_match).any(1).float()
            attr_match = attr_match.any(1).float()
           # print(attr_match[:20])
            obj_match = obj_match.any(1).float()
            # Match of seen and unseen pairs
            seen_match = match[seen_ind]
            unseen_match = match[unseen_ind]
            
            
            
         #   print(attr_match[seen_ind])
         #   print(attr_match[unseen_ind])
            ### Calculating class average accuracy
            
            # local_score_dict = copy.deepcopy(self.test_pair_dict)
            # for pair_gt, pair_pred in zip(pairs, match):
            #     # print(pair_gt)
            #     local_score_dict[pair_gt][2] += 1.0 #increase counter
            #     if int(pair_pred) == 1:
            #         local_score_dict[pair_gt][1] += 1.0

            # # Now we have hits and totals for classes in evaluation set
            # seen_score, unseen_score = [], []
            # for key, (idx, hits, total) in local_score_dict.items():
            #     score = hits/total
            #     if bool(self.seen_mask[idx]) == True:
            #         seen_score.append(score)
            #     else:
            #         unseen_score.append(score)
            # print(seen_ind)
            # print(unseen_ind)
            seen_score, unseen_score = torch.ones(512,5), torch.ones(512,5)

            return attr_match, obj_match, match, seen_match, unseen_match, \
            torch.Tensor(seen_score+unseen_score), torch.Tensor(seen_score), torch.Tensor(unseen_score),seen_ind, unseen_ind, 

        def _add_to_dict(_scores, type_name, stats):
            base = ['_attr_match', '_obj_match', '_match', '_seen_match', '_unseen_match', '_ca', '_seen_ca', '_unseen_ca']
            for val, name in zip(_scores, base):
                stats[type_name + name] = val

        ##################### Match in places where corrent object
        obj_oracle_match = (attr_truth == predictions['object_oracle'][0][:, 0]).float()  #object is already conditioned
        obj_oracle_match_unbiased = (attr_truth == predictions['object_oracle_unbiased'][0][:, 0]).float()

        stats = dict(obj_oracle_match = obj_oracle_match, obj_oracle_match_unbiased = obj_oracle_match_unbiased)

        #################### Closed world
        closed_scores = _process(predictions['closed'])
        unbiased_closed = _process(predictions['unbiased_closed'])
        _add_to_dict(closed_scores, 'closed', stats)
        _add_to_dict(unbiased_closed, 'closed_ub', stats)
        
        
        #################### Open world
        open_scores = _process(predictions['open'])
        unbiased_open = _process(predictions['unbiased_open'])
        _add_to_dict(open_scores, 'open', stats)
        _add_to_dict(unbiased_open, 'open_ub', stats)
        

        #################### Calculating AUC
        scores = predictions['scores']
        # getting score for each ground truth class
        correct_scores = scores[torch.arange(scores.shape[0]), pair_truth][unseen_ind]

        # Getting top predicted score for these unseen classes
        
        # print(  predictions['scores'][unseen_ind][:, self.seen_mask])
        max_seen_scores = predictions['scores'][unseen_ind][:, self.seen_mask].topk(topk, dim=1)[0][:, topk - 1]

        # Getting difference between these scores
        unseen_score_diff = max_seen_scores - correct_scores

        # Getting matched classes at max bias for diff
        unseen_matches = stats['closed_unseen_match'].bool()
        
        
       # print(unseen_matches)
        # print(correct_scores)
        # print(max_seen_scores)
         
         
        correct_unseen_score_diff = unseen_score_diff[unseen_matches] - 1e-4

        # sorting these diffs
        correct_unseen_score_diff = torch.sort(correct_unseen_score_diff)[0]
        magic_binsize = 20
        # getting step size for these bias values
        bias_skip = max(len(correct_unseen_score_diff) // magic_binsize, 1)
        # Getting list
        biaslist = correct_unseen_score_diff[::bias_skip]
        # print(biaslist)
        seen_match_max = float(stats['closed_seen_match'].mean())
        
        #if(len(unseen_ind)>0):
        unseen_match_max = float(stats['closed_unseen_match'].mean())
            
        # else:
        #     unseen_match_max = float(0)
        seen_accuracy, unseen_accuracy = [], []

        # Go to CPU
        base_scores = {k: v.to('cpu') for k, v in allpred.items()}
        obj_truth = obj_truth.to('cpu')

        # Gather scores for all relevant (a,o) pairs
        base_scores = torch.stack(
            [allpred[(attr,obj)] for attr, obj in self.dset.pairs], 1
        ) # (Batch, #pairs)
        # print(len(unseen_ind))
        # print(len(seen_ind))
        res_ub=unbiased_open
        res_ub_op=unbiased_open
        
        seen_accuracy_attr_cl=[]
        unseen_accuracy_attr_cl=[]
        hm_cl=[]
        
        
        seen_accuracy_attr_op=[]
        unseen_accuracy_attr_op=[]
        hm_op=[]
        
        if(len(biaslist)==0):
                biaslist=torch.FloatTensor(list(np.linspace(-2,2,magic_binsize+1)))
        
        
        for bias in biaslist:
            scores = base_scores.clone()
            results = self.score_fast_model(scores, obj_truth, bias = bias, topk = topk)
            res=results
            results = results['closed'] # we only need biased
            results_op = results
            
            
            results = _process(results)
            seen_match = float(results[3].mean())
            if(len(unseen_ind)>0):
                unseen_match = float(results[4].mean())
            else:
                unseen_match = seen_atr_acc
            seen_accuracy.append(seen_match)
            unseen_accuracy.append(unseen_match)
            
           
            seen_attr=results[0][res_ub[8]]
            un_seen_attr=results[0][res_ub[9]]
            seen_atr_acc=(sum(seen_attr==1))/len(seen_attr)

            if(len(un_seen_attr)>0):
        
            
                unseen_atr_acc=(sum(un_seen_attr==1))/len(un_seen_attr)

                attr_hm=2*(seen_atr_acc*unseen_atr_acc)/(seen_atr_acc+unseen_atr_acc)
            
            else:
                unseen_atr_acc=torch.tensor(np.nan)
                attr_hm=seen_atr_acc


        
            seen_accuracy_attr_cl.append(seen_atr_acc)
            unseen_accuracy_attr_cl.append(unseen_atr_acc)
            hm_cl.append(attr_hm)
            
        #  print(len(seen_attr),len(un_seen_attr),len(seen_attr)+len(un_seen_attr))
             # we only need biased
            results_op = _process(results_op)
            
        
        
            seen_attr_op=results_op[0][res_ub_op[8]]
            unseen_attr_op=results_op[0][res_ub_op[9]]
        
            seen_atr_acc=(sum(seen_attr_op==1))/len(seen_attr_op)
            if(len(unseen_attr_op)>0):
              #  print("!!! >0")
            
                unseen_atr_acc=(sum(unseen_attr_op==1))/len(unseen_attr_op)

                attr_hm=2*(seen_atr_acc*unseen_atr_acc)/(seen_atr_acc+unseen_atr_acc)
            
            else:
              #  print("!!! 0")
                unseen_atr_acc=np.nan
                attr_hm=np.nan
                
                
            seen_accuracy_attr_op.append(seen_atr_acc)
            unseen_accuracy_attr_op.append(unseen_atr_acc)
            hm_op.append(attr_hm)
         
            
        

        seen_accuracy.append(seen_match_max)
        unseen_accuracy.append(unseen_match_max)
        seen_accuracy, unseen_accuracy = np.array(seen_accuracy), np.array(unseen_accuracy)
        area = np.trapz(seen_accuracy, unseen_accuracy)
        
      
        # print(seen_accuracy)
        # print(unseen_accuracy)
        
        seen_accuracy_attr_op, unseen_accuracy_attr_op = np.array(seen_accuracy_attr_op), np.array(unseen_accuracy_attr_op)
        area_aatr = np.trapz(seen_accuracy_attr_op, unseen_accuracy_attr_op)

        #print(len(results))

       # print(results[7])
      #  print(results[8])
        

        for key in stats:
             
                stats[key] = float(stats[key].mean())

        # print(scores)
        # print(obj_truth)
        # print(biaslist)
        # print(seen_match_max)
        # print(unseen_match_max)
        # print(seen_accuracy, unseen_accuracy)
        
        seen_attr=res_ub[0][res_ub[8]]
        un_seen_attr=res_ub[0][res_ub[9]]
        seen_atr_acc=(sum(seen_attr==1))/len(seen_attr)

        if(len(un_seen_attr)>0):
       
          
            unseen_atr_acc=(sum(un_seen_attr==1))/len(un_seen_attr)

            attr_hm=2*(seen_atr_acc*unseen_atr_acc)/(seen_atr_acc+unseen_atr_acc)
        
        else:
            unseen_atr_acc=np.nan
            attr_hm=np.nan


       
        # stats['seen_closed_attr_all_bias_mean'] = np.mean(seen_accuracy_attr_cl)
        # stats['unseen_closed_attr_all_bias_mean'] =  np.mean(unseen_accuracy_attr_cl)
        # stats['hm_closed_attr_all_bias_mean'] =  np.mean(hm_cl)
        
        if(len(seen_accuracy_attr_cl)>0):
        
            stats['seen_closed_attr_all_bias_max'] = float(np.max(seen_accuracy_attr_cl))
        else:
            stats['seen_closed_attr_all_bias_max']=np.nan
            
            
        if(len(unseen_accuracy_attr_cl)>0):
            stats['unseen_closed_attr_all_bias_max'] =  float(np.max(unseen_accuracy_attr_cl))
            
        else:
             stats['unseen_closed_attr_all_bias_max'] =np.nan
            
            
        if(len(hm_cl)>0):
            stats['hm_closed_attr_all_bias_max'] =  float(np.max(hm_cl))
            
            
        else:
            stats['hm_closed_attr_all_bias_max'] = np.nan
        stats['AUC_attr'] =  area_aatr
        
        stats['seen_closed_attr'] = float(seen_atr_acc)
        stats['unseen_closed_attr'] = float(unseen_atr_acc)
        stats['hm_closed_attr'] = float(attr_hm)
        
        
        
        # print(seen_atr_acc,unseen_atr_acc,attr_hm,area_aatr)
        
      #  print(len(seen_attr),len(un_seen_attr),len(seen_attr)+len(un_seen_attr))
        seen_attr_op=res_ub_op[0][res_ub_op[8]]
        unseen_attr_op=res_ub_op[0][res_ub_op[9]]
       
        seen_atr_acc=(sum(seen_attr_op==1))/len(seen_attr_op)
        if(len(unseen_attr_op)>0):
       
          
            unseen_atr_acc=(sum(unseen_attr_op==1))/len(unseen_attr_op)

            attr_hm=2*(seen_atr_acc*unseen_atr_acc)/(seen_atr_acc+unseen_atr_acc)
        
        else:
            attr_hm=seen_atr_acc
            
           
          
        # stats['seen_open_attr_all_bias_mean'] = np.mean(seen_accuracy_attr_op)
        # stats['unseen_open_attr_all_bias_mean'] =  np.mean(unseen_accuracy_attr_op)
        # stats['hm_open_attr_all_bias_mean'] =  np.mean(hm_op)
        
        if(len(seen_accuracy_attr_op)>0):
        
            stats['seen_open_attr_all_bias_max'] = float(np.max(seen_accuracy_attr_op))
        else:
            stats['seen_open_attr_all_bias_max'] = 0
            
        if(len(unseen_accuracy_attr_op)>0):
        
             stats['unseen_open_attr_all_bias_max'] =  float(np.max(unseen_accuracy_attr_op))
        else:
            stats['unseen_open_attr_all_bias_max'] = 0    
            
            
       
       
        
        if(len(hm_op)>0):
            stats['hm_open_attr_all_bias_max'] =  np.max(hm_op)  
            idx=np.argmax(hm_op)  
            stats['hm_open_attr_all_bias_max_sa'] =  float(seen_accuracy_attr_op[idx])
            stats['hm_open_attr_all_bias_max_un'] =  float(unseen_accuracy_attr_op[idx])
        
        else:
            stats['hm_open_attr_all_bias_max']= 0    
            
            stats['hm_open_attr_all_bias_max_sa'] =   0   
            
            stats['hm_open_attr_all_bias_max_un'] =  0   
            
        
            
         
        stats['seen_open_attr'] = float(seen_atr_acc)
        stats['seen_open_attr'] = float(unseen_atr_acc)   
      
        stats['hm_open_attr'] = float(attr_hm)
        
        
        
        
        print(seen_atr_acc,unseen_atr_acc,attr_hm)
        
        harmonic_mean = hmean([seen_accuracy, unseen_accuracy], axis = 0)
        max_hm = np.max(harmonic_mean)
        idx = np.argmax(harmonic_mean)
        if idx == len(biaslist):
            bias_term = 1e3
        else:
            bias_term = biaslist[idx]
        stats['biasterm'] = float(bias_term)
        stats['best_unseen'] = np.max(unseen_accuracy)
        stats['best_seen'] = np.max(seen_accuracy)
        stats['AUC'] = area
        stats['hm_unseen'] = unseen_accuracy[idx]
        stats['hm_seen'] = seen_accuracy[idx]
        stats['best_hm'] = max_hm
        return stats



    def get_accuracies(self, predictions, attr_truth, obj_truth, pair_truth, topk=1):
        # Go to CPU
        attr_truth, obj_truth, pair_truth = attr_truth.to('cpu'), obj_truth.to('cpu'), pair_truth.to('cpu')

        pairs = list(
            zip(list(attr_truth.numpy()), list(obj_truth.numpy())))

        seen_ind, unseen_ind = [], []
        for i in range(len(attr_truth)):
            if pairs[i] in self.train_pairs:
                seen_ind.append(i)
            else:
                unseen_ind.append(i)

        seen_ind, unseen_ind = torch.LongTensor(seen_ind), torch.LongTensor(unseen_ind)

        scores = predictions['closed']

        attr_match = (attr_truth.unsqueeze(1).repeat(1, topk) == scores[0][:, :topk])
        obj_match = (obj_truth.unsqueeze(1).repeat(1, topk) == scores[1][:, :topk])

        # Match of object pair
        match = (attr_match * obj_match).any(1).float()
        attr_match = attr_match.any(1).float()
        obj_match = obj_match.any(1).float()
        # Match of seen and unseen pairs
        seen_match = match[seen_ind]
        unseen_match = match[unseen_ind]

        return attr_match.sum(), obj_match.sum(), seen_match.sum(), unseen_match.sum(), len(seen_ind), len(unseen_ind)




    def get_accuracies_fast(self, scores, attr_truth, obj_truth, pair_truth, bias=0., seen_mask=None,closed=False):
        # Valid only for OW!!!

        offset = len(self.dset.objs)
        biased_scores =  ((scores + (1-self.seen_mask.float())*bias))#
        if closed:
            biased_scores*=self.closed_mask
        #biased_scores =  ((scores + (1-self.seen_mask.float())*bias))
        # Go to CPU
        idx_attr, idx_obj, idx_pair = attr_truth.to('cpu'), obj_truth.to('cpu'), pair_truth.to('cpu')
        if seen_mask == None:
            seen_mask = torch.Tensor([self.seen_mask[p] for p in pair_truth])

        unseen_mask = (1.-seen_mask).float()

        idx_pred = biased_scores.max(1)[1]

        correct_pair = idx_pred == idx_pair

        attr_match = ((idx_pred//offset) == idx_attr).float()
        obj_match = ((idx_pred%offset) == idx_obj).float()

        seen_match = (correct_pair*seen_mask).sum()
        unseen_match = (correct_pair * unseen_mask).sum()

        return attr_match.sum(), obj_match.sum(), seen_match, unseen_match, seen_mask


    def compute_biases(self, scores, attr_truth, obj_truth, pair_truth, previous_list = None, bias=0.0, closed=False):

        # Valid only for OW!!!

        # Go to CPU
        idx_attr, idx_obj, idx_pair = attr_truth.to('cpu'), obj_truth.to('cpu'), pair_truth.to('cpu')

        if closed:
            scores*=self.closed_mask

        unseen_scores = scores - self.seen_mask*1e4
        seen_scores = scores - (1.-self.seen_mask.float()) * 1e4

        max_pred, idx_pred = unseen_scores.max(1)

        correct_pair = idx_pred == idx_pair
        if correct_pair.sum()==0:
            return previous_list

        correct_scores = max_pred[correct_pair]

        max_seen_scores = seen_scores[correct_pair].max(1)[0]

        # Getting difference between these scores
        unseen_score_diff = max_seen_scores - correct_scores - 1e-4

        if previous_list is not None:
            unseen_score_diff = torch.cat([unseen_score_diff,previous_list],dim=0)

        return unseen_score_diff

    def get_biases(self,unseen_score_diff):
        # sorting these diffs
        correct_unseen_score_diff = torch.sort(unseen_score_diff)[0]
        magic_binsize = 20
        # getting step size for these bias values
        bias_skip = max(len(correct_unseen_score_diff) // magic_binsize, 1)
        # Getting list
        biaslist = correct_unseen_score_diff[::bias_skip]

        return biaslist


    def collect_results(self,biaslist,results):
        stats = {}
        seen_accuracy = []
        unseen_accuracy = []
        for bias in biaslist:
            seen_match = float(results[bias]['seen'])
            unseen_match = float(results[bias]['unseen'])
            seen_accuracy.append(seen_match)
            unseen_accuracy.append(unseen_match)

        seen_accuracy, unseen_accuracy = np.array(seen_accuracy), np.array(unseen_accuracy)
        area = np.trapz(seen_accuracy, unseen_accuracy)

        harmonic_mean = hmean([seen_accuracy, unseen_accuracy], axis=0)
        max_hm = np.max(harmonic_mean)
        idx = np.argmax(harmonic_mean)

        bias_term = biaslist[idx]

        stats['biasterm'] = bias_term
        stats['best_unseen'] = np.max(unseen_accuracy)
        stats['best_seen'] = np.max(seen_accuracy)
        stats['AUC'] = area
        stats['hm_unseen'] = unseen_accuracy[idx]
        stats['hm_seen'] = seen_accuracy[idx]
        stats['biased_unseen'] = unseen_accuracy[-1]
        stats['biased_seen'] = seen_accuracy[-1]
        stats['best_hm'] = max_hm
        stats['hm_attr'] = results[bias_term]['attr_match']
        stats['hm_obj'] = results[bias_term]['obj_match']
        stats['b_attr'] = results[biaslist[-1]]['attr_match']
        stats['b_obj'] = results[biaslist[-1]]['obj_match']

        return stats

