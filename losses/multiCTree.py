import numpy as np
import torch as tc
import higra as hg
import imageio

import math
from torch.nn import Module
from torch.autograd import Function
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn

try:
    from utils import * # imshow, locate_resource
except: # we are probably running from the cloud, try to fetch utils functions from URL
    import urllib.request as request; exec(request.urlopen('https://github.com/higra/Higra-Notebooks/raw/master/utils.py').read(), globals())


class ComponentTreeFunction(Function):
  @staticmethod
  def forward(ctx, graph, vertex_weights, tree_type="max", plateau_derivative="full"):
    """
    Construct a component tree of the given vertex weighted graph.

    tree_type must be in ("min", "max", "tos")

    plateau_derivative can be "full" or "single". In the first case, the gradient of an altitude component
    is back-propagated to the vertex weights of the whole plateau (to all proper vertices of the component).
    In the second case, an arbitrary vertex of the plateau is selected and will receive the gradient.

    return: the altitudes of the tree (torch tensor), the tree itself is stored as an attribute of the tensor
    """
    if tree_type == "max":
      tree, altitudes = hg.component_tree_max_tree(graph, vertex_weights.detach().numpy())
    elif tree_type == "min":
      tree, altitudes = hg.component_tree_min_tree(graph, vertex_weights.detach().numpy())
    elif tree_type == "tos":
      tree, altitudes = hg.component_tree_tree_of_shapes_image2d(vertex_weights.detach().numpy())
    else:
      raise ValueError("Unknown tree type " + str(tree_type))

    if plateau_derivative == "full":
      plateau_derivative = True
    elif plateau_derivative == "single":
      plateau_derivative = False
    else:
      raise ValueError("Unknown plateau derivative type " + str(plateau_derivative))
    ctx.saved = (tree, graph, plateau_derivative)
    altitudes = tc.from_numpy(altitudes).clone().requires_grad_(True)
    # torch function can only return tensors, so we hide the tree as a an attribute of altitudes
    altitudes.tree = tree
    return altitudes

  @staticmethod
  def backward(ctx, grad_output):
    tree, graph, plateau_derivative = ctx.saved
    if plateau_derivative:
      grad_in = grad_output[tree.parents()[:tree.num_leaves()]]
    else:
      leaf_parents = tree.parents()[:tree.num_leaves()]
      _, indices = np.unique(leaf_parents, return_index=True)
      grad_in = tc.zeros((tree.num_leaves(),), dtype=grad_output.dtype)
      grad_in[indices] = grad_output[leaf_parents[indices]]
    return None, hg.delinearize_vertex_weights(grad_in, graph), None

class ComponentTree(Module):
    def __init__(self, tree_type):
        super().__init__()
        tree_types = ("max", "min", "tos")
        if tree_type not in tree_types:
          raise ValueError("Unknown tree type " + str(tree_type) + " possible values are " + " ".join(tree_types))

        self.tree_type = tree_type

    def forward(self, graph, vertex_weights):
        altitudes = ComponentTreeFunction.apply(graph, vertex_weights, self.tree_type)
        return altitudes.tree, altitudes

max_tree = ComponentTree("max")
min_tree = ComponentTree("min")
tos_tree = ComponentTree("tos")


class Optimizer:
    def __init__(self, loss, lr, optimizer="adam"):
        """
        Create an Optimizer utility object

        loss: function that takes a single torch tensor which support requires_grad = True and returns a torch scalar
        lr: learning rate
        optimizer: "adam" or "sgd"
        """
        self.loss_function = loss
        self.history = []
        self.optimizer = optimizer
        self.lr = lr
        self.best = None
        self.best_loss = 1

    def fit(self, data, iter=100, debug=False, min_lr=1e-6):
        """
        Fit the given data

        data: torch tensor, input data
        iter: int, maximum number of iterations
        debug: int, if > 0, print current loss value and learning rate every debug iterations
        min_lr: float, minimum learning rate (an LR scheduler is used), if None, no LR scheduler is used 
        """
        data = data.clone().requires_grad_(True)
        if self.optimizer == "adam":
            optimizer = tc.optim.Adam([data], lr=self.lr, amsgrad=True)
        else:
            optimizer = tc.optim.SGD([data], lr=self.lr)

        if min_lr:
            lr_scheduler = tc.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=10)
    
        for t in range(iter):
            optimizer.zero_grad()
            print(data)
            loss = self.loss_function(tc.relu(data))

            loss.backward()
            optimizer.step()  
            loss_value = loss.item()
            
            self.history.append(loss_value) 
            if loss_value < self.best_loss:
                self.best_loss = loss_value
                self.best = tc.relu(data).clone()
                
            if min_lr:
                lr_scheduler.step(loss_value)
                if optimizer.param_groups[0]['lr'] <= min_lr:
                    break

            if debug and t % debug == 0:
                print("Iteration {}: Loss: {:.4f}, LR: {}".format(t, loss_value, optimizer.param_groups[0]['lr']))
        return self.best

    def show_history(self):
        """
        Plot loss history
        """
        plt.plot(self.history)
        plt.show()


def loss_ranked_selection(saliency_measure, importance_measure, num_positives, margin, p=1):
    """
    Will try to increase the ranked_measure value of the num_positives first elements to the margin value and decrease the measure on the remaining elements

    :param saliency_measure: 1d torch tensor
    :param importance_measure: torch tensor (same shape as saliency_measure)
    :param num_positive: int >= 0
    :param p: float >= 0 
    :return: a torch scalar
    """
    sorted_indices = tc.argsort(importance_measure, descending=True)
    saliency_measure = saliency_measure[sorted_indices]

    if len(saliency_measure) <= num_positives:
        return tc.sum(tc.relu(margin - saliency_measure)**p)
    else:
        return tc.sum(tc.relu(margin - saliency_measure[:num_positives])**p) + tc.sum(saliency_measure[num_positives:]**p)


def attribute_depth(tree, altitudes):
    """
    Compute the depth of any node of the tree which is equal to the largest altitude 
    in the subtree rooted in the current node. 

    :param tree: input tree
    :param altitudes: np array (1d), altitudes of the input tree nodes
    :return: np array (1d), depth of the tree nodes
    """
    return hg.accumulate_sequential(tree, altitudes[:tree.num_leaves()], hg.Accumulators.max)

def attribute_saddle_nodes(tree, attribute):
    """
    Let n be a node and let an be an ancestor of n. The node an has a single child node that contains n denoted by ch(an -> n). 
    The saddle and base nodes associated to a node n for the given attribute values are respectively the closest ancestor an  
    of n and the node ch(an -> n) such that there exists a child c of an with attr(ch(an -> n)) < attr(c). 

    :param tree: input tree
    :param attribute: np array (1d), attribute of the input tree nodes
    :return: (np array, np array), saddle and base nodes of the input tree nodes for the given attribute
    """
    max_child_index = hg.accumulate_parallel(tree, attribute, hg.Accumulators.argmax)
    child_index = hg.attribute_child_number(tree)
    main_branch = child_index == max_child_index[tree.parents()]
    main_branch[:tree.num_leaves()] = True

    saddle_nodes = hg.propagate_sequential(tree, np.arange(tree.num_vertices())[tree.parents()], main_branch)
    base_nodes = hg.propagate_sequential(tree, np.arange(tree.num_vertices()), main_branch)
    return saddle_nodes, base_nodes

def attribute_dice(image_gt, image, tree):
    nbGTPix = image_gt.sum()
    areaNodes = hg.attribute_area(tree)
    gt = np.reshape(image_gt,len(image_gt))
    image = np.reshape(image,len(image))
    att = hg.accumulate_sequential(tree, (image != 0) & (gt != 0), hg.Accumulators.sum)
    union = nbGTPix + areaNodes
    dice = 2 * att / union
    return hg.accumulate_and_max_sequential(tree, dice, dice[:len(image_gt)], hg.Accumulators.max)

def attribute_precision(image_gt, image, tree):
    nbGTPix = image_gt.sum()
    areaNodes = hg.attribute_area(tree)
    gt = np.reshape(image_gt,len(image_gt))
    image = np.reshape(image,len(image))
    att = hg.accumulate_sequential(tree, ((image != 0) & (gt != 0)).astype(int), hg.Accumulators.sum)
    precision = att / areaNodes
    precision[:len(image_gt)] = 0
    return hg.accumulate_and_max_sequential(tree, precision, precision[:len(image_gt)], hg.Accumulators.max)

def normalize(lst):
    min_val = min(lst)
    max_val = max(lst)
    normalized_lst = [(x - min_val) / (max_val - min_val) for x in lst]
    normalized_lst = [min(max(x, 0), 1) for x in normalized_lst]
    return normalized_lst


from skimage.filters import threshold_otsu, threshold_li

def otsu_threshold(importance_tensor):
    valid = importance_tensor[(importance_tensor > 0) & (importance_tensor < 1)]
    if valid.numel() == 0:
        return 0.5
    imp_np = valid.detach().cpu().numpy()
    if np.all(imp_np == imp_np[0]):
        return float(imp_np[0])

    return threshold_otsu(imp_np)

def li_threshold(importance_tensor):
    valid = importance_tensor[(importance_tensor > 0) & (importance_tensor < 1)]
    if valid.numel() == 0:
        return 0.5
    imp_np = valid.detach().cpu().numpy()
    if np.all(imp_np == imp_np[0]):
        return float(imp_np[0])

    return threshold_li(imp_np)


def sigmoid(x, t, lambda_=10):
    x = torch.tensor(x) if isinstance(x, list) else x
    return 1 / (1 + torch.exp(-lambda_ * (x - t)))

def loss_maxima(graph, image, image_gt, saliency_measure, importance_measure, p=1):
    """
    Loss that favors the presence of num_target_maxima in the given image.


    :param graph: adjacency pixel graph
    :param image: torch tensor 1d, vertex values of the input graph
    :param saliency_measure: string, how the saliency of maxima is measured, can be "altitude" or "dynamics"
    :param importance_measure: string, how the importance of maxima is measured, can be "altitude", "dynamics", "area", or "volume"
    :param num_target_maxima: int >=0, number of maxima that should be present in the result
    :param margin: float >=0, target altitude fo preserved maxima
    :param p: float >=0, power (see parameter p in loss_ranked_selection)
    :return: a torch scalar
    """
    if not saliency_measure in ["altitude", "dynamics", "disconnection", "connection"]:
    raise ValueError("Saliency_measure can be either 'altitude', 'dynamics', 'connect', 'disconnection', or 'connection'")

    if not importance_measure in ["altitude", "dynamics", "area", "volume", "precision"]:
    raise ValueError("Saliency_measure can be either 'altitude', 'dynamics', 'area', 'volume', or 'precision")

    tree, altitudes = max_tree(graph, image)
    altitudes_np = altitudes.detach().numpy()

    extrema = hg.attribute_extrema(tree, altitudes_np)
    extrema_indices = np.arange(tree.num_vertices())[extrema]
    extrema_altitudes = altitudes[tc.from_numpy(extrema_indices).long()]

    if importance_measure == "area":
        area = hg.attribute_area(tree)
        pass_nodes, base_nodes = attribute_saddle_nodes(tree, area)
        extrema_area = tc.from_numpy(area[base_nodes[extrema_indices]])

    if importance_measure == "volume":
        volume = hg.attribute_volume(tree, altitudes_np)
        pass_nodes, base_nodes = attribute_saddle_nodes(tree, volume)
        extrema_volume = tc.from_numpy(volume[base_nodes[extrema_indices]])

    saliency = []
    if saliency_measure[0] == "altitude" and saliency_measure[1] == "connection" and saliency_measure[2] == "disconnection":
        depth = attribute_depth(tree, altitudes_np)
        saddle_nodes = tc.from_numpy(attribute_saddle_nodes(tree, depth)[0]).long()
        extrema_connection = altitudes[saddle_nodes[extrema_indices]]
        extrema_disconnection = tc.max(image) - altitudes[saddle_nodes[extrema_indices]]
        saliency.append(extrema_altitudes)
        saliency.append(extrema_connection)
        saliency.append(extrema_disconnection)

    if importance_measure == "altitude":
        importance = extrema_altitudes
    elif importance_measure == "dynamics":
        importance = extrema_dynamics
    elif importance_measure == "area":
        importance = extrema_area
    elif importance_measure == "volume":
        importance = extrema_volume

    elif importance_measure == "precision":
        height, width = image_gt.shape  # get from input directly
        image_gt = image_gt.reshape(height * width, 1)
        image = image.detach().numpy().reshape(height * width, 1)
        precision = attribute_precision(image_gt, image, tree)
        extinction_value = hg.attribute_extinction_value(tree, altitudes_np, np.array(precision))
        importance = tc.tensor([extinction_value[i] for i in extrema_indices])
    
    threshold = otsu_threshold(importance)

    if hasattr(threshold, 'item'):
        threshold = threshold.item()

    sigmoid_values = sigmoid(importance, threshold)

    return tc.sum((saliency[0] * (1 - sigmoid_values)) + (tc.relu(saliency[1] * (1 - sigmoid_values)) + (saliency[2] * sigmoid_values))), threshold