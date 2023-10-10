import numpy as np
from scipy.linalg import sqrtm
from scipy.special import softmax
import networkx as nx
from networkx.algorithms.community.modularity_max import greedy_modularity_communities
import matplotlib.pyplot as plt
from matplotlib import animation
# from IPython.display import HTML


g = nx.karate_club_graph()
print((g.number_of_nodes(), g.number_of_edges()))

A = nx.to_numpy_array(g)
A_mod = A + np.eye(g.number_of_nodes())  # with self connections

D_mod = np.zeros_like(A_mod)
np.fill_diagonal(D_mod, A_mod.sum(axis=1))
D_mod_sqrt_inv = np.linalg.inv(np.sqrt(D_mod))

A_hat = D_mod_sqrt_inv@A_mod@D_mod_sqrt_inv

X = np.eye(g.number_of_nodes())


def glorot_init(n_in, n_out):
    sd = np.sqrt(6/(n_in + n_out))
    return np.random.uniform(-sd, sd, size=(n_in, n_out))


def xent(pred, labels):
    return -np.log(pred)[np.arange(pred.shape[0]), np.argmax(labels, axis=1)]


def norm_diff(dW, dW_approx):
    return np.linalg.norm(dW-dW_approx)/(np.linalg.norm(dW)+np.linalg.norm(dW_approx))


class GradDescentOptim():
    def __init__(self, lr, wd):
        self.lr = lr
        self.wd = wd
        self._y_true = None
        self._y_pred = None
        self._out = None
        self.bs = None
        self.train_nodes = None

    def __call__(self, y_pred, y_true, train_nodes=None):
        self.y_pred = y_pred
        self.y_true = y_true

        if train_nodes is None:
            self.train_nodes = np.arange(y_pred.shape[0])
        else:
            self.train_nodes = train_nodes

        self.bs = self.train_nodes.shape[0]

    @property
    def out(self):
        return self._out

    @out.setter
    def out(self, y):
        self._out = y


class GCNLayer():
    def __init__(self, n_inputs, n_outputs, activation=None, name=''):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.W = glorot_init(n_inputs, n_outputs)
        self.activation = activation
        self.name = name

    def __repr__(self):
        return f"GCN: W{'_' + self.name if self.name else ''} ({self.n_inputs}, {self.n_outputs})"

    def forward(self, A, X, W=None):
        """
        Assumes A is (bs, bs) adjacency matrix and X is (bs, D),
            where bs = "batch size" and D = input feature length
        """
        self._A = A
        self._X = (A@X).T

        if W is None:
            W = self.W
        H = 

