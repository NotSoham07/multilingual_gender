
from typing import Dict
import numpy as np
import scipy
import classifier
from typing import List
from tqdm import tqdm
import random
import warnings


def get_rowspace_projection(W: np.ndarray) -> np.ndarray:
    if np.allclose(W, 0):
        w_basis = np.zeros_like(W.T)
    else:
        w_basis = scipy.linalg.orth(W.T) # orthogonal basis

    P_W = w_basis.dot(w_basis.T) # orthogonal projection on W's rowspace

    return P_W

def get_projection_to_intersection_of_nullspaces(rowspace_projection_matrices: List[np.ndarray], input_dim: int):
    """
    Given a list of rowspace projection matrices P_R(w_1), ..., P_R(w_n),
    this function calculates the projection to the intersection of all nullspasces of the matrices w_1, ..., w_n.
    uses the intersection-projection formula of Ben-Israel 2013 http://benisrael.net/BEN-ISRAEL-NOV-30-13.pdf:
    N(w1)∩ N(w2) ∩ ... ∩ N(wn) = N(P_R(w1) + P_R(w2) + ... + P_R(wn))
    :param rowspace_projection_matrices: list of rowspace projection matrices
    """

    I = np.eye(input_dim)
    Q = np.sum(rowspace_projection_matrices, axis = 0)
    P = I - get_rowspace_projection(Q)

    return P

def debias_by_specific_directions(directions: List[np.ndarray], input_dim: int):
    """
    the goal of this function is to perform INLP on a set of user-provided directiosn (instead of learning those directions).
    :param directions: list of vectors, each of size input_dim
    :param input_dim: size of input vectors
    :return: P, the debiasing projection
    """

    rowspace_projections = []

    for v in directions:
        P_v = get_rowspace_projection(v)
        rowspace_projections.append(P_v)

    P = get_projection_to_intersection_of_nullspaces(rowspace_projections, input_dim)

    return P


def get_debiasing_projection(classifier_class, cls_params: Dict, num_classifiers: int, input_dim: int,
                             is_autoregressive: bool,
                             min_accuracy: float, X_train: np.ndarray, Y_train: np.ndarray, X_dev: np.ndarray,
                             Y_dev: np.ndarray, by_class=True, Y_train_main=None,
                             Y_dev_main=None, dropout_rate = 0) -> np.ndarray:
    """
    :param classifier_class: a class of a linear classifier (e.g. classifier.SKlearnClassifier)
    :param cls_params: parameters for the classifier
    :param num_classifiers: number of classifiers to train
    :param input_dim: size of input vectors
    :param is_autoregressive: whether the training is autoregressive (i.e. the training set is augmented with the predictions of the previous classifier)
    :param min_accuracy: minimum accuracy of the classifier on the dev set
    :param X_train: training set
    :param Y_train: training labels
    :param X_dev: dev set
    :param Y_dev: dev labels

    """
    if dropout_rate > 0 and is_autoregressive:
        warnings.warn("Note: when using dropout with autoregressive training, the property w_i.dot(w_(i+1)) = 0 no longer holds.")

    I = np.eye(input_dim)

    if by_class:
        if ((Y_train_main is None) or (Y_dev_main is None)):
            raise Exception("Need main-task labels for by-class training.")
        main_task_labels = list(set(Y_train_main.tolist()))

    X_train_cp = X_train.copy()
    X_dev_cp = X_dev.copy()
    rowspace_projections = []
    Ws = []

    pbar = tqdm(range(num_classifiers))
    for i in pbar:

        clf = classifier.SKlearnClassifier(classifier_class(**cls_params))
        dropout_scale = 1./(1 - dropout_rate + 1e-6)
        dropout_mask = (np.random.rand(*X_train.shape) < (1-dropout_rate)).astype(float) * dropout_scale


        if by_class:
            cls = random.choice(main_task_labels)
            relevant_idx_train = Y_train_main == cls
            relevant_idx_dev = Y_dev_main == cls
        else:
            relevant_idx_train = np.ones(X_train_cp.shape[0], dtype=bool)
            relevant_idx_dev = np.ones(X_dev_cp.shape[0], dtype=bool)

        acc = clf.train_network((X_train_cp * dropout_mask)[relevant_idx_train], Y_train[relevant_idx_train], X_dev_cp[relevant_idx_dev], Y_dev[relevant_idx_dev])
        pbar.set_description("iteration: {}, accuracy: {}".format(i, acc))
        if acc < min_accuracy: continue

        W = clf.get_weights()
        Ws.append(W)
        P_rowspace_wi = get_rowspace_projection(W) # projection to W's rowspace
        rowspace_projections.append(P_rowspace_wi)

        if is_autoregressive:

            """
            project the training set to the intersection of all nullspaces of the matrices w_1, ..., w_i

            """
    
            P = get_projection_to_intersection_of_nullspaces(rowspace_projections, input_dim)
            # project

            X_train_cp = (P.dot(X_train.T)).T
            X_dev_cp = (P.dot(X_dev.T)).T

    """
    project the training set to the intersection of all nullspaces of the matrices w_1, ..., w_n

    """

    P = get_projection_to_intersection_of_nullspaces(rowspace_projections, input_dim)

    return P, rowspace_projections, Ws


def get_debiasing_projection_iters(classifier_class, cls_params: Dict, num_classifiers: int, input_dim: int,
                             is_autoregressive: bool,
                             min_accuracy: float, X_train: np.ndarray, Y_train: np.ndarray, X_dev: np.ndarray,
                             Y_dev: np.ndarray, by_class=False, Y_train_main=None,
                             Y_dev_main=None, dropout_rate = 0) -> np.ndarray:
    """
    :param classifier_class: a class of a linear classifier (e.g. classifier.SKlearnClassifier)
    :param cls_params: parameters for the classifier
    :param num_classifiers: number of classifiers to train
    :param input_dim: size of input vectors
    :param is_autoregressive: whether the training is autoregressive (i.e. the training set is augmented with the predictions of the previous classifier)
    :param min_accuracy: minimum accuracy of the classifier on the dev set
    :param X_train: training set
    :param Y_train: training labels
    :param X_dev: dev set
    :param Y_dev: dev labels

    """
    if dropout_rate > 0 and is_autoregressive:
        warnings.warn("Note: when using dropout with autoregressive training, the property w_i.dot(w_(i+1)) = 0 no longer holds.")

    I = np.eye(input_dim)
    P = np.eye(input_dim)

    if by_class:
        if ((Y_train_main is None) or (Y_dev_main is None)):
            raise Exception("Need main-task labels for by-class training.")
        main_task_labels = list(set(Y_train_main.tolist()))

    X_train_cp = X_train.copy()
    X_dev_cp = X_dev.copy()
    rowspace_projections = []
    Ws = []
    bs = []

    all_scores = []
    pbar = tqdm(range(num_classifiers))
    for i in pbar:

        clf = classifier.SKlearnClassifier(classifier_class(**cls_params))
        dropout_scale = 1./(1 - dropout_rate + 1e-6)
        dropout_mask = (np.random.rand(*X_train.shape) < (1-dropout_rate)).astype(float) * dropout_scale

        if by_class:
            #cls = np.random.choice(Y_train_main)  # uncomment for frequency-based sampling
            cls = random.choice(main_task_labels)
            relevant_idx_train = Y_train_main == cls
            relevant_idx_dev = Y_dev_main == cls
        else:
            relevant_idx_train = np.ones(X_train_cp.shape[0], dtype=bool)
            relevant_idx_dev = np.ones(X_dev_cp.shape[0], dtype=bool)

        acc = clf.train_network((X_train_cp * dropout_mask)[relevant_idx_train], Y_train[relevant_idx_train], X_dev_cp[relevant_idx_dev], Y_dev[relevant_idx_dev])
        pbar.set_description("iteration: {}, accuracy: {}".format(i, acc))
        if acc < min_accuracy: continue
        print("acc", acc)
        score = clf.model.score(X_train_cp.dot(P), Y_train)
        print("score train", score)

        all_scores.append(acc)
        W = clf.get_weights()
        b = clf.model.intercept_
        Ws.append(W)
        bs.append(b)
        P_rowspace_wi = get_rowspace_projection(W) # projection to W's rowspace
        rowspace_projections.append(P_rowspace_wi)

        if is_autoregressive:

            """
            project the training set to the intersection of all nullspaces of the matrices w_1, ..., w_i
            """

            P = get_projection_to_intersection_of_nullspaces(rowspace_projections, input_dim)
            # project

            X_train_cp = (P.dot(X_train.T)).T
            X_dev_cp = (P.dot(X_dev.T)).T



    """
    calculae final projection matrix P=PnPn-1....P2P1
    since w_i.dot(w_i-1) = 0, P2P1 = I - P1 - P2 (proof in the paper); this is more stable.
    by induction, PnPn-1....P2P1 = I - (P1+..+PN). We will use instead Ben-Israel's formula to increase stability and also generalize to the non-orthogonal case (e.g. with dropout),
    i.e., we explicitly project to intersection of all nullspaces (this is not critical at this point; I-(P1+...+PN) is roughly as accurate as this provided no dropout & regularization)
    """

    P = get_projection_to_intersection_of_nullspaces(rowspace_projections, input_dim)

    return P, rowspace_projections, Ws, bs, all_scores


if __name__ == '__main__':

    from sklearn.linear_model import SGDClassifier, Perceptron, LogisticRegression

    N = 10000
    d = 300
    X = np.random.rand(N, d) - 0.5
    Y = np.array([1 if sum(x) > 0 else 0 for x in X]) #X < 0 #np.random.rand(N) < 0.5 #(X + 0.01 * (np.random.rand(*X.shape) - 0.5)) < 0 #np.random.rand(5000) < 0.5
    #Y = np.array(Y, dtype = int)

    num_classifiers = 200
    classifier_class = SGDClassifier #Perceptron
    input_dim = d
    is_autoregressive = True
    min_accuracy = 0.0

    P, rowspace_projections, Ws = get_debiasing_projection(classifier_class, {}, num_classifiers, input_dim, is_autoregressive, min_accuracy, X, Y, X, Y, by_class = False)

    I = np.eye(P.shape[0])
    P_alternative = I - np.sum(rowspace_projections, axis = 0)
    P_by_product = I.copy()

    for P_Rwi in rowspace_projections:

        P_Nwi = I - P_Rwi
        P_by_product = P_Nwi.dot(P_by_product)


    """testing"""

    # validate that P = PnPn-1...P2P1 (should be true only when w_i.dot(w_(i+1)) = 0, in autoregressive training)

    if is_autoregressive:
        assert np.allclose(P_alternative, P)
        assert np.allclose(P_by_product, P)

    # validate that P is a projection

    assert np.allclose(P.dot(P), P)

    # validate that P projects to N(w1)∩ N(w2) ∩ ... ∩ N(wn)

    x = np.random.rand(d) - 0.5
    for w in Ws:

        assert np.allclose(np.linalg.norm(w.dot(P.dot(x))), 0.0)

    # validate that each two classifiers are orthogonal (this is expected to be true only with autoregressive training)

    if is_autoregressive:
        for i,w in enumerate(Ws):

            for j, w2 in enumerate(Ws):

                if i == j: continue

                assert np.allclose(np.linalg.norm(w.dot(w2.T)), 0)
