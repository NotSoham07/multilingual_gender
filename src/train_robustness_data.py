import argparse
import json
import numpy as np
import pickle
import random
import time
import warnings
from collections import defaultdict, Counter
from typing import List, Dict
from sklearn.linear_model import SGDClassifier
import debias

warnings.filterwarnings("ignore")


class DatasetLoader:

    def __init__(self, lang, layer, embed_type, no_gender):
        self.layer = layer
        self.embed_type = embed_type
        self.no_gender = no_gender
        self.p2i, self.i2p = self.load_dictionary("../data/robustness/data/profession2index.txt")
        self.g2i, self.i2g = self.load_dictionary("../data/robustness/data/gender2index.txt")

    def load_dictionary(self, path):
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        k2v, v2k = {}, {}
        for line in lines:
            k, v = line.strip().split("\t")
            v = int(v)
            k2v[k] = v
            v2k[v] = k

        return k2v, v2k

    def load_dataset(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data

    def count_profs_and_gender(self, data: List[dict]):
        counter = defaultdict(Counter)
        for entry in data:
            gender, prof = entry["g"], entry["p"]
            counter[prof][gender] += 1

        return counter

    def load_data(self):
        train = self.load_dataset("../data/robustness/data/train.pickle")
        dev = self.load_dataset("../data/robustness/data/dev.pickle")
        test = self.load_dataset("../data/robustness/data/test.pickle")
        counter = self.count_profs_and_gender(train + dev + test)

        path = "../data/robustness/bert_encode_data/"
        if self.no_gender:
            path = "../data/robustness/bert_encode_data/"

        if not self.layer:
            if not self.embed_type or self.embed_type == "cls":
                x_train = np.load(path + "train_cls_mbert.npy")
                x_dev = np.load(path + "dev_cls_mbert.npy")
                x_test = np.load(path + "test_cls_mbert.npy")
            else:
                assert(self.embed_type == "avg")
                x_train = np.load(path + "train_avg_mbert.npy")
                x_dev = np.load(path + "dev_avg_mbert.npy")
                x_test = np.load(path + "test_avg_mbert.npy")

        else:
            if self.embed_type:
                x_train = np.load(path + "train_{}_layer{}_mbert.npy".format(self.embed_type, self.layer))
                x_dev = np.load(path + "dev_{}_layer{}_mbert.npy".format(self.embed_type, self.layer))
                x_test = np.load(path + "test_{}_layer{}_mbert.npy".format(self.embed_type, self.layer))
            else:
                raise ValueError("please indicate type")

        assert len(train) == len(x_train)
        assert len(dev) == len(x_dev)
        assert len(test) == len(x_test)

        f, m = 0., 0.
        prof2fem = dict()

        for k, values in counter.items():
            f += values['f']
            m += values['m']
            prof2fem[k] = values['f'] / (values['f'] + values['m'])

        print(f / (f + m))
        print(prof2fem)

        y_train = np.array([self.p2i[entry["p"]] for entry in train])
        y_dev = np.array([self.p2i[entry["p"]] for entry in dev])
        y_test = np.array([self.p2i[entry["p"]] for entry in test])

        return dev, train, test, x_train, x_dev, x_test, y_train, y_dev, y_test


class ProjectionMatrix:

    @staticmethod
    def get_projection_matrix(num_clfs, X_train, Y_train_gender, X_dev, Y_dev_gender, Y_train_task, Y_dev_task):

        is_autoregressive = True
        min_acc = 0.
        dim = 768
        n = num_clfs
        start = time.time()

        gender_clf = SGDClassifier
        params = {'loss': 'hinge', 'fit_intercept': True, 'max_iter': 3000000, 'tol': 1e-5, 'n_iter_no_change': 600,
                  'n_jobs': 16}

        P, rowspace_projections, Ws, bs, iters = debias.get_debiasing_projection_iters(gender_clf, params, n, dim, is_autoregressive,
                                                                                      min_acc,
                                                                                      X_train, Y_train_gender, X_dev, Y_dev_gender,
                                                                                      Y_train_main=Y_train_task, Y_dev_main=Y_dev_task,
                                                                                      by_class=False)

        print("time: {}".format(time.time() - start))
        return P, rowspace_projections, Ws, bs, iters


def main():

    random.seed(10)
    np.random.seed(10)
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default="test", help="language")
    parser.add_argument("--layer", help="layer to use")
    parser.add_argument("--type", help="cls or avg")
    parser.add_argument("--iters", type=int, default=300, help="num of iterations")
    parser.add_argument("--output_path", default="../data/robustness/inlp_matrices/", help="where to save P")
    parser.add_argument("--no_gender", action="store_true")

    args = parser.parse_args()

    lang = args.lang.upper()

    if args.no_gender:
        args.output_path += "wo_gender/"

    dataset_loader = DatasetLoader(lang, args.layer, args.type, args.no_gender)
    dev, train, test, x_train, x_dev, x_test, y_train, y_dev, y_test = dataset_loader.load_data()
    num_clfs = args.iters
    y_dev_gender = np.array([dataset_loader.g2i[d["g"]] for d in dev])
    y_train_gender = np.array([dataset_loader.g2i[d["g"]] for d in train])

    P, rowspace_projections, Ws, bs, iters = ProjectionMatrix.get_projection_matrix(num_clfs, x_train, y_train_gender, x_dev, y_dev_gender, y_train, y_dev)

    # Save to file

        # Save to file
    if not args.layer:
        if not args.type or args.type == "cls":
            # normal - last cls, no indication in the names
            np.save(args.output_path + "P_mbert_{}_{}.npy".format(args.iters, lang), P)
            np.save(args.output_path + "Ws_mbert_{}_{}.npy".format(args.iters, lang), Ws)
            np.save(args.output_path + "bs_mbert_{}_{}.npy".format(args.iters, lang), bs)
            with open(args.output_path + "iters_mbert_{}_{}.json".format(args.iters, lang), "w") as f:
                json.dump(iters, f)

        else:
            assert(args.type == "avg")
            # last avg, no indication of layer in the names
            np.save(args.output_path + "P_mbert_{}_{}_avg.npy".format(args.iters, lang), P)
            np.save(args.output_path + "Ws_mbert_{}_{}_avg.npy".format(args.iters, lang), Ws)
            np.save(args.output_path + "bs_mbert_{}_{}_avg.npy".format(args.iters, lang), bs)
            with open(args.output_path + "iters_mbert_{}_{}_avg.json".format(args.iters, lang), "w") as f:
                json.dump(iters, f)

    else:
        if args.type:
            np.save(args.output_path + "P_mbert_{}_{}_{}_layer{}.npy".format(args.iters, lang, args.type, args.layer), P)
            np.save(args.output_path + "Ws_mbert_{}_{}_{}_layer{}.npy".format(args.iters, lang, args.type, args.layer), Ws)
            np.save(args.output_path + "bs_mbert_{}_{}_{}_layer{}.npy".format(args.iters, lang, args.type, args.layer), bs)
            with open(args.output_path + "iters_mbert_{}_{}_{}_layer{}.json".format(args.iters, lang, args.type, args.layer), "w") as f:
                json.dump(iters, f)

        else:
            raise ValueError("please indicate type")



if __name__ == '__main__':
    main()

