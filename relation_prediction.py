import random
import time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import tqdm
from scipy.sparse import csr_matrix
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import OneHotEncoder

from utils import get_prime_map_from_rel, load_data


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)


# For reproducibility
set_all_seeds(42)


def get_filtering_cache(df_train, df_eval):
    """
    Helper function that create a dict of the form:
    [(h,t)]-> [r1, ...] keeping an index of which (h,t) are
    connected through which relations from the train and eval
    triples.
    """
    cache_triples = defaultdict(list)
    all_triples = pd.concat((df_train, df_eval))
    for triple in tqdm.tqdm(all_triples.to_records()):
        # Adding h,t -> r
        cache_triples[(triple[1], triple[3])].append(triple[2])
    return cache_triples


projects = {
    "DDB14": {
        "path_to_files": "./data/DDB14/",
        "add_inverse_edges": "YES",
        "max_order": 4,
        "sim_pairs": 20,
        "selection_strategy": "tf-idf",
    },
    "WN18RR": {
        "path_to_files": "./data/WN18RR/",
        "add_inverse_edges": "YES",
        "max_order": 6,
        "sim_pairs": 40,
        "selection_strategy": "most_common",
    },
    "NELL995": {
        "path_to_files": "./data/NELL995/",
        "add_inverse_edges": "YES",
        "max_order": 4,
        "sim_pairs": 100,
        "selection_strategy": "tf-idf",
    },
}


total_results = []
for project_name, project_settings in projects.items():

    path_to_files = project_settings["path_to_files"]
    add_inverse_edges = project_settings["add_inverse_edges"]
    max_order = project_settings["max_order"]
    sim_pairs = project_settings["sim_pairs"]
    selection_strategy = project_settings["selection_strategy"]

    # Loading the data
    time_s = time.time()
    df_train_orig, df_train, df_eval, df_test, already_seen_triples = load_data(
        path_to_files, project_name, add_inverse_edges=add_inverse_edges
    )

    # Info on the data
    unique_rels = sorted(list(df_train["rel"].unique()))
    unique_nodes = sorted(
        set(df_train["head"].values.tolist() + df_train["tail"].values.tolist())
    )
    print(
        f"# of unique rels: {len(unique_rels)} \t | # of unique nodes: {len(unique_nodes)}"
    )

    time_prev = time.time()
    time_needed = time_prev - time_s
    print(
        f"Total time for reading: {time_needed:.5f} secs ({time_needed/60:.2f} mins)\n"
    )

    time_s = time.time()

    # Creating an ordinal mapping for nodes and prime mapping for relations
    node2id = {}
    id2node = {}
    for i, node in enumerate(unique_nodes):
        node2id[node] = i
        id2node[i] = node

    rel2id, id2rel = get_prime_map_from_rel(unique_rels)

    time_prev = time.time()
    time_needed = time_prev - time_s
    print(
        f"Total time for mapping: {time_needed:.5f} secs ({time_needed/60:.2f} mins)\n"
    )

    # If inverse edges are added with the extra INV suffix, use this to keep track
    # of the original edges.
    if "YES__INV" in add_inverse_edges:
        original_rel2id = {}
        for i, (key, value) in enumerate(rel2id.items()):
            if i % 2 == 1:
                continue
            else:
                original_rel2id[key] = value
    else:
        original_rel2id = rel2id.copy()

    time_s = time.time()

    # Create the 1-hop sparse PAM
    type_of_values_prime = np.float64
    type_of_values_non_prime = type_of_values_prime

    val_rels_dict_product = defaultdict(lambda: 1)
    val_rels_dict_sum = defaultdict(int)
    num_rels_dict = defaultdict(int)
    c = 0

    skipped_edge_indexes = []
    for i, row in df_train.iterrows():
        val_rels_dict_sum[(node2id[row["head"]], node2id[row["tail"]])] += rel2id[
            str(row["rel"])
        ]
        num_rels_dict[(node2id[row["head"]], node2id[row["tail"]])] += 1

    row = []
    col = []
    val_rels = []

    num_rels = []
    for key, val in val_rels_dict_sum.items():
        row.append(key[0])
        col.append(key[1])
        val_rels.append(val)
        num_rels.append(num_rels_dict[key])

    row = np.array(row)
    col = np.array(col)
    val_rels = np.array(val_rels)

    num_rels = np.array(num_rels)
    print("Will create the sparse matrices")

    A_big = csr_matrix(
        (val_rels, (row, col)), shape=(len(unique_nodes), len(unique_nodes))
    )

    A_num_rels = csr_matrix(
        (num_rels, (row, col)), shape=(len(unique_nodes), len(unique_nodes))
    )
    print("Created the sparse matrices")
    print(A_big.shape, f"Sparsity: {100 * (1 - A_big.nnz/(A_big.shape[0]**2)):.2f} %")

    time_prev = time.time()
    time_needed = time_prev - time_s
    print(
        f"Total time for creating 1-hop sparse: {time_needed:.5f} secs ({time_needed/60:.2f} mins)\n"
    )

    # Move on to create the P^k for the wanted k

    A_big_log = A_big.copy()

    A_big_log.data = np.log(A_big_log.data)

    power_A = [A_big_log]
    time_s = time.time()
    time_prev = time_s
    for ii in range(1, max_order):
        updated_power = power_A[-1] * A_big_log
        updated_power.sort_indices()
        if "YES" in add_inverse_edges:
            pass

        updated_power.setdiag(0)
        updated_power.eliminate_zeros()
        power_A.append(updated_power)
        sparsity = 1 - updated_power.nnz / (updated_power.shape[0] ** 2)
        print(f"Sparsity {ii}-hop: {100 * sparsity:.2f} %")

        time_prev = time.time()
        time_needed = time_prev - time_s
        print(f"{ii}: {time_needed:.5f} secs ({time_needed/60:.2f} mins)")
        if sparsity <= 0.8 and ii >= 2:
            print()
            print(f"This is not sparse anymore, breaking for efficiency..")
            print()
            max_order = len(power_A)
            break
    time_prev = time.time()
    time_needed = time_prev - time_s
    print(
        f"Total time for P^{max_order}: {time_needed:.5f} secs ({time_needed/60:.2f} mins)\n"
    )
    len(power_A)

    # Map the initial data to node indices and relation primes
    df_train_mapped = df_train.copy()
    df_train_mapped["rel"] = df_train["rel"].map(rel2id).astype(int)
    df_train_mapped["head"] = df_train["head"].map(node2id).astype(int)
    df_train_mapped["tail"] = df_train["tail"].map(node2id).astype(int)
    for i, row in df_train_mapped.iterrows():
        print(row)
        break
    print(f"Length before dropping nan", len(df_train_mapped))
    df_train_mapped.dropna(inplace=True)
    print(f"Length after dropping nan", len(df_train_mapped))
    df_train_mapped = df_train_mapped.astype(int)

    # Create the features of the paths for the original train pairs
    features_ij, labels_ij = {}, {}
    if add_inverse_edges == "YES":
        true_train = df_train_mapped.iloc[: df_train_mapped.shape[0] // 2]
    elif add_inverse_edges == "NO":
        true_train = df_train_mapped
    elif add_inverse_edges == "YES__INV":
        true_train = df_train_mapped[
            df_train_mapped["rel"].isin(original_rel2id.values())
        ]

    else:
        raise KeyError(f"{add_inverse_edges} not understood..")

    for i, row in true_train.iterrows():
        cur_row, cur_col = row["head"], row["tail"]
        if (cur_row, cur_col) not in features_ij:
            features_ij[(cur_row, cur_col)] = np.zeros((max_order,))
            labels_ij[(cur_row, cur_col)] = set()
        for cur_hop in range(0, max_order):
            cur_power = power_A[cur_hop]
            features_ij[(cur_row, cur_col)][cur_hop] = cur_power[(cur_row, cur_col)]
        labels_ij[(cur_row, cur_col)] = labels_ij[(cur_row, cur_col)].union(
            [row["rel"]]
        )

    # Check unique (i,j) pairs are as expected
    assert len(features_ij) == true_train.groupby(["head", "tail"]).nunique().shape[0]

    # Create the feature vectors for the nodes
    node_feats = []
    for cur in power_A:
        cp = cur.copy()
        outgoing = cp.sum(axis=1).A1
        incoming = cp.sum(axis=0).A1
        node_feats.append(outgoing)
        node_feats.append(incoming)
    node_feats = np.array(node_feats).T
    print(node_feats.shape)

    X_train = []
    y_train = []
    train_pairs = []

    add_inverse_pair_features = True
    add_node_features = True

    for ij, ij_feature_dict in features_ij.items():
        cur_features_pair = ij_feature_dict
        if add_inverse_pair_features:
            try:
                cur_features_inverse = features_ij[(ij[1], features_ij[0])]
            except KeyError:
                cur_features_inverse = np.zeros((max_order,))
            cur_features_pair = np.hstack(
                (cur_features_pair, cur_features_inverse)
            ).reshape(1, -1)
        if add_node_features:
            cur_features_nodes = np.hstack(
                (
                    node_feats[ij[0], :].reshape(1, -1),
                    node_feats[ij[1], :].reshape(1, -1),
                )
            )
            cur_features = np.hstack((cur_features_pair, cur_features_nodes))
        else:
            cur_features = cur_features_pair
        X_train.append(cur_features.reshape(-1))
        y_train.append(list(labels_ij[ij]))
        train_pairs.append(ij)

    assert 0 == (np.array(X_train).sum(axis=1) == 0).sum()
    assert len(X_train) == len(features_ij)

    # One-hot encode the data
    ohe = OneHotEncoder(handle_unknown="ignore")
    X_train_ohe = ohe.fit_transform(X_train)

    print(f"Extracted features for the train pairs...\n")

    # Repeat the same feature extraction procedure for the test data
    df_test_mapped = df_test.copy()
    df_test_mapped["rel"] = df_test["rel"].map(rel2id)
    df_test_mapped["head"] = df_test["head"].map(node2id)
    df_test_mapped["tail"] = df_test["tail"].map(node2id)
    for i, row in df_test_mapped.iterrows():
        print(row)
        break
    print(f"Length before dropping nan", len(df_test_mapped))
    df_test_mapped.dropna(inplace=True)
    print(f"Length after dropping nan", len(df_test_mapped))
    df_test_mapped = df_test_mapped.astype(int)

    df_test_mapped["ij"] = tuple(
        zip(
            df_test_mapped["head"].astype(int).values,
            df_test_mapped["tail"].astype(int).values,
        )
    )
    X_test = []
    y_test = []
    count_no_features = 0
    for i, row in df_test_mapped[["rel", "ij"]].iterrows():
        cur_dict = np.zeros((2 * max_order))
        for cur_hop in range(0, max_order):
            cur_power = power_A[cur_hop]
            cur_dict[cur_hop] = cur_power[row["ij"]]
            if add_inverse_pair_features:
                cur_dict[cur_hop + max_order] = cur_power[row["ij"][1], row["ij"][0]]
            if add_node_features:
                cur_features_nodes = np.hstack(
                    (
                        node_feats[row["ij"][0], :].reshape(1, -1),
                        node_feats[row["ij"][1], :].reshape(1, -1),
                    )
                )
                cur_features = np.hstack(
                    (cur_dict.reshape(1, -1), cur_features_nodes.reshape(1, -1))
                )
            else:
                cur_features = cur_dict
        X_test.append(cur_features.reshape(-1))
        y_test.append([row["rel"]])
    X_test = np.array(X_test)

    X_test_ohe = ohe.transform(X_test)

    print(f"Extracted features for the test pairs...\n")

    # Calculate distances between train and test
    distances = pairwise_distances(X_test_ohe, X_train_ohe, metric="manhattan")
    sorted_ids = np.argsort(distances, axis=1)

    # Keep track of already seen relations between pairs, to exclude them from predictions
    cache_triples = get_filtering_cache(df_train, df_eval)
    cache_mapped = {}
    for (h, t), rels in cache_triples.items():
        try:
            cache_mapped[(node2id[h], node2id[t])] = [rel2id[rel] for rel in rels]
        except KeyError:
            continue

    # Iterate over the test pairs
    list_of_pairs = set(list(features_ij.keys()))

    res = []
    test_index = 0
    train_pairs = np.array(train_pairs)

    # Calculate the IDF of the relations on the train set
    idf = np.log(
        (len(df_train) / df_train["rel"].value_counts(ascending=True))
    ).to_dict()

    # For each test pair keep their most similar pair
    for i_test, similar_pairs in enumerate(sorted_ids[:, :sim_pairs]):
        cur_row = df_test_mapped.iloc[i_test]

        current_similar_pairs = train_pairs[similar_pairs, :]
        # remove the test pair from its list of most similar pairs, if it exists
        if (cur_row["head"], cur_row["tail"]) in list_of_pairs:
            similar_pairs = similar_pairs[1 : sim_pairs + 1]
        else:
            similar_pairs = similar_pairs[:sim_pairs]

        # Find the possible labels, according to the labels of the most similar pairs
        poss_labels = [label for id_ in similar_pairs for label in y_train[id_]]
        pred_labels = list(dict.fromkeys(poss_labels))

        # Rank them
        # Either by calculating their tf-idf weights (largest tf-idf weights on top)
        if selection_strategy == "tf-idf":
            tf = dict(Counter(poss_labels))
            for key, val in tf.items():
                tf[key] = tf[key] / len(poss_labels)
            tfidf = {}
            for rel_id, rel_freq in tf.items():
                tfidf[rel_id] = rel_freq * idf[id2rel[rel_id]]
            sorted_tfidf = dict(sorted(tfidf.items(), key=lambda item: item[1])[::-1])
            pred_labels = list(sorted_tfidf.keys())
            proba_labels = list(sorted_tfidf.values())
        # Or rank them by their term frequency (most frequent on top)
        elif selection_strategy == "most_common":
            predictions = [
                (label, count)
                for (label, count) in (Counter(poss_labels).most_common())[:sim_pairs]
            ]
            pred_labels = [int(pred[0]) for pred in predictions]
            proba_labels = [pred[1] for pred in predictions]
        else:
            raise KeyError(f"{selection_strategy} not understood")

        try:
            rank = pred_labels.index(int(cur_row["rel"])) + 1
        except ValueError:
            rank = len(unique_rels) + 1

        cur_res = {
            "predicted": pred_labels,
            "probas": proba_labels,
            "rank": rank,
            "similar_ij": current_similar_pairs,
            **cur_row,
        }
        res.append(cur_res)
    # break

    df_res = pd.DataFrame(res)
    pr_results = {}
    pr_results["MRR"] = (1 / df_res["rank"]).mean()

    print(f"\n #### RESULTS FOR {project_name} #######")
    print(f"MRR:{pr_results['MRR']:.4f}")

    for k in [1, 3, 10]:
        pr_results[f"h@{k}"] = (df_res["rank"] <= k).sum() / df_res.shape[0]
        print(f"Hits@{k}: {pr_results[f'h@{k}']:.4f}")
    total_results.append([project_name] + list(pr_results.values()))
    print(f"\n")


df = pd.DataFrame(total_results, columns=["dataset", "mrr", "h@1", "h@3", "h@10"])
print(df.to_string())
