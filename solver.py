import numpy as np


class Model:
    pass


def primal_dual_subroutine(model, batch, epsilon=0.001):
    """
    run batched primal-dual subroutine
    :param model: see initialize_model(*)
    :param batch: dictionary with
        key: element_id (e.g., e1)
        value: element_subset_ids (e.g., [S1, S3, S5])
    :param epsilon: discretization step size
    :return:
    """

    # constraint size
    d = model.m

    # store information in batch
    for element_id, element_subset_ids in batch.items():
        model.elements[element_id] = element_subset_ids
        model.y[element_id] = 0.0
        for subset_id in element_subset_ids:
            model.subsets[subset_id].append(element_id)

    # primal-dual subroutine
    while any(
                    sum(model.x[subset_id] for subset_id in element_subset_ids) < 1
                    for _, element_subset_ids in batch.items()
    ):

        for element_id, element_subset_ids in batch.items():
            if sum(model.x[subset_id] for subset_id in element_subset_ids) < 1:
                model.y[element_id] += epsilon

        for subset_id in model.subsets.keys():
            model.x[subset_id] = 1 / d * (
                np.exp(
                    np.log(1 + d)
                    / model.c[subset_id]
                    * sum(model.y[element_id] for element_id in model.subsets[subset_id])
                )
                - 1
            )

    model.primal_objective_value = sum(model.c[subset_id] * model.x[subset_id] for subset_id in model.subsets.keys())
    model.dual_objective_value = sum(model.y[element_id] for element_id in model.elements.keys())


def initialize_model(subsets, c=None):
    """
    initialize model structure
    :param subsets: dictionary of subsets with
        key: subset_id
        value: [] ; elements in subset subset_id
    :param c: dictionary of subsets with
        key: subset_id
        value: c_id  ; cost of bringing subset subset_id to set cover
    :return:
    """

    if c is None:  # construct unweighted instance
        c = dict()
        for subset_id in subsets.keys():
            c[subset_id] = 1
    else:  # test compatibility of weighted instance
        assert set(subsets.keys()) == set(c.keys())

    # empty structure
    model = Model()

    # store input
    model.subsets = subsets
    model.c = c
    model.elements = dict()  # collect elements received

    # store variables
    model.x = {subset_id: 0.0 for subset_id in model.subsets.keys()}
    model.y = dict()

    # store objective values
    model.primal_objective_value = 0.0
    model.dual_objective_value = 0.0

    # store auxiliary information
    model.m = len(subsets)

    return model


if __name__ == '_main__':
    pass
