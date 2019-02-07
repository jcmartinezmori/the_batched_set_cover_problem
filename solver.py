import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import copy
from itertools import chain, combinations
rc('text', usetex=True)


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


def run_worst_case(max_m=64, max_vcd=6, mode='dedicated', epsilon=0.001):
    """
    runs the dedicated (or trivial) primal-dual algorithm for the batched set cover problem on the
    worst case instance identified in martinez mori and samarayanake, 2018
    :param max_m: maximum number of subsets in the problem instance
    :param max_vcd: maximum VC-dimension restriction in the problem instance
    :param mode: 'dedicated' for simultaneous dual variable update, 'trivial' for sequential dual variable update
    :param epsilon: discretization step of primal-dual algorithm
    :return:
    """

    # prepare basic set systems characterizing the worst case instance
    set_systems = {
        vcd: max_vcd_least_subsets_set_system(list(range(vcd))) for vcd in range(max_vcd)
    }

    # prepare and populate batches
    batches = {vcd: {} for vcd in range(max_vcd)}
    for m in range(max_m):
        for vcd, set_system in set_systems.items():

            # compute number of batches to be revealed, if any
            num_batches = max(0, m - 2 ** vcd + 1)
            if num_batches > 0:

                ct_elements = 0  # counter for elements revealed by the adversary
                batches[vcd][m] = []  # list of batches revealed by the adversary
                for num_batch in range(num_batches):

                    if mode == 'dedicated':  # batch will be processed simultaneously
                        batch = {}

                    # process elements in basic set system form (see fig. 1 of reference)
                    for element, subsets in set_system.items():
                        if mode == 'trivial':  # batch will be processed sequentially
                            batch = {}

                        # populate batch
                        batch[ct_elements] = [num_batch + j for j in subsets if j != 2 ** vcd - 1] \
                            + list(range(2 ** vcd - 1 + num_batch, m))
                        ct_elements += 1

                        if mode == 'trivial':  # artificially add element as its own batch
                            batches[vcd][m].append(batch)

                    # process element in intersection of basic set system form (see fig. 1 of reference)
                    if mode == 'trivial':
                        batch = {}
                    batch[ct_elements] = list(range(2 ** vcd - 1 + num_batch, m))
                    ct_elements += 1

                    # add entire batch in 'dedicated' mode, or simply the last batch in the 'trivial' mode
                    batches[vcd][m].append(batch)

    # storage
    results = {vcd: [] for vcd in range(max_vcd)}
    lower_bound = {vcd: [] for vcd in range(max_vcd)}

    for vcd in range(max_vcd):
        for m in range(max_m):

            # compute number of batches to be revealed, if any
            num_batches = max(0, m - 2 ** vcd + 1)
            if num_batches > 0:

                # initialize and run solver
                subsets = {j: [] for j in range(m)}
                model = initialize_model(copy.deepcopy(subsets))
                for batch in batches[vcd][m]:  # run solver
                    primal_dual_subroutine(model, batch, epsilon)

                # store
                results[vcd].append((m, model.primal_objective_value))
                lower_bound[vcd].append((m, sum(1 / i for i in range(1, m - 2 ** vcd + 1 + 1))))

    # plot
    markers = ['D', '^', 's', 'v', 'p', '+', 'o', '*']
    cmap = plt.get_cmap('Dark2')
    colors = [cmap(i) for i in np.linspace(0, 1, len(markers))]

    if mode == 'dedicated':
        y_label = r'$\textit{ALG}^{B,D}(I_z^*) / \textit{OPT}(I_z^*)$'
        file_name = 'alg-b-d.pdf'
    else:
        y_label = r'$\textit{ALG}^{B,T}(I_z^*) / \textit{OPT}(I_z^*)$'
        file_name = 'alg-b-t.pdf'

    plt.figure()
    plt.grid()
    plt.xlabel(r'$m = |\mathcal{S}|$')
    plt.ylabel(y_label)
    for vcd in range(max_vcd):
        color = colors[vcd]
        marker = markers[vcd]
        plt.plot(*zip(*lower_bound[vcd]), color=color, linestyle='--', linewidth=0.75)
        plt.plot(*zip(*results[vcd]), marker=marker, color=color, linestyle='-', markersize=3, label=r'$z \geq {0}$'.format(vcd))
    plt.legend()
    plt.ylim(0.8, 5.2)
    plt.savefig(file_name, format='pdf', bbox_inches='tight')


def max_vcd_least_subsets_set_system(ground_set):
    """
    prepares the set system of the maximum possible VC-dimension and the least number of subsets
    :param ground_set: list of elements in the ground set
    :return set_system: dictionary with
        key: element
        value: subsets in which element is contained
    """

    # all possible subsets of the ground set
    subsets = chain.from_iterable(combinations(ground_set, r) for r in range(len(ground_set) + 1))

    # prepare set system dictionary with
    #   key: element
    #   value: subsets in which element is contained
    set_system = {element: [] for element in ground_set}
    for subset_id, subset in enumerate(subsets):
        for element in subset:
            set_system[element].append(subset_id)

    return set_system


if __name__ == '_main__':
    pass
