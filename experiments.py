import copy
import matplotlib.pyplot as plt
import numpy as np
from itertools import chain, combinations
from matplotlib import rc
from solver import primal_dual_subroutine, initialize_model
rc('text', usetex=True)
protocol = 2


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


if __name__ == '__main__':

    run_worst_case(mode='trivial')
    run_worst_case(mode='dedicated')
