import numpy as np
import scipy.cluster
from tqdm.auto import tqdm
import multiprocessing as mp
import math
from .games import GameConfig


def welfare(R, N_AGENTS, welfareType="AVERAGE"):
    if welfareType == "AVERAGE":
        return R.sum() / N_AGENTS
    elif welfareType == "MIN":
        return R.min()
    elif welfareType == "MAX":
        return R.max()
    else:
        raise "SPECIFY WELFARE TYPE"


def count_groups(q_values, dist):
    y = scipy.cluster.hierarchy.average(q_values)
    z = scipy.cluster.hierarchy.fcluster(y, dist, criterion='distance')
    groups = np.bincount(z)
    return len(groups)


# def calculate_alignment(q_table):
#     argmax_q_table = np.argmax(q_table, axis=2)
#     return (argmax_q_table == np.broadcast_to(np.arange(q_table.shape[2]), (q_table.shape[0], q_table.shape[1]))).mean(axis=0)


def calculate_alignment(q_table, recommendation, actions):
    argmax_q_table = np.argmax(q_table, axis=2)
    belief_alignment = (argmax_q_table == np.broadcast_to(np.arange(q_table.shape[2]),
                                                          (q_table.shape[0], q_table.shape[1]))).mean(axis=0)
    recommendation_alignment = (recommendation == argmax_q_table[np.arange(q_table.shape[0]), recommendation]).mean()
    action_alignment = (recommendation == actions).mean()
    return belief_alignment, recommendation_alignment, action_alignment


def run_apply_async_multiprocessing(func, argument_list, num_processes=None):
    if num_processes:
        pool = mp.Pool(processes=num_processes)
    else:
        pool = mp.Pool(processes=mp.cpu_count())

    jobs = [
        pool.apply_async(func=func, args=(*argument,)) if isinstance(argument, tuple) else pool.apply_async(func=func,
                                                                                                            args=(
                                                                                                                argument,))
        for argument in argument_list]
    pool.close()
    result_list_tqdm = []
    for job in tqdm(jobs):
        result_list_tqdm.append(job.get())

    return result_list_tqdm
