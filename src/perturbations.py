import copy

def reconstruct_solution_by_exchange(problem, existing_solution, paths_ruined):
    path0 = copy.deepcopy(existing_solution[paths_ruined[0]])
    path1 = copy.deepcopy(existing_solution[paths_ruined[1]])
    num_exchanged = 0
    for i in range(1, len(path0) - 1):
        for j in range(1, len(path1) - 1):
            if problem.get_capacity(path0[i]) == problem.get_capacity(path1[j]):
                #TODO
                if problem.get_distance(path0[i], path1[j]) < 0.2:
                    path0[i], path1[j] = path1[j], path0[i]
                    num_exchanged += 1
                    break
    if num_exchanged >= 0:
        return [path0, path1]
    else:
        return []


