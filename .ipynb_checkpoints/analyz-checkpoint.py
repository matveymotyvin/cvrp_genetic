import os
import json
import numpy as np
import matplotlib.pyplot as plt


def load(file):
    with open(file) as fr:
        return json.load(fr)


def plot_hist(data, name, task_type):
    plt.figure(figsize=(15, 10))
    plt.title(name, fontdict={'fontsize': 20})
    plt.hist(data, color='red')
    plt.show()
    plt.savefig(f'docs/{task_type}_histogram_100_000.png')


def get_errors(data):
    errors = []
    for sol in data:
        errors.append((abs(sol['optimal'] - sol['cost']) / float(sol['optimal'])) * 100)
    return errors


def get_time(data):
    time = []
    for sol in data:
        time.append(float(sol['time'].split()[0]))
    return time


if __name__ == "__main__":
    data = {}
    for task_type in ["A", "B", "E"]:
        data[task_type] = load(os.path.join('results', f"{task_type}_output_100_000.json"))

    for task_type in ["A", "B", "E"]:
        print(task_type)
        errors = get_errors(data[task_type].values())
        time = get_time(data[task_type].values())
        avg_time = np.mean(time)
        print(f'avg time: {avg_time}')
        count = len(errors)
        avg_error = np.mean(errors)
        print(f'avg er: {avg_time}')
        std_error = np.std(errors)
        print(f'std: {avg_time}')
        median_error = np.median(errors)
        print(f'med: {avg_time}')
        plot_hist(errors,
                  f"{task_type} tasks, Avg. Time = {round(avg_time, 2)}\n"
                  f" Avg. Error = {round(avg_error, 2)}%, Median Error = {round(median_error, 2)}%, Std Error = {std_error}%", task_type)
        

    all_data = []
    for key, d in data.items():
        for tname, t_d in d.items():
            all_data.append(t_d)

    def filter_by_size(data, concrete_filter):
        return list(filter(concrete_filter, data))

    def get_size(data):
        return int(data['name'].split('-')[1].strip('n'))

    small_data = filter_by_size(all_data, lambda x: get_size(x) <= 30)
    medium_data = filter_by_size(all_data, lambda x: 30 < get_size(x) <= 60)
    large_data = filter_by_size(all_data, lambda x: get_size(x) > 60)

    for task_type, data in [("Small (<= 30)", small_data), ("Medium (30<, <=60)", medium_data), ("Large (> 60)", large_data)]:
        errors = get_errors(data)
        time = get_time(data)
        avg_time = np.mean(time)
        count = len(errors)
        avg_error = np.mean(errors)
        std_error = np.std(errors)
        median_error = np.median(errors)
        plot_hist(errors,
                  f"{task_type} tasks, Avg. Time = {round(avg_time, 2)}\n"
                  f" Avg. Error = {round(avg_error, 2)}%, Median Error = {round(median_error, 2)}%, Std Error = {std_error}%"
                  , task_type)

    size = [get_size(d) for d in all_data]
    time = get_time(all_data)
    idxs = np.argsort(size)

    plt.figure(figsize=(15, 10))
    plt.title("Time (node amount) dependency", fontdict={'fontsize': 20})
    plt.xlabel("Nodes", fontdict={'fontsize': 20})
    plt.ylabel("Time, seconds", fontdict={'fontsize': 20})
    plt.plot(np.array(size)[idxs], np.array(time)[idxs], color='red')
    plt.show()
    plt.savefig(f'docs/Time histogram_100000.png')