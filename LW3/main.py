import numpy as np
import time
import multiprocessing as mp
import matplotlib.pyplot as plt

# Ядро Епанечникова
def kernel(u):
    u = np.asarray(u)
    return np.where(np.abs(u) <= 1.0, 0.75 * (1.0 - u**2), 0.0)

# Рабочие функции
def worker_numerator(weights, y_train, chis_shared):
    chis_shared.value = float(np.sum(weights * y_train))

def worker_denominator(weights, znam_shared):
    znam_shared.value = float(np.sum(weights))

# 1 процесс — Python-цикл (эталон)
def sequential_num_den(weights, y_train):
    start = time.perf_counter()
    chis = 0.0
    for i in range(len(weights)):
        chis += weights[i] * y_train[i]
    znam = 0.0
    for i in range(len(weights)):
        znam += weights[i]
    return float(chis), float(znam), time.perf_counter() - start

# 2 процесса: числитель и знаменатель параллельно
def parallel_2proc(weights, y_train):
    chis_shared = mp.Value('d', 0.0)
    znam_shared = mp.Value('d', 0.0)
    p1 = mp.Process(target=worker_numerator,   args=(weights, y_train, chis_shared))
    p2 = mp.Process(target=worker_denominator, args=(weights, znam_shared))
    start = time.perf_counter()
    p1.start(); p2.start()
    p1.join();  p2.join()
    return chis_shared.value, znam_shared.value, time.perf_counter() - start

# 4 процесса: числитель и знаменатель по двум половинам массива
def parallel_4proc(weights, y_train):
    mid = len(weights) // 2
    chis1, chis2 = mp.Value('d', 0.0), mp.Value('d', 0.0)
    znam1, znam2 = mp.Value('d', 0.0), mp.Value('d', 0.0)
    p1 = mp.Process(target=worker_numerator,   args=(weights[:mid], y_train[:mid], chis1))
    p2 = mp.Process(target=worker_numerator,   args=(weights[mid:], y_train[mid:], chis2))
    p3 = mp.Process(target=worker_denominator, args=(weights[:mid], znam1))
    p4 = mp.Process(target=worker_denominator, args=(weights[mid:], znam2))
    start = time.perf_counter()
    for p in (p1, p2, p3, p4): p.start()
    for p in (p1, p2, p3, p4): p.join()
    return chis1.value + chis2.value, znam1.value + znam2.value, time.perf_counter() - start

# Замер среднего времени по сетке x_grid
def measure(x, y, beta, delta, x_grid):
    times_1, times_2, times_4 = [], [], []
    h = delta / beta
    for x0 in x_grid:
        weights = kernel((x0 - x) / h)
        if np.sum(weights) == 0:
            continue
        _, _, t1 = sequential_num_den(weights, y)
        _, _, t2 = parallel_2proc(weights, y)
        _, _, t4 = parallel_4proc(weights, y)
        times_1.append(t1)
        times_2.append(t2)
        times_4.append(t4)
    return np.mean(times_1), np.mean(times_2), np.mean(times_4)

if __name__ == "__main__":
    print("CPU:", mp.cpu_count())

    np.random.seed(69)
    n = 5_000_000          # большой n — иначе параллелизм не даст выигрыша
    x = np.random.uniform(-3, 7, n)
    noise_std = 0.1 + 0.2 * np.abs(np.sin(x))
    y = (0.1*(x-4)*np.cos(x) + 0.5*np.sin(1.5*x)
         + 0.2*np.cos(0.5*x**2) + 0.05*x**2
         + np.random.normal(0, noise_std, n))

    sorted_x = np.sort(x)
    delta = np.max(np.diff(sorted_x))
    beta  = 1.0
    x_grid = np.linspace(-3, 7, 20)

    t1, t2, t4 = measure(x, y, beta, delta, x_grid)
    s2, s4 = t1 / t2, t1 / t4

    print(f"T1: {t1:.6f} с")
    print(f"T2: {t2:.6f} с  (S2={s2:.3f})")
    print(f"T4: {t4:.6f} с  (S4={s4:.3f})")

    # График времени
    plt.figure()
    plt.bar(["1 процесс", "2 процесса", "4 процесса"],
            [t1, t2, t4], color=["#4fc3f7", "#81c784", "#f06292"])
    plt.ylabel("Время (сек)")
    plt.title("Сравнение времени выполнения")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.show()

    # График ускорения
    plt.figure()
    plt.bar(["2 процесса", "4 процесса"], [s2, s4],
            color=["#81c784", "#f06292"])
    plt.axhline(1, color="gray", linestyle="--", linewidth=0.8)
    plt.ylabel("Ускорение")
    plt.title("Ускорение относительно 1 процесса")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.show()