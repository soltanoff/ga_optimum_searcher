#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
from functools import wraps
from itertools import combinations
from operator import itemgetter
from typing import Tuple, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm


def pretty(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        print("Phenotype: %s | "
              "X = %s | "
              "f(X) = %s (extr)" % (result[0], result[1][0], result[2]))
        return result

    return wrapper


class CGAOptimumSearcher:
    u"""Генетическая оптимизация функции двух переменных"""
    SEARCH_MAXIMUM = 0
    SEARCH_MINIMUM = 1
    RESULT_REPEAT_COUNT = 15

    def __init__(
            self,
            func,
            search_type=SEARCH_MAXIMUM,
            count_params=2,
            mut_prob=0.8,
            cross_prob=0.5,
            portion=0.2,
            max_pairs=1000,
            plot_enabled=False
    ):
        self.__func = func
        self.__population = []  # Current symbolic expression
        self.__mutation_probability = mut_prob  # type: float
        self.__crossover_probability = cross_prob  # type: float
        self.__portion = portion  # type: float
        self.__max_pairs = max_pairs  # type: int
        self.__count_params = count_params  # type: int
        self.__plot_enabled = plot_enabled  # type: bool
        self.__search_type = search_type  # type: int
        self.__left_border = 0  # type: int
        self.__right_border = 0  # type: int

        assert search_type in (self.SEARCH_MAXIMUM, self.SEARCH_MINIMUM), "Incorrect search type"

    # начальная случайная популяция
    def generate(self, size=30, left=9, right=1) -> None:
        self.__left_border, self.__right_border = right, left
        self.__population = (left * np.random.random_sample((size, self.__count_params)) + right).tolist()

    def plot(self, population, func, title: str, show=False) -> None:
        if self.__plot_enabled:
            if self.__count_params == 1:
                x = np.arange(self.__left_border, self.__right_border, 1)
                y = list(map(fitness_func, x))

                x_temp = [x[0] for x in population]
                y_temp = [func(*a) for a in population]
                plt.figure()
                plt.plot(x_temp, y_temp, 'go', x, y)
                plt.title(title)
                plt.show() if show else plt.draw()
            elif self.__count_params == 2:
                from mpl_toolkits.mplot3d import Axes3D

                def f(*args):
                    return func(*args[0])

                fig = plt.figure()
                ax = Axes3D(fig)

                x = np.arange(-500, 500, 10)
                y = np.arange(-500, 500, 10)

                x, y = np.meshgrid(x, y)
                z = np.fromiter(
                    map(f, zip(x.flat, y.flat)),
                    dtype=np.float,
                    count=x.shape[0] * x.shape[1]
                ).reshape(x.shape)

                x_temp = [x[0] for x in population]
                y_temp = [x[1] for x in population]
                z_temp = [func(*a) for a in population]

                ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap=cm.jet, linewidth=0.2)
                ax.plot(x_temp, y_temp, z_temp, 'go')

                plt.xlabel("x")
                plt.ylabel("y")

                plt.title(title)
                plt.show() if show else plt.draw()

    def search_optimum(
            self, population: List[List[float]]
    ) -> Tuple[int, float, float] or Tuple[int, List[float], float]:
        if self.__search_type == self.SEARCH_MAXIMUM:
            return self.__max(population)
        elif self.__search_type == self.SEARCH_MINIMUM:
            return self.__max(population)

    # основная функция "генетического" поиска
    def solve(self, iteration_count=1000) -> Tuple[int, float, float] or Tuple[int, List[float], float]:
        result = None
        prev_result = None
        result_repeat_count = 0

        self.plot(self.__population, self.__func, 'Первая популяция')

        for n in range(1, iteration_count + 1):
            print('Шаг:', n)
            ind = 0
            new_population = copy.copy(self.__population)
            for pair in combinations(range(len(self.__population)), 2):
                ind += 1
                if ind > self.__max_pairs:
                    break

                a = self.__mutate(self.__population[pair[0]])
                b = self.__mutate(self.__population[pair[1]])
                new_item = self.__crossover(a, b)
                new_population.append(new_item)

            self.__population = self.__reduction(new_population)

            result = pretty(self.search_optimum)(self.__population)
            result_repeat_count = self.__check_repeatable_result(result, prev_result, result_repeat_count)
            prev_result = result

            if result_repeat_count == self.RESULT_REPEAT_COUNT:
                break

            if n == 2:
                self.plot(self.__population, self.__func, 'Вторая популяция')

        self.plot(self.__population, self.__func, 'Последняя популяция', show=True)
        return result

    @staticmethod
    def __check_repeatable_result(result, prev_result, result_repeat_count):
        if result == prev_result:
            return result_repeat_count + 1
        else:
            return 0

    # обмен значениями (кроссинговер), происходит с вероятностью 0.5 по умолчанию
    def __crossover(
            self, a: List[float], b: List[float]
    ) -> List[float] or List[List[float], List[float]]:
        new_item = []
        for x in range(len(a)):
            if np.random.rand() > self.__crossover_probability:
                new_item.append(b[x])
            else:
                new_item.append(a[x])
        return new_item

    # мутация значений происходит с определенной вероятностью (0.8 - по умолчанию)
    def __mutate(self, a: List[float]) -> List[float]:
        if np.random.rand() < self.__mutation_probability:
            mutation_param = (np.random.rand(self.__count_params) - 0.05) * np.random.rand()
            new_a = a + mutation_param if np.random.rand() > 0.5 else a - mutation_param
        else:
            new_a = a
        return new_a

    def __reduction(self, population):
        res = np.argsort([self.__func(*item) for item in population])

        if self.__search_type == self.SEARCH_MAXIMUM:
            res = res[np.random.poisson(int(len(population) * self.__portion)):]
        elif self.__search_type == self.SEARCH_MINIMUM:
            res = res[:np.random.poisson(int(len(population) * self.__portion))]

        return np.array(population)[res].tolist()

    def __max(self, population: List[List[float]]) -> Tuple[int, float, float] or Tuple[int, List[float], float]:
        return max(
            [(i, params, self.__func(*params)) for i, params in enumerate(population)],
            key=itemgetter(2)
        )

    def __min(self, population: List[List[float]]) -> Tuple[int, float, float] or Tuple[int, List[float], float]:
        return min(
            [(i, params, self.__func(*params)) for i, params in enumerate(population)],
            key=itemgetter(2)
        )


if __name__ == '__main__':
    from math import cos, log


    def fitness_func(*args):
        return sum(map(lambda x: log(x) * cos(3 * x - 15), args))


    def fitness_func2(x, y):
        return log(x * y) * cos(3 * x - 15 * y)


    def fitness_func3(*args):
        return sum(map(lambda x: x ** 2, args))


    def fitness_func4(*args):
        return sum(map(lambda x: -x * np.sin(np.sqrt(abs(x))), args))


    g = CGAOptimumSearcher(func=fitness_func4,
                           search_type=CGAOptimumSearcher.SEARCH_MINIMUM,
                           count_params=2,
                           max_pairs=1000,
                           portion=0.8,
                           plot_enabled=True)
    g.generate(left=1000, right=-500)  # range x -> [-500; 500]
    g.solve(iteration_count=250)
