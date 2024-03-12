import enum
import math
import random as r
import time
from typing import Dict, Tuple, List, Union

import matplotlib.pyplot as plt

from ocvrp import algorithms as alg
from ocvrp.util import Building, Individual, OCVRPParser


class ReplStrat(enum.Enum):
    RAND = enum.auto()
    BEST = enum.auto()
    WORST = enum.auto()


class CVRP:

    def __init__(self, problem_set_path: str,
                 population_size: int = 800,
                 selection_size: int = 5,
                 ngen: int = 100_000,
                 mutpb: float = 0.15,
                 cxpb: float = 0.85,
                 cx_algo=alg.best_route_xo,
                 mt_algo=alg.swap_mut,
                 pgen: bool = False,
                 agen: bool = False,
                 plot: bool = False,
                 verbose_routes: bool = False):
        print("Loading problem set...")
        ps_strat = OCVRPParser(problem_set_path).parse()

        self._problem_set_name = ps_strat.get_ps_name()
        self._problem_set_comments = ps_strat.get_ps_comments()
        self._vehicle_cap = ps_strat.get_ps_capacity()
        self._optimal_fitness = ps_strat.get_ps_optimal()
        self._dim = ps_strat.get_ps_dim()
        self._depot = ps_strat.get_ps_depot()
        self._problem_set_buildings_orig = ps_strat.get_ps_buildings()
        self._pop = []

        self.population_size = population_size
        self.selection_size = selection_size
        self.ngen = ngen
        self.mutpb = mutpb
        self.cxpb = cxpb
        self.cx_algo = cx_algo
        self.mt_algo = mt_algo
        self.pgen = pgen
        self.agen = agen
        self.plot = plot
        self.verbose_routes = verbose_routes

        # Create n random permutations from the problem set
        self.reset()

    def calc_fitness(self, individual):
        distance = 0
        partitioned_routes = self.partition_routes(individual)
        for _, route in partitioned_routes.items():
            for i in range(len(route) - 1):
                h1, h2 = route[i], route[i + 1]
                distance += Building.distance(h1, h2)
            distance += Building.distance(self._depot, route[0])
            distance += Building.distance(route[len(route) - 1], self._depot)

        return distance

    def partition_routes(self, individual: Individual) -> Dict:
        routes = {}
        current_weight = 0
        route_counter = 1

        for building in individual:
            if route_counter not in routes:
                routes[route_counter] = []

            if current_weight + building.quant > self._vehicle_cap:
                route_counter += 1
                current_weight = 0
                routes[route_counter] = []

            routes[route_counter].append(building)
            current_weight += building.quant

        return routes

    @staticmethod
    def de_partition_routes(partitioned_routes: Dict) -> List:
        ll = []
        for v in partitioned_routes.values():
            ll.extend(v)
        return ll

    def select(self) -> Tuple[Individual, Individual]:
        # take_five is the mating pool for this generation
        take_five = r.sample(self._pop, self._selection_size)

        parent1 = self._get_and_remove(take_five, ReplStrat.BEST)
        parent2 = self._get_and_remove(take_five, ReplStrat.BEST)

        return parent1, parent2

    def replacement_strat(self, individual: Individual, rs) -> None:
        self._get_and_remove(self._pop, rs)
        self._pop.append(individual)

    @staticmethod
    def _get_nworst(sel_values, n):
        v = sel_values[:]
        v.sort()
        return v[-n:]

    @staticmethod
    def _get_and_remove(sel_values, rs):
        if rs == rs.RAND:
            val = r.choice(sel_values)
        elif rs == rs.BEST:
            val = min(sel_values)
        else:
            val = max(sel_values)
        sel_values.remove(val)
        return val

    def reset(self):
        self._pop = []
        for _ in range(self._population_size):
            rpmt = r.sample(self._problem_set_buildings_orig, self._dim)
            self._pop.append(Individual(rpmt, self.calc_fitness(rpmt)))

    def run(self) -> dict:
        print(f"Running {self._ngen} generation(s)...")

        best_data, avg_data = [], []

        # The bound at which we start diversity maintenance
        div_thresh_lb = math.ceil(0.01 * self._population_size)

        # The amount of individuals to replace for diversity maintenance
        div_picking_rng = round(0.75 * self._population_size)

        t = time.process_time()
        found = False
        indiv = None

        # Start the generation count
        for i in range(1, self._ngen + 1):

            mut_prob = r.choices([True, False], weights=(self._mutpb, 1 - self._mutpb), k=1)[0]
            cx_prob = r.choices([True, False], weights=(self._cxpb, 1 - self._cxpb), k=1)[0]

            parent1, parent2 = self.select()
            if cx_prob:
                if self._cx_algo == 'best_route_xo':
                    child1 = alg.best_route_xo(parent1, parent2, self)
                    child2 = alg.best_route_xo(parent2, parent1, self)
                elif self._cx_algo == 'cycle_xo':
                    cxo = alg.cycle_xo(parent1, parent2, self)
                    child1 = cxo['o-child']
                    child2 = cxo['e-child']
                elif self._cx_algo == 'edge_recomb_xo':
                    child1 = alg.edge_recomb_xo(parent1, parent2)
                    child2 = alg.edge_recomb_xo(parent2, parent1)
                else:
                    child1 = alg.order_xo(parent1, parent2)
                    child2 = alg.order_xo(parent2, parent1)
            else:
                child1 = parent1
                child2 = parent2

            if mut_prob:
                if self._mt_algo == 'inversion_mut':
                    child1 = alg.inversion_mut(child1)
                    child2 = alg.inversion_mut(child2)
                elif self._mt_algo == 'swap_mut':
                    child1 = alg.swap_mut(child1)
                    child2 = alg.swap_mut(child2)
                else:
                    child1 = alg.gvr_scramble_mut(child1, self)
                    child2 = alg.gvr_scramble_mut(child2, self)

            if child1.fitness is None:
                child1.fitness = self.calc_fitness(child1)

            if child2.fitness is None:
                child2.fitness = self.calc_fitness(child2)

            # One of the children were found to have an optimal fitness, so I'll save that
            if child1.fitness == self._optimal_fitness or child2.fitness == self._optimal_fitness:
                indiv = child1 if child1.fitness == self._optimal_fitness else child2
                found = True
                break

            self.replacement_strat(child1, ReplStrat.WORST)
            self.replacement_strat(child2, ReplStrat.WORST)

            if self._pgen:
                print(f'GEN: {i}/{self._ngen}', end='\r')

            uq_indv = len(set(self._pop))

            min_indv, max_indv, avg_fit = None, None, None

            if i % 250 == 0 or i == 1:
                if self._agen:
                    min_indv = min(self._pop).fitness
                    max_indv = max(self._pop).fitness
                    avg_fit = round(sum(self._pop) / self._population_size)

                    print(f"UNIQUE FITNESS CNT: {uq_indv}/{self._population_size}")
                    print(f"GEN {i} BEST FITNESS: {min_indv}")
                    print(f"GEN {i} WORST FITNESS: {max_indv}")
                    print(f"GEN {i} AVG FITNESS: {avg_fit}\n\n")

                if self._plot:
                    min_indv = min(self._pop).fitness if min_indv is None else min_indv
                    best_data.append(min_indv)

                    avg_fit = round(sum(self._pop) / self._population_size) if avg_fit is None else avg_fit
                    avg_data.append(avg_fit)

            if i % 10000 == 0 and uq_indv <= div_thresh_lb:
                print("===============DIVERSITY MAINT===============") if self._agen else None
                worst = self._get_nworst(self._pop, div_picking_rng)

                for k in range(div_picking_rng):
                    c = min(self._pop)
                    if self._mt_algo == 'inversion_mut':
                        rsamp = alg.inversion_mut(c)
                    elif self._mt_algo == 'swap_mut':
                        rsamp = alg.swap_mut(c)
                    else:
                        rsamp = alg.gvr_scramble_mut(c, self)
                    i = Individual(rsamp, self.calc_fitness(rsamp))

                    self._pop.remove(worst[k])
                    self._pop.append(i)

        # Find the closest value to the optimal fitness (in case we don't find a solution)
        closest = min(self._pop)
        end = time.process_time() - t

        return self._create_solution(indiv if found else closest, end, best_data, avg_data)

    def _create_solution(self, individual, comp_time, best_data, avg_data) -> dict:


        if self._plot:
            plt.figure(figsize=(10, 9), dpi=200)
            plt.plot(best_data, linestyle="solid", label="Best Fitness Value")
            plt.plot(avg_data, linestyle="solid", label="Average Fitness Value")
            plt.title(f'{self._cx_algo}_{self._ngen}_{self._cxpb}_{self._problem_set_name}__graph')
            plt.legend(loc='upper right')
            plt.xlabel("Generations (x250)")
            plt.ylabel("Fitness")

        partitioned = self.partition_routes(individual)

        obj = {
            "name": type(self).__name__,
            "problem_set_name": self._problem_set_name,
            "problem_set_optimal": self._optimal_fitness,
            "time": f"{comp_time} seconds",
            "vehicles": len(partitioned.keys()),
            "vehicle_capacity": self._vehicle_cap,
            "dimension": self._dim,
            "population_size": self._population_size,
            "selection_size": self._selection_size,
            "generations": self._ngen,
            "cxpb": self._cxpb,
            "mutpb": self._mutpb,
            "cx_algorithm": self._cx_algo,
            "mut_algorithm": self._mt_algo,
            "best_individual_fitness": individual.fitness,
        }

        if self._plot:
            obj["mat_plot"] = plt

        if self._verbose_routes:
            obj["best_individual"] = partitioned

        return obj

    @property
    def problem_set_name(self) -> float:
        return self._problem_set_name

    @property
    def problem_set_comments(self) -> Union[float, None]:
        return self._problem_set_comments

    @property
    def vehicle_cap(self) -> float:
        return self._vehicle_cap

    @property
    def optimal_fitness(self) -> float:
        return self._optimal_fitness

    @property
    def dim(self) -> float:
        return self._dim

    @property
    def depot(self) -> Building:
        return self._depot

    @property
    def pop(self) -> List[Individual]:

        return self._pop

    @property
    def population_size(self) -> int:

        return self._population_size

    @population_size.setter
    def population_size(self, population_size: int) -> None:

        self._is_int_ge(population_size, 5)
        self._population_size = population_size

    @property
    def selection_size(self) -> int:

        return self._selection_size

    @selection_size.setter
    def selection_size(self, selection_size: int) -> None:

        self._is_int_ge(selection_size, 1)
        self._selection_size = selection_size

    @property
    def ngen(self) -> int:
        return self._ngen

    @ngen.setter
    def ngen(self, ngen: int) -> None:
        self._is_int_ge(ngen, 1)
        self._ngen = ngen

    @property
    def mutpb(self) -> float:
        return self._mutpb

    @mutpb.setter
    def mutpb(self, mutpb: float) -> None:
        self._is_probability(mutpb)
        self._mutpb = mutpb

    @property
    def cxpb(self) -> float:
        return self._cxpb

    @cxpb.setter
    def cxpb(self, cxpb: float) -> None:
        self._is_probability(cxpb)
        self._cxpb = cxpb

    @property
    def cx_algo(self) -> str:
        return self._cx_algo

    @cx_algo.setter
    def cx_algo(self, cx_algo) -> None:
        self._cx_algo = cx_algo.__name__

    @property
    def mt_algo(self) -> str:
        return self._mt_algo

    @mt_algo.setter
    def mt_algo(self, mt_algo) -> None:
        self._mt_algo = mt_algo.__name__

    @property
    def pgen(self) -> bool:
        return self._pgen

    @pgen.setter
    def pgen(self, pgen: bool) -> None:
        self._is_bool(pgen)
        self._pgen = pgen

    @property
    def agen(self) -> bool:
        return self._agen

    @agen.setter
    def agen(self, agen: bool) -> None:
        self._is_bool(agen)
        self._agen = agen

    @property
    def plot(self) -> bool:
        return self._plot

    @plot.setter
    def plot(self, plot: bool) -> None:
        self._is_bool(plot)
        self._plot = plot

    @property
    def verbose_routes(self) -> bool:
        return self._verbose_routes

    @verbose_routes.setter
    def verbose_routes(self, verbose_routes: bool) -> None:
        self._is_bool(verbose_routes)
        self._verbose_routes = verbose_routes

    @staticmethod
    def _is_probability(value):
        if (not isinstance(value, int)) and (not isinstance(value, float)):
            raise AttributeError("Probability must be numeric")

        if not 0 <= value <= 1:
            raise ValueError('Value must be >= 0 and <= 1')

    @staticmethod
    def _is_int_ge(value: int, ge: int):
        if not isinstance(value, int):
            raise AttributeError("value must be float")

        if not value >= ge:
            raise ValueError('Value must be >= 1')

    @staticmethod
    def _is_bool(value: bool):
        if not isinstance(value, bool):
            raise ValueError("Value must be bool")
