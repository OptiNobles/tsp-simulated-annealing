from scipy.spatial import distance
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 8})
plt.rcParams['figure.facecolor'] = 'dimgrey'
import numpy as np
from random import choice, random, randint
from itertools import combinations
import time 

# Helper functions ------------------
def percentage(improved, best):
  percentage = round(100 * abs(improved-best)/best, 3)
  return percentage

def time_passed_from(start):
    time_passed = time.time() - start
    return time_passed
# -----------------------------------

class SimulatedAnnealing:
    def __init__(self, start_temperature = 100.0, end_temperature = 1e-8, alpha = 0.995, beta = 2.0):
        
        """
        Simulated Annealing Optimizer. Finds the path that minimizes distance traveled between nodes.
        This optimizer is devoted to solving Traveling Salesman Problem (TSP).

        It consists of 3 stages:
            - Stage 1: Greedy algorithm provides an initial solution,
            - Stage 2: Simulated Annealing searches solutions space,
            - Stage 3: 2-Opt algorithm improves the solution obtained in stage 2. (optional)

        :param start_temperature: System's initial temperature.
        :param end_temperature: Temperature at which the system reaches steady-state. 
        :param alpha: Temperature decrease rate (exponential cooling mode).
        :param beta: Temperature decrease rate (polynomial cooling mode).

        Created by Radosław Sergiusz Jasiewicz. Enjoy :)
        """

        # Parameters
        self.start_temperature = start_temperature
        self.end_temperature = end_temperature
        self.alpha = alpha
        self.beta = beta

        # Matrices 
        self.distance_matrix = None

        # Nodes
        self.list_of_nodes = None
        self.num_nodes = None
        self.nodes = None

        # Iterations
        self.end_iteration = None
        self.iterations = None
        self.temperature = None

        # Series
        self.temperature_series = []
        self.solution_series = []
        self.score_series = []

        # Best stats
        self.best_solution = None
        self.best_score = 0

        # Fit stats
        self.fit_time = 0
        self.fitted = False

        # Improvement (%)
        self.improvement = 0
        self.improved = False

    def __str__(self):

        string = "Simulated Annealing Optimizer\n"
        string += "----------------------------\n"
        string += "Designed to solve travelling salesman problem. Optimizes the minimum distance travelled.\n"
        string += "----------------------------\n"
        string += f"Start temperature:\t\t{self.start_temperature}\n"
        string += f"End temperature:\t\t{self.end_temperature}\n"
        string += f"Alpha:\t\t\t\t{self.alpha}\n"
        string += f"Beta:\t\t\t\t{self.beta}\n"
        string += "----------------------------\n"

        if self.fitted:
            string += "\n\nThis optimizer has been fitted."
        else:
            string += "\n\nThis optimizer has NOT been fitted."
        return string

    def __nearest_node_to(self, that_node: int):
        """
        Finds the nearest node to a given node.
        """
        nearest_node = None
        nearest_distance = None

        for other_node in self.list_of_nodes:

            distance = self.distance_matrix[that_node, other_node]

            if nearest_node == None:
                nearest_node = other_node
                nearest_distance = distance
            elif distance < nearest_distance:
                nearest_node = other_node
                nearest_distance = distance

        return nearest_node

    def __greedy_algorithm(self):
        """
        Provides an initial solution.

        Starting at an arbitrary node it starts a path containing the smallest-weight edge from that node, 
        and then continuously extends this path by appending the smallest-weight edge from the last node 
        on the path to an unvisited node, until all nodes have been visited.
        """
        solution = [0] * (self.num_nodes + 1)
        
        current_node = choice(self.list_of_nodes)
        start_node = current_node
        node = 0

        while True:
            solution[node] = current_node
            self.__remove_node(current_node)
            if len(self.list_of_nodes) != 0:
                next_node = self.__nearest_node_to(current_node)
                current_node = next_node
                node += 1
            else:
                break

        solution[node + 1] = start_node
        self.__reset_list_of_nodes()

        return solution

    def __2_opt_algorithm(self, improvement_threshold = 0.005, max_time = 20):
        """
        IMP. Starts with the best solution SAO has provided. It improves the solution incrementally by exchanging 2 nodes in the path with two other nodes.
        This process is repeated until the necessary precision is reached. The whole process is constructed in such a way that if the necessary precision
        is reached IMP updates SAO's best solution, otherwise it keep searching, but for no more than 10% of the maximum time. By default this function's runtime
        is set to 20 seconds.

        :param improvement_threshold: Desired improvement of the best solution at each iteration.
        :param max_time: 2-Opt algorithm maximum working time.
        """
        def __2_opt_swap(u, v):
            solution = self.best_solution.copy()
            solution[u:v] = solution[(v-1):(u-1):-1]
            solution_score = self.__energy(solution)
            return solution, solution_score

        start = time.time()
        max_search_time = max_time // 10
        score_to_beat = self.best_score.copy()

        while time_passed_from(start) < max_time:

            improved = False
            improved_score = 0
            improved_solution = None
            improvement_factor = 1
            iteration_start = time.time()

            while (improvement_factor > improvement_threshold) and (time_passed_from(iteration_start) < max_search_time):

                for u in range(1, len(self.best_solution) - 2):
                    for v in range(u + 1, len(self.best_solution)):
                        if v - u == 1:
                            continue

                        solution, solution_score = __2_opt_swap(u, v)

                        if solution_score < self.best_score:
                            improved = True
                            improved_score = solution_score
                            improved_solution = solution

                improvement_factor = 1 - (improved_score/self.best_score)

            if improved:
                self.improvement = percentage(improved_score, score_to_beat)
                self.best_score = improved_score
                self.best_solution = improved_solution

        if self.improvement > 0:
            self.improved = True

        if not self.improved: 
            print("IMP runtime exceeded. Solution has NOT been improved.")
        elif self.improved:
            print(f"IMP runtime exceeded. Solution has been improved by {self.improvement}%.\n")
        print(f'Runtime: {round(time_passed_from(start))} second(s).\nFinal score: {round(self.best_score, 2)}')
                
    def __remove_node(self, node: int):
        """
        Removes the node from the list of nodes.
        """
        self.list_of_nodes.remove(node)

    def __reset_list_of_nodes(self):
        """
        Resets the list of nodes for the next iteration.
        """
        self.list_of_nodes = list(range(0, self.num_nodes))

    def __energy(self, solution: list) -> float:
        """
        Evaluates the solution's energy.
        """
        energy = 0

        for node in range(len(solution) - 1):
            energy += self.distance_matrix[solution[node], solution[node+1]]
        
        return energy

    def __probability(self, solution: list):
        """
        Calculates the probability of accepting the potential solution based on two things: 

            - the difference in energy of the potential solution and current solution,
            - system's current temperature. 
        """
        current_energy = self.__energy(solution)
        previous_energy = self.__energy(self.solution_series[-1])

        return np.exp(-(current_energy-previous_energy)/self.temperature)

    def __evaluate(self, potential_solution: list, current_score = -1):
        """
        The function is responsible for choosing a solution. If a potential solution's score is smaller (means it's better) than the lastly obtained solution's score - it is accepted immediately. 
        Otherwise, the probability of picking that particular solution (p) is evaluated and if p is satisfied, that solution is accepted (even if it is a worse solution).
        If a potential solution isn't better nor p is satisfied - the current solution is preserved and passed to the next iteration. 

        :param potential_solution: A solution transformed by __vibrate() method. 
        """
        potential_solution_score = self.__energy(potential_solution)

        choosen_solution = None
        choosen_solution_score = 0

        if potential_solution_score < self.score_series[current_score]:
            choosen_solution = potential_solution
            choosen_solution_score = potential_solution_score
            if potential_solution_score < self.best_score:
                self.best_solution = potential_solution
                self.best_score = potential_solution_score

        elif self.__probability(potential_solution) > random():
            choosen_solution = potential_solution
            choosen_solution_score = potential_solution_score

        else:
            choosen_solution = self.solution_series[current_score]
            choosen_solution_score = self.score_series[current_score]

        return choosen_solution, choosen_solution_score

    def __vibrate(self, solution: list):
        """
        Makes a slice of random length inside the solution's interior and reverses it.
        """
        vibrated_solution = solution[:-1]

        i = randint(2, len(vibrated_solution) - 1)
        j = randint(0, len(vibrated_solution) - i)

        vibrated_solution[i : (i + j)] = reversed(vibrated_solution[i : (i + j)])
        vibrated_solution.append(vibrated_solution[0])

        return vibrated_solution

    def __cool_down(self, iteration: int, cooling: str):
        """
        Guides how the system's temperature decreases.
        """
        if cooling == 'exponential':
            self.temperature *= self.alpha 

        if cooling == 'polynomial':
            self.temperature = self.start_temperature * (1 - (iteration/self.iterations))**self.beta

    def __initialize(self, nodes: list):
        """
        Initializes a distance matrix and checks given list of nodes. 
        It also generates initial solution using greedy algorithm.
        """
        for node in nodes:
            assert len(node) == 2, "These are not valid points! Maybe check them? :)"

        # Create distance matrix 
        self.nodes = nodes
        self.num_nodes = len(nodes)
        self.distance_matrix = distance.cdist(nodes, nodes)

        # Fill the list of nodes 
        self.list_of_nodes = list(range(0, self.num_nodes))

        # Initial solution
        initial_solution = self.__greedy_algorithm() ; self.solution_series.append(initial_solution)
        initial_score = self.__energy(initial_solution) ; self.score_series.append(initial_score)

        self.best_score = initial_score ; self.best_solution = initial_solution
        
    def fit(self, nodes: list, iterations = 1000, cooling = 'exponential', improve = False, verbose = True):
        """
        The core function of the optimizer. It fits SAO to a specific map.

        :param nodes: List of positions (x, y) of the nodes.
        :param iterations: Number of iterations.
        :param cooling: Cooling mode. (exponential or polynomial)
        :param improve: If enabled 2-Opt (IMP) improves the solution obtained by SAO. (optional)
        :param verbose: If enabled SAO informs you about the progress.
        """
        start = time.time()

        self.iterations = iterations + 1
        self.temperature = self.start_temperature
        self.__initialize(nodes)

        if verbose: print(f'{self.num_nodes} nodes were given. Beginning SAO optimization with {iterations} itarations...\n')

        iteration = 1 ; current_solution = -1

        while (self.temperature > self.end_temperature) and (iteration < self.iterations):

            solution = self.solution_series[current_solution]
            vibrated_solution = self.__vibrate(solution)

            accepted_solution, accepted_score = self.__evaluate(vibrated_solution)

            self.temperature_series.append(self.temperature)
            self.solution_series.append(accepted_solution)
            self.score_series.append(accepted_score)

            self.__cool_down(iteration, cooling)
            iteration += 1

        self.end_iteration = iteration
        self.fitted = True
        self.fit_time = round(time_passed_from(start))

        if verbose: print(f"SAO fitted.\nRuntime: {self.fit_time // 60} minute(s)\nBest score: {round(self.best_score, 2)}\nCooling: {cooling}\n")

        if improve:
            if verbose: print("Beginning IMP (2-OPT) improvement...\n")
            self.__2_opt_algorithm()

    def show_graph(self, fitted = True, figsize = (10,5), dpi = 200):
        """
        Shows the graph of nodes that SAO is working on.

        :param fitted: If enabled it shows the best path that the optimizer has found.
        """
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        for node in self.nodes:
            ax.scatter(
                x = node[0],
                y = node[1],
                linewidth=0.5,
                marker="o",
                s=8,
                edgecolor="cyan",
                c="black",
                zorder=2
            )

        for node, other_node in combinations(self.nodes, r=2):
            ax.plot(
                [node[0], other_node[0]],
                [node[1], other_node[1]],
                color="blue",
                linewidth=0.2,
                alpha=0.05,
                zorder=0
            )

        if self.fitted and fitted:
            for node in range(len(self.best_solution) - 1):
                ax.plot(
                    [self.nodes[self.best_solution[node]][0], self.nodes[self.best_solution[node+1]][0]],
                    [self.nodes[self.best_solution[node]][1], self.nodes[self.best_solution[node+1]][1]],
                    color = "cyan",
                    linewidth = 0.5,
                    alpha = 1,
                    zorder = 1
                )

        ax.set_title(r"$\bf{Graph}$ $\bf{of}$ $\bf{nodes}$")
        ax.axis('off')
        plt.show()

    def plot(self, figsize = (10,5), dpi = 200):
        """
        Plots performance over iterations. If SAO has NOT been fitted returns None.
        """
        if not self.fitted:
            print("Simulated Annealing Optimizer not fitted! There is nothing to plot.")
            return None
        else:
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

            score = ax.plot(
                list(range(self.end_iteration)), 
                self.score_series,
                lw = 0.75,
                color = 'teal',
                zorder = 1,
                label = 'Performance (score)'
                )

            ax_temperature = ax.twinx()
            temperature = ax_temperature.plot(
                list(range(self.end_iteration - 1)),
                self.temperature_series,
                lw = 0.75,
                color = 'red',
                alpha = 0.25,
                zorder = 0,
                label = '    Temperature'
            )

            ax.set_xlabel(r'$\bf{Iteration}$ $\bf{number}$')
            ax.set_ylabel(r'$\bf{Performance}$')
            ax.set_xlim(0, self.end_iteration) ; ax_temperature.set_ylim(0, self.start_temperature)
            ax_temperature.axis('off')

            ax.set_facecolor('lightgrey')

            # Legend
            lns = temperature + score
            labs = [l.get_label() for l in lns]
            ax.legend(lns, labs, frameon = False)

            plt.show()

    def get_result(self):
        """
        :return: Tuple consisted of the best path, best distance, fit time, and list of each iteration's best distance.
        """
        return self.best_solution, self.best_score, self.fit_time, self.score_series