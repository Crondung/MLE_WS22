import math
import numpy as np
import random

import numpy.random

POPULATION_SIZE = 1000
GENE_SIZE = 100
C = 0.001
RNG = np.random.default_rng()
CROSSOVER_POINT = 40


def init_gene(gene_size: int):
    gene = np.random.rand(gene_size)
    return [1 if (i >= 0.5) else 0 for i in gene]


def init_population(population_size: int, gene_size: int):
    return [init_gene(gene_size) for i in range(population_size)]


def init_volumes(volume_size: int):
    return np.random.rand(volume_size) * 2


def calculate_fitness(gene: np.ndarray, volumes: np.ndarray):
    return math.exp(-C * (pow(100 - sum([gene[k] * volumes[k] for k in range(len(gene))]), 2)))


def crossover(parent1: np.ndarray, parent2: np.ndarray, crossover_point: int = CROSSOVER_POINT):
    child1: np.ndarray = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
    child2: np.ndarray = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
    return [child1, child2]


def encode_greycode(arr: np.ndarray):
    return [arr[0]] + [arr[i] ^ arr[i - 1] for i in range(1, len(arr))]


def decode_greycode(encoded: np.ndarray, original_arr: np.ndarray):
    return [encoded[0]] + [encoded[i] ^ original_arr[i - 1] for i in range(1, len(encoded))]


def mutate(gene: np.ndarray):
    index = random.randint(0, len(gene)-1)
    gene[index] = 1 if gene[index] == 0 else 0
    return gene


def calculate_probabilities(fitness_arr: np.ndarray):
    return [fitness / sum(fitness_arr) for fitness in fitness_arr]


def calculate_fitness_array(genes: np.ndarray, volumes: np.ndarray):
    return [calculate_fitness(gene, volumes) for gene in genes]


def calculate_present_bag_weight(gene: np.ndarray, volumes: np.ndarray):
    return sum([gene[i] * volumes[i] for i in range(len(gene))])


def genetic_evolution(r: float, m: int, fitness_threshold: float):
    """
    execute a genetic algorithm to maximize the fitness function
    :param r: crossover rate = share of genes to replace with crossover per generation
    :param m: mutation rate = # of genes to mutate per generation
    :param fitness_threshold:
    :return:
    """
    population = init_population(POPULATION_SIZE, GENE_SIZE)
    volumes = init_volumes(GENE_SIZE)
    fitness_arr = calculate_fitness_array(population, volumes)
    while max(fitness_arr) < fitness_threshold:
        best_gene_index = fitness_arr.index(max(fitness_arr))
        best_gene = population[best_gene_index]
        #print(calculate_present_bag_weight(best_gene, volumes))
        print(max(fitness_arr))
        probabilities = calculate_probabilities(fitness_arr)
        selection = RNG.choice(population, int((1 - r) * POPULATION_SIZE), p=probabilities)
        # optional: best_gene_index = fitness_arr.index(max(fitness_arr))
        # optional: best_gene = population[best_gene_index]
        # optional: next_generation = selection[:-1] + best_gene
        selection_fitness_arr = calculate_fitness_array(selection, volumes)
        selection_probabilities = calculate_probabilities(selection_fitness_arr) # [probabilities[population.index(gene.tolist())] for gene in selection]
        parents = [RNG.choice(selection, 2, p=selection_probabilities, replace=False) for i in
                   range(int(r * POPULATION_SIZE
                             / 2))]
        children = [crossover(parent[0], parent[1]) for parent in parents]
        next_generation: np.ndarray = np.concatenate([selection, np.concatenate(children)])
        mutation_candidates = RNG.choice(next_generation, m, replace=False)
        for candidate in mutation_candidates:
            next_generation[next_generation.tolist().index(candidate.tolist())] = decode_greycode(mutate(encode_greycode(candidate)), candidate)
        population = next_generation.tolist()
        fitness_arr = calculate_fitness_array(population, volumes)


if __name__ == "__main__":
    genetic_evolution(0.5, 200, 0.9)
