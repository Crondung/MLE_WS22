import math

import numpy as np
import random

def generate_weights(array_length: int):
    return np.random.rand(array_length) * 2

def fitness(c: float, gen: np.ndarray, weights: np.ndarray):
    return math.exp(-c*(pow(100-sum([gen[k] * weights[k] for k in range(len(gen))]), 2)))

def generate_gene(array_length: int):
    gene = np.random.rand(array_length)
    return [1 if (i >= 0.5) else 0 for i in gene]

def tasmanian_devil(gene1: np.ndarray, gene2: np.ndarray):
    crossover_point = random.randint(0, len(gene1)-1)
    child1 = gene1[:crossover_point] + gene2[crossover_point:]
    child2 = gene2[:crossover_point] + gene1[crossover_point:]
    return child1, child2

def evolutionary_santa(genes: np.ndarray, weights: np.ndarray, c: int, maxThreshold: float):
    fitnesses = [fitness(c, gen, weights) for gen in genes]
    while np.max(fitnesses) < maxThreshold:
        print(np.max(fitnesses))
        fitnesses_copy = np.copy(fitnesses)
        np.sort(fitnesses_copy)

        gene_fitness_tuples_sorted = sorted(zip(fitnesses, genes), reverse=True)
        sorted_fitnesses, sorted_genes = zip(*list(gene_fitness_tuples_sorted))
        sorted_genes_list = list(sorted_genes)
        new_generation = sorted_genes_list[:len(genes)//2]
        for i in range(len(new_generation)//2):
            parent1 = new_generation[i]
            parent2 = new_generation[-i]

            child1, child2 = tasmanian_devil(parent1, parent2)
            new_generation += child1
            new_generation += child2

        genes = new_generation
        fitnesses = [fitness(c, gen, weights) for gen in genes]


if __name__ == "__main__":
    population_size = 20
    gene_size = 10
    genes = [generate_gene(gene_size) for i in range(population_size)]
    weights = generate_weights(gene_size)
    evolutionary_santa(genes, weights, 0.001, 80)

