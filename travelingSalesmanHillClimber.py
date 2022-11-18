import random

import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
import math
import matplotlib.pyplot

STEPS: int = 3000#100000

def generate_city_distances_array(number_of_cities: int, max_distance: int):
    distance_matrix: ndarray = np.empty((number_of_cities, number_of_cities))  # generate new array
    for i in range(0, number_of_cities):
        for j in range(0, number_of_cities):
            if i == j:
                distance_matrix[i, j] = 0
            elif j < i:
                distance_matrix[i, j] = distance_matrix[j, i]
            else:
                distance_matrix[i, j] = random.random() * max_distance
    return distance_matrix


def calculate_distance_of_path(distance_matrix: ndarray, path: list):
    distance = 0
    for i in range(1, len(path)):
        distance += distance_matrix[path[i - 1], path[i]]
    return distance


def solve_hill_climb(distance_matrix: ndarray, steps: int):
    yplot_array: ndarray = np.array([])  # for plotting
    array_length = len(distance_matrix)
    hypothesis = [*range(0, array_length), 0]
    fitness = calculate_distance_of_path(distance_matrix, hypothesis) * -1
    #print("path: " + str(hypothesis) + ", distance: " + str(fitness * -1))
    hypothesis_without_start_and_stop = hypothesis[1:len(hypothesis) - 1]
    for i in range(0, steps):
        choice = random.sample(hypothesis_without_start_and_stop, 2)
        new_hypothesis = hypothesis.copy()
        new_hypothesis[hypothesis.index(choice[0])] = choice[1]
        new_hypothesis[hypothesis.index(choice[1])] = choice[0]
        new_fitness = calculate_distance_of_path(distance_matrix, new_hypothesis) * -1
        if new_fitness > fitness: # -50 > -100 => 100 besser als 50
            #print("found better path in step " + str(i))
            #print("better path: " + str(new_hypothesis) + ", distance: " + str(new_fitness * -1))
            fitness = new_fitness
            hypothesis = new_hypothesis
        yplot_array = np.append(yplot_array, fitness)
    print("finished")
    return fitness, yplot_array

def simulated_annealing(distance_matrix: ndarray, steps: int, temp: float, epsilon: float):
    yplot_array: ndarray = np.array([]) #for plotting
    array_length = len(distance_matrix)
    hypothesis = [*range(0, array_length), 0]
    fitness = calculate_distance_of_path(distance_matrix, hypothesis) * -1
    #print(f"path: {hypothesis} distance: {fitness * -1}")
    hypothesis_without_start_and_stop = hypothesis[1:len(hypothesis) - 1]
    for i in range(0, steps):
        choice = random.sample(hypothesis_without_start_and_stop, 2)
        new_hypothesis = hypothesis.copy()
        new_hypothesis[hypothesis.index(choice[0])] = choice[1]
        new_hypothesis[hypothesis.index(choice[1])] = choice[0]
        new_fitness = calculate_distance_of_path(distance_matrix, new_hypothesis) * -1
        if new_fitness > fitness:
            #print(f"found better path in step {i}")
            #print(f"better path: {new_hypothesis}, distance: {new_fitness*-1}")
            fitness = new_fitness
            hypothesis = new_hypothesis
        elif(random.random() < math.exp((new_fitness - fitness)/temp)):
            #print(f'swapped with {math.exp((new_fitness - fitness)/temp)}')
            #print(f'new path: {new_hypothesis}, distance: {new_fitness*-1}')
            fitness = new_fitness
            hypothesis = new_hypothesis
        temp = temp - epsilon
        yplot_array = np.append(yplot_array, fitness)

    print("finished")
    return fitness, yplot_array

if __name__ == "__main__":
    distance_array = generate_city_distances_array(20, 100)
    xpoints = np.array(range(STEPS))
    distance_HC, ypoints_HC = solve_hill_climb(distance_array, STEPS)
    distance_SA, ypoints_SA = simulated_annealing(distance_array, STEPS, 10, 0.00001)
    print(f'Hillclimb: {distance_HC}')
    print(f'Simulated Annealing: {distance_SA}')
    plt.plot(xpoints, ypoints_HC, color='r', label='HC')
    plt.plot(xpoints, ypoints_SA, color='g', label='SA')
    plt.xlabel("Steps")
    plt.ylabel("Fitness")
    plt.legend()
    plt.show()
"""
    0 2 3
    2 0 4
    3 4 0
"""
