import matplotlib.pyplot as plt
import numpy as np


# Funktion, um die Daten aus der Textdatei zu lesen
def read_data(filename):
    x, y = [], []
    with open(filename) as f:
        for line in f:
            data = line.split(';')
            x.append(float(data[0]))
            y.append(float(data[1]))
    return x, y


# Funktion, um den K-Nearest-Neighbor Algorithmus auszuführen
def classify(x, y, test_point, k):
    # Berechne die Distanzen zwischen dem Testpunkt und den Trainingspunkten
    distances = [(xi - test_point[0]) ** 2 + (yi - test_point[1]) ** 2 for xi, yi in zip(x, y)]
    # Sortiere die Punkte basierend auf ihrer Distanz zum Testpunkt
    sorted_points = [point for _, point in sorted(zip(distances, list(zip(x, y))))]
    # Wähle die k nächsten Nachbarn
    nearest_neighbors = sorted_points[:k]
    # Zähle, wie viele Nachbarn jeder Klasse zugeordnet sind
    class_counts = {}
    for xi, yi in nearest_neighbors:
        if (xi, yi) in class_counts:
            class_counts[(xi, yi)] += 1
        else:
            class_counts[(xi, yi)] = 1
    # Ordne den Testpunkt der Klasse zu, die die meisten Nachbarn hat
    return max(class_counts, key=class_counts.get)


# Lese die Daten aus der Textdatei ein
x, y = read_data('data.txt')

# Führe den K-Nearest-Neighbor Algorithmus für k = 3 und k = 5 aus
predictions_3, predictions_5 = [], []
for xi, yi in zip(x, y):
    prediction_3 = classify(x, y, (xi, yi), 3)
    predictions_3.append(prediction_3)
    prediction_5 = classify(x, y, (xi, yi), 5)
    predictions_5.append(prediction_5)

# Erstelle einen Plot mit den Vorhersagen für k = 3
plt.scatter(x, y, c=predictions_3)
plt.title('K-Nearest-Neighbors (k = 3)')
plt.show()

# Erstelle einen Plot mit den Vorhersagen für k = 5
plt.scatter(x, y, c=predictions_5)
plt.title('K-Nearest-Neighbors (k = 5)')
plt.show()
