# tsp-simulated-annealing

Simulated annealing optimizer designed to solve Travelling Salesman Problem (TSP). Working of this optimizer consists of three stages:
* Stage 1: Greedy algorithm provides an initial solution,
* Stage 2: Simulated Annealing searches solutions space,
* Stage 3: 2-Opt algorithm improves the solution obtained in stage 2. (optional)

# DEMO 

* Input should be in the form of a list [ [x_1, y_1], [x_2, y_2], ... , [x_n, y_n] ]
```python
from random import randint

towns = []

for town in range(60):
    towns.append([randint(0, 100), randint(0, 100)])
```
* How to initialize 
```python
from SimulatedAnnealing import SimulatedAnnealing

SA_optimizer = SimulatedAnnealing()
SA_optimizer.fit(towns, iterations=60000, cooling='polynomial', improve=True)
```

* Plotting results 
```python
SA_optimizer.plot()
```
![plot_output_SAO](https://user-images.githubusercontent.com/114445740/194719053-a7f8d110-6f98-420d-bd41-590b99291171.png)


* Visualization 
```python
SA_optimizer.show_graph(fitted=True)
```
![graph_output_SAO](https://user-images.githubusercontent.com/114445740/194719056-fc9573a0-f075-4f16-b8e9-4db2bae1528c.png)
