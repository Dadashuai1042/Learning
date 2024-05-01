import pandas as pd
import random
import numpy as np


# 假设NUM_DAYS, NUM_WORKERS, NUM_SHIFTS, WORKING_HOURS, POPULATION_SIZE, MUTATION_RATE, GENERATIONS是预先定义的
# employees = [{'id': i, 'seniority': random.randint(1, 5)} for i in range(NUM_WORKERS)]  # 示例员工信息

# 假设seniority_correlation是一个DataFrame，包含了工龄与合格率和合格数的相关性
# seniority_correlation = pd.DataFrame({
#     'Seniority': [...],
#     'QualificationRate': [...],
#     'QualifiedCount': [...]
# })

# 根据工龄相关性数据，定义一个函数来计算个体的适应度
def calculate_fitness(schedule, seniority_correlation, employees):
    total_qualified_count = 0
    for shift in schedule:
        worker_id = shift
        seniority = employees[worker_id]['seniority']
        corresponding_count = \
        seniority_correlation.loc[seniority_correlation['Seniority'] == seniority, 'QualifiedCount'].values[0]
        total_qualified_count += corresponding_count
    return total_qualified_count


# 初始化种群
def initialize_population(population_size, num_days, num_shifts):
    population = []
    for _ in range(population_size):
        schedule = [random.randint(0, num_workers - 1) for _ in range(num_days * num_shifts)]
        population.append(schedule)
    return population


# 选择操作
def select(population, fitnesses, selection_size):
    selected = random.choices(population, weights=fitnesses, k=selection_size)
    parents = []
    while len(parents) < 2:
        parents.extend(selected)
    return parents


# 交叉操作
def crossover(parent1, parent2):
    point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2


# 变异操作
def mutate(schedule, mutation_rate, num_workers):
    if random.random() < mutation_rate:
        i, j = random.sample(range(len(schedule)), 2)
        schedule[i], schedule[j] = schedule[j], schedule[i]
    return schedule


# 遗传算法主函数
def genetic_algorithm(population, seniority_correlation, employees):
    fitnesses = [calculate_fitness(schedule, seniority_correlation, employees) for schedule in population]

    for generation in range(GENERATIONS):
        new_population = []
        for _ in range(0, POPULATION_SIZE, 2):
            parent1, parent2 = select(population, fitnesses, 2)
            child1, child2 = crossover(parent1, parent2)
            mutate(child1, MUTATION_RATE, NUM_WORKERS)
            mutate(child2, MUTATION_RATE, NUM_WORKERS)
            new_population.extend([child1, child2])

        population = new_population
        new_fitnesses = [calculate_fitness(schedule, seniority_correlation, employees) for schedule in population]
        best_fitness = max(new_fitnesses)
        print(f"Generation {generation}: Best Fitness = {best_fitness}")

        if best_fitness == NUM_DAYS * NUM_WORKERS:  # 假设完美适应度
            break

    best_index = new_fitnesses.index(best_fitness)
    return population[best_index]


# 运行遗传算法
population = initialize_population(POPULATION_SIZE, NUM_DAYS, NUM_SHIFTS)
best_schedule = genetic_algorithm(population, seniority_correlation, employees)
print("Best Schedule Found:", best_schedule)