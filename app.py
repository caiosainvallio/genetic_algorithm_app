import streamlit as st
import random
import numpy as np
import matplotlib.pyplot as plt

# Define a simple fitness function
def fitness(individual):
    return sum(individual)

# Initialize a population of random individuals
def initialize_population(pop_size, gene_length):
    return [np.random.randint(2, size=gene_length).tolist() for _ in range(pop_size)]

# Roulette Wheel Selection function
def roulette_wheel_selection(population, fitnesses):
    total_fitness = sum(fitnesses)
    selection_probs = [f / total_fitness for f in fitnesses]
    selected = []
    for _ in range(len(population)):
        r = random.random()
        cumulative_prob = 0.0
        for ind, prob in zip(population, selection_probs):
            cumulative_prob += prob
            if r < cumulative_prob:
                selected.append(ind)
                break
    return selected

# Crossover function to produce offspring
def crossover(parent1, parent2):
    point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2

# Mutation function to introduce variability
def mutate(individual, mutation_rate=0.01):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] = 1 - individual[i]  # Flip the bit
    return individual

# Genetic Diversity Calculation
def calculate_diversity(population):
    gene_matrix = np.array(population)
    gene_variances = np.var(gene_matrix, axis=0)
    diversity = np.mean(gene_variances)
    return diversity

# Fitness Visualization
def visualize_fitness(fitnesses):
    fig, ax = plt.subplots()
    ax.bar(range(len(fitnesses)), fitnesses)
    ax.set_xlabel("Individuals")
    ax.set_ylabel("Fitness")
    ax.set_title("Fitness of Population")
    st.pyplot(fig)

# Diversity Visualization
def visualize_diversity(diversity_list):
    fig, ax = plt.subplots()
    ax.plot(diversity_list)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Diversity")
    ax.set_title("Genetic Diversity Over Generations")
    st.pyplot(fig)

# Population Visualization
def visualize_population(population):
    fig, ax = plt.subplots()
    for i, individual in enumerate(population):
        ax.plot(individual, label=f"Ind {i+1}")
    ax.set_xlabel("Gene Position")
    ax.set_ylabel("Gene Value")
    ax.set_title("Population Genes")
    ax.legend()
    st.pyplot(fig)

# Action Logging
def log_action(log, message):
    log.append(message)

# Streamlit App Structure
st.title("Genetic Algorithm: Minimal Interface with Multiple Execution Modes")

# Sidebar for Controls
st.sidebar.header("Controls")
st.sidebar.markdown("Use these sliders to adjust the parameters of the genetic algorithm.")

pop_size = st.sidebar.slider("Population Size", 2, 20, 10, help="The number of individuals in the population.")
gene_length = st.sidebar.slider("Gene Length", 2, 20, 10, help="The length of each individual's gene.")
mutation_rate = st.sidebar.slider("Mutation Rate", 0.0, 1.0, 0.01, help="The probability of each gene mutating.")
num_generations = st.sidebar.slider("Number of Generations", 1, 50, 10, help="The number of generations to run the algorithm.")

st.sidebar.markdown("Use the buttons below to control the genetic algorithm's execution.")

# Initialize app state
if "current_generation" not in st.session_state:
    st.session_state.current_generation = 0
    st.session_state.diversity = []
    st.session_state.log = []
    st.session_state.population = initialize_population(pop_size, gene_length)
    st.session_state.fitnesses = [fitness(ind) for ind in st.session_state.population]
    st.session_state.step = 0

# Function to run one complete iteration (Selection -> Crossover -> Mutation)
def run_one_iteration():
    st.session_state.current_generation += 1
    
    # Selection
    log_action(st.session_state.log, f"Generation {st.session_state.current_generation}: Selection using Roulette Wheel.")
    st.session_state.population = roulette_wheel_selection(st.session_state.population, st.session_state.fitnesses)
    st.session_state.fitnesses = [fitness(ind) for ind in st.session_state.population]

    # Crossover
    log_action(st.session_state.log, f"Generation {st.session_state.current_generation}: Crossover of selected individuals.")
    new_population = []
    for i in range(0, len(st.session_state.population), 2):
        if i + 1 < len(st.session_state.population):
            parent1 = st.session_state.population[i]
            parent2 = st.session_state.population[i + 1]
            child1, child2 = crossover(parent1, parent2)
            new_population.append(child1)
            new_population.append(child2)
            log_action(st.session_state.log, f"Crossover between {parent1} and {parent2} produced {child1} and {child2}.")
    st.session_state.population = new_population
    st.session_state.fitnesses = [fitness(ind) for ind in st.session_state.population]

    # Mutation
    log_action(st.session_state.log, f"Generation {st.session_state.current_generation}: Mutation of individuals.")
    st.session_state.population = [mutate(ind, mutation_rate) for ind in st.session_state.population]
    st.session_state.fitnesses = [fitness(ind) for ind in st.session_state.population]

    # Calculate and store diversity
    diversity = calculate_diversity(st.session_state.population)
    st.session_state.diversity.append(diversity)

# Function to run a single step in the step-by-step mode
def run_step():
    if st.session_state.step == 0:
        st.session_state.current_generation += 1
        log_action(st.session_state.log, f"Generation {st.session_state.current_generation}: Selection using Roulette Wheel.")
        st.session_state.population = roulette_wheel_selection(st.session_state.population, st.session_state.fitnesses)
        st.session_state.fitnesses = [fitness(ind) for ind in st.session_state.population]
        st.session_state.step += 1
    elif st.session_state.step == 1:
        log_action(st.session_state.log, f"Generation {st.session_state.current_generation}: Crossover of selected individuals.")
        new_population = []
        for i in range(0, len(st.session_state.population), 2):
            if i + 1 < len(st.session_state.population):
                parent1 = st.session_state.population[i]
                parent2 = st.session_state.population[i + 1]
                child1, child2 = crossover(parent1, parent2)
                new_population.append(child1)
                new_population.append(child2)
                log_action(st.session_state.log, f"Crossover between {parent1} and {parent2} produced {child1} and {child2}.")
        st.session_state.population = new_population
        st.session_state.fitnesses = [fitness(ind) for ind in st.session_state.population]
        st.session_state.step += 1
    elif st.session_state.step == 2:
        log_action(st.session_state.log, f"Generation {st.session_state.current_generation}: Mutation of individuals.")
        st.session_state.population = [mutate(ind, mutation_rate) for ind in st.session_state.population]
        st.session_state.fitnesses = [fitness(ind) for ind in st.session_state.population]
        st.session_state.step = 0

        # Calculate and store diversity
        diversity = calculate_diversity(st.session_state.population)
        st.session_state.diversity.append(diversity)

# Function to run multiple generations
def run_n_generations(n):
    for _ in range(n):
        run_one_iteration()

# Sidebar buttons to control the flow
if st.sidebar.button("Run One Generation"):
    run_one_iteration()

if st.sidebar.button("Run Step-by-Step"):
    run_step()

if st.sidebar.button("Run N Generations"):
    run_n_generations(num_generations)

if st.sidebar.button("Restart"):
    st.session_state.current_generation = 0
    st.session_state.diversity = []
    st.session_state.log = []
    st.session_state.population = initialize_population(pop_size, gene_length)
    st.session_state.fitnesses = [fitness(ind) for ind in st.session_state.population]
    st.session_state.step = 0
    st.sidebar.write("Algorithm restarted.")

# Display minimal information at the top
st.markdown(f"### Current Generation: {st.session_state.current_generation}")
st.markdown(f"**Population Size:** {pop_size}  |  **Gene Length:** {gene_length}  |  **Mutation Rate:** {mutation_rate}")

# Display results
st.subheader("Population Visualization")
visualize_population(st.session_state.population)

st.subheader("Fitness Visualization")
visualize_fitness(st.session_state.fitnesses)

st.subheader("Diversity Visualization")
visualize_diversity(st.session_state.diversity)

# Display action log in an expander at the bottom
with st.expander("Action Log"):
    for log_entry in st.session_state.log:
        st.write(log_entry)
