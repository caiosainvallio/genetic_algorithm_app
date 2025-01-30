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
    selected_indices = []
    for _ in range(len(population)):
        r = random.random()
        cumulative_prob = 0.0
        for i, (ind, prob) in enumerate(zip(population, selection_probs)):
            cumulative_prob += prob
            if r < cumulative_prob:
                selected.append(ind)
                selected_indices.append(i)
                break
    return selected, selected_indices

# Crossover function to produce offspring
def crossover(parent1, parent2):
    point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2

# Mutation function to introduce variability
def mutate(individual, mutation_rate=0.01):
    mutated_individual = individual.copy()
    for i in range(len(mutated_individual)):
        if random.random() < mutation_rate:
            mutated_individual[i] = 1 - mutated_individual[i]  # Flip the bit
    return mutated_individual

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
def visualize_population(population, title="Population Genes"):
    fig, ax = plt.subplots()
    for i, individual in enumerate(population):
        ax.plot(individual, label=f"Ind {i+1}")
    ax.set_xlabel("Gene Position")
    ax.set_ylabel("Gene Value")
    ax.set_title(title)
    ax.legend()
    st.pyplot(fig)

# Selection Visualization
def visualize_selection(population, selected_indices):
    if selected_indices:
        selected_index = selected_indices[0]  # Pick the first selected individual for visualization
        fig, ax = plt.subplots()
        for i, individual in enumerate(population):
            if i == selected_index:
                ax.plot(individual, label=f"Ind {i+1} (Selected)", linestyle='-', color='green')
            else:
                ax.plot(individual, label=f"Ind {i+1}", linestyle='--', color='red')
        ax.set_xlabel("Gene Position")
        ax.set_ylabel("Gene Value")
        ax.set_title("Selection: Example of Selected Individual")
        ax.legend()
        st.pyplot(fig)

# Crossover Visualization
def visualize_crossover(parents, children):
    fig, ax = plt.subplots(2, 1, figsize=(8, 6))
    
    ax[0].plot(parents[0], label="Parent 1", color='blue')
    ax[0].plot(parents[1], label="Parent 2", color='orange')
    ax[0].set_title("Crossover: Example Parents")
    ax[0].legend()

    ax[1].plot(children[0], label="Child 1", color='blue')
    ax[1].plot(children[1], label="Child 2", color='orange')
    ax[1].set_title("Crossover: Example Offspring")
    ax[1].legend()

    st.pyplot(fig)

# Mutation Visualization
def visualize_mutation(before, after):
    fig, ax = plt.subplots()
    ax.plot(before, label="Before Mutation", linestyle='-', color='blue')
    ax.plot(after, label="After Mutation", linestyle='--', color='orange')
    ax.set_xlabel("Gene Position")
    ax.set_ylabel("Gene Value")
    ax.set_title("Mutation: Example Before and After")
    ax.legend()
    st.pyplot(fig)

# Action Logging
def log_action(log, message):
    log.append(message)

# Streamlit App Structure
st.title("Genetic Algorithm: Simplified Visualizations")

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
    selected_population, selected_indices = roulette_wheel_selection(st.session_state.population, st.session_state.fitnesses)
    log_action(st.session_state.log, f"Generation {st.session_state.current_generation}: Selection using Roulette Wheel.")
    visualize_selection(st.session_state.population, selected_indices)
    st.session_state.population = selected_population
    st.session_state.fitnesses = [fitness(ind) for ind in st.session_state.population]

    # Crossover
    new_population = []
    log_action(st.session_state.log, f"Generation {st.session_state.current_generation}: Crossover of selected individuals.")
    if len(st.session_state.population) >= 2:
        parent1 = st.session_state.population[0]
        parent2 = st.session_state.population[1]
        child1, child2 = crossover(parent1, parent2)
        new_population.extend([child1, child2])
        log_action(st.session_state.log, f"Crossover between Parent 1 and Parent 2 produced two children.")
        visualize_crossover([parent1, parent2], [child1, child2])
        new_population.extend(st.session_state.population[2:])  # Add remaining individuals unchanged
    st.session_state.population = new_population
    st.session_state.fitnesses = [fitness(ind) for ind in st.session_state.population]

    # Mutation
    log_action(st.session_state.log, f"Generation {st.session_state.current_generation}: Mutation of individuals.")
    if st.session_state.population:
        individual = st.session_state.population[0]  # Pick the first individual for mutation example
        mutated_individual = mutate(individual, mutation_rate)
        st.session_state.population[0] = mutated_individual
        visualize_mutation(individual, mutated_individual)
    st.session_state.fitnesses = [fitness(ind) for ind in st.session_state.population]

    # Calculate and store diversity
    diversity = calculate_diversity(st.session_state.population)
    st.session_state.diversity.append(diversity)

# Function to run a single step in the step-by-step mode
def run_step():
    if st.session_state.step == 0:
        st.session_state.current_generation += 1
        selected_population, selected_indices = roulette_wheel_selection(st.session_state.population, st.session_state.fitnesses)
        log_action(st.session_state.log, f"Generation {st.session_state.current_generation}: Selection using Roulette Wheel.")
        visualize_selection(st.session_state.population, selected_indices)
        st.session_state.population = selected_population
        st.session_state.fitnesses = [fitness(ind) for ind in st.session_state.population]
        st.session_state.step += 1
    elif st.session_state.step == 1:
        new_population = []
        log_action(st.session_state.log, f"Generation {st.session_state.current_generation}: Crossover of selected individuals.")
        if len(st.session_state.population) >= 2:
            parent1 = st.session_state.population[0]
            parent2 = st.session_state.population[1]
            child1, child2 = crossover(parent1, parent2)
            new_population.extend([child1, child2])
            log_action(st.session_state.log, f"Crossover between Parent 1 and Parent 2 produced two children.")
            visualize_crossover([parent1, parent2], [child1, child2])
            new_population.extend(st.session_state.population[2:])  # Add remaining individuals unchanged
        st.session_state.population = new_population
        st.session_state.fitnesses = [fitness(ind) for ind in st.session_state.population]
        st.session_state.step += 1
    elif st.session_state.step == 2:
        log_action(st.session_state.log, f"Generation {st.session_state.current_generation}: Mutation of individuals.")
        if st.session_state.population:
            individual = st.session_state.population[0]  # Pick the first individual for mutation example
            mutated_individual = mutate(individual, mutation_rate)
            st.session_state.population[0] = mutated_individual
            visualize_mutation(individual, mutated_individual)
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
visualize_population(st.session_state.population, title="Current Population")

st.subheader("Fitness Visualization")
visualize_fitness(st.session_state.fitnesses)

st.subheader("Diversity Visualization")
visualize_diversity(st.session_state.diversity)

# Display action log in an expander at the bottom
with st.expander("Action Log"):
    for log_entry in st.session_state.log:
        st.write(log_entry)
