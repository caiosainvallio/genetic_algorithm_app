Here's the `README.md` file for your genetic algorithm visualization app:

---

# Genetic Algorithm Visualization App

## Overview

This app provides an interactive environment for exploring and visualizing the operation of a genetic algorithm. It is designed to help users understand the core concepts of genetic algorithms, including selection, crossover, mutation, and how these processes affect a population over multiple generations.

The app offers three execution modes:

1. **Run Step-by-Step**: Allows users to manually progress through each step of the genetic algorithm (Selection, Crossover, Mutation).
2. **Run One Generation**: Executes a full iteration of the genetic algorithm (Selection, Crossover, Mutation) in one step.
3. **Run N Generations**: Automatically runs the genetic algorithm for a specified number of generations.

## Features

- **Interactive Controls**: Adjust parameters such as population size, gene length, mutation rate, and number of generations.
- **Visualization**: Real-time visualization of the population, fitness, and genetic diversity across generations.
- **Expandable Action Log**: Detailed logs of each step in the genetic algorithm, available in an expandable section at the bottom of the app.

## Parameters

The app allows you to control the following parameters via the sidebar:

- **Population Size**: The number of individuals in the population.
- **Gene Length**: The length of each individual's gene sequence.
- **Mutation Rate**: The probability of each gene mutating during the mutation step.
- **Number of Generations**: The number of generations to run when using the "Run N Generations" option.

## Execution Modes

### 1. Run Step-by-Step
- This mode lets you manually step through each stage of the genetic algorithm. After each step (Selection, Crossover, Mutation), the app will wait for your input to proceed to the next step.

### 2. Run One Generation
- Executes one complete iteration of the genetic algorithm. This includes performing selection, crossover, and mutation in sequence.

### 3. Run N Generations
- Automatically runs the genetic algorithm for the number of generations specified by the "Number of Generations" slider. Each generation is logged in the expandable action log at the bottom of the app.

## Visualization

- **Population Visualization**: Shows the gene values of each individual in the population.
- **Fitness Visualization**: Displays a bar chart of the fitness values for the population after the current iteration.
- **Diversity Visualization**: Plots the genetic diversity of the population across generations, helping to visualize how diversity changes as the algorithm progresses.

## Action Log

The action log provides a detailed record of the operations performed in each generation. It logs actions such as "Selection using Roulette Wheel," "Crossover of selected individuals," and "Mutation of individuals." The log is contained within an expandable section at the bottom of the app, allowing you to review the steps taken by the algorithm without cluttering the main interface.

## How to Use

1. **Set Parameters**: Use the sliders in the sidebar to set your desired parameters (population size, gene length, mutation rate, number of generations).
2. **Choose Execution Mode**: Select one of the three execution modes by clicking the corresponding button:
   - **Run Step-by-Step**
   - **Run One Generation**
   - **Run N Generations**
3. **Observe Results**: Watch the visualizations update in real-time as the algorithm runs. Expand the action log to see detailed information about each step.
4. **Restart**: Use the "Restart" button in the sidebar to reset the algorithm and start over with new parameters.

## Requirements

To run this app, you need to have the following Python packages installed:

- `streamlit`
- `numpy`
- `matplotlib`

You can install these packages using pip:

```bash
pip install streamlit numpy matplotlib
```

## Running the App

To run the app locally, use the following command in your terminal:

```bash
streamlit run app.py
```

This will start the app in your default web browser.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

