import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import argparse

def read_ctmc_file(filename):
    with open(filename, "r") as file:
        lines = file.readlines()
        num_states = int(lines[0].strip())
        num_edges = int(lines[1].strip())
        Q = np.zeros((num_states, num_states))

        for line in lines[2:]:
            i, j, rate = line.split()
            i, j = int(i), int(j)
            Q[i, j] = float(rate)

        for i in range(num_states):
            Q[i, i] = -np.sum(Q[i, :])

    return Q

# Function to compute transition matrix P
def compute_discrete_transition_matrix(Q, delta_t):
    P_dt = np.eye(Q.shape[0]) + Q * delta_t
    return P_dt

# Visualizing the Markov Chain
def visualize_markov_chain(P):
    G = nx.DiGraph()
    num_states = P.shape[0]

    for i in range(num_states):
        for j in range(num_states):
            if P[i, j] > 0:
                G.add_edge(i, j, weight=P[i, j])

    pos = nx.spring_layout(G)
    labels = {(i, j): f"{P[i, j]:.2f}" for i, j in G.edges}

    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color="lightblue",
        edge_color="black",
        node_size=2000,
        font_size=12,
    )
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=labels, connectionstyle="arc3,rad=0.2"
    )
    plt.title("Markov Chain Visualization")
    plt.show()


# Function to compute steady-state probabilities
def find_steady_state(P):
    num_states = P.shape[0]
    A = P.T - np.eye(num_states)
    A[-1] = np.ones(num_states)
    b = np.zeros(num_states)
    b[-1] = 1
    steady_state = np.linalg.solve(A, b)
    return steady_state


# Probability distribution after n steps
def probability_after_n_steps(P, n, initial_state):
    P_n = np.linalg.matrix_power(P, n)
    probabilities = []
    for i in range(1, n + 1):
        P_i = np.linalg.matrix_power(P, i)
        prob_i = np.dot(initial_state, P_i)
        probabilities.append(prob_i)
        print(f"Probability distribution after {i} steps: {prob_i}")
    return probabilities[-1]


def compute_ok_probability(steady_state, ok_probs):
    return np.dot(steady_state, ok_probs)


def compute_throughput(steady_state, Q, from_state, to_state):
    return steady_state[from_state] * Q[from_state, to_state]


def compute_source1_empty_probability(steady_state):
    return 1 - steady_state[1]


def compute_source1_throughput(steady_state, Q):
    return steady_state[1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mc_file", type=str, help="Path to the .mc file")

    args = parser.parse_args()
    ctmc_file = args.mc_file

    Q = read_ctmc_file("assignment3_model.mc")
    initial_state = np.array([1, 0])
    ok_probs = np.array([0.9, 0.95])
    time_steps = [2, 1, 0.5, 0.25, 0.1]

    for delta in time_steps:
        print(f"\n--- Time Step Δt = {delta} ---")

        P = compute_discrete_transition_matrix(Q, delta)

        steady_state = find_steady_state(P)
        print(
            f"Steady-State Probability of source 0 being active: {steady_state[0]:.4f}"
        )

        throughput_0_to_1 = compute_throughput(steady_state, Q, 0, 1)
        print(f"Throughput from Source 0 to Source 1: {throughput_0_to_1:.4f}")

        source1_empty_prob = compute_source1_empty_probability(steady_state)
        print(f"Probability of Source 1 Active being empty: {source1_empty_prob:.4f}")

        source1_throughput = compute_source1_throughput(steady_state, Q)
        print(f"Throughput of Source 1 in steady state: {source1_throughput:.4f}")
