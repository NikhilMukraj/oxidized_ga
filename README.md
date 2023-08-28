# Oxidized GA

## Core Functionality

### Bit String

---

`oxidized_ga.BitString(string)`

Creates a BitString given a string of numerical values

- `string`: A string of characters that are either `'0'` or `'1'`

---

`oxidized_ga.BitString.set(string)`

Sets the string value of the BitString to a given string

- `string`: A string of characters that are either `'0'` or `'1'`

---

Returns the length of the given BitString

`oxidized_ga.BitString.length()`

---

### Decode

---

`oxidized_ga.decode(bitstring, lower_bounds, upper_bounds, n_bits)`

---

### Genetic Algorithm

---

`genetic_algo(objective_func, lower_bounds, upper_bounds, n_bits, n_iter, n_pop, r_cross, r_mut, k, settings, workers, total_n_bits, print_output)`

---

## Neural Networks

### Activation

---

Activation Types: `ReLU`, `Sigmoid`, `Tanh`, `Swish`, `ELU`

---

`oxidized_ga.Activation.calculate(x)`

---

### Layer

---

`oxidized_ga.Layer(size, weights, biases, activation)`

---

`oxidized_ga.Layer.set_weights(new_weights)`

---

`oxidized_ga.Layer.set_biases(new_biases)`

---

`oxidized_ga.Layer.calculate(inputs)`

---

### Neural Network

---

`oxidized_ga.NeuralNetwork(size, layers)`

---

`oxidized_ga.NeuralNetwork.predict(inputs)`

---

`oxidized_ga.NeuralNetwork.set_layer(index, new_weights, new_biases)`

---

`oxidized_ga.NeuralNetwork.re_init_layer(index, new_layer)`

---

`oxidized_ga.NeuralNetwork.add_layer(new_layer)`

---

### Neural Network Get Bits

---

`oxidized_ga.neural_network_get_bits(precision, activation_precision, architecture)`

---

### Neural Network Decode

---

`oxidized_ga.neural_network_decode(bitstring, lower_bounds, upper_bounds, n_bits, activation_precision, architecture)`

---

### Neural Network Genetic Algorithm

---

`oxidized_ga.neural_net_genetic_algo(objective_func, lower_bounds, upper_bounds, n_bits, activation_precision, architecture, n_iter, n_pop, r_cross, r_mut, k, settings, workers, print_output)`

---

## Graphs

### Graph

---

`oxidized_ga.Graph(nodes)`

---

`oxidized_ga.Graph.get_nodes()`

---

`oxidized_ga.Graph.reset_with_nodes(new_nodes)`

---

`oxidized_ga.Graph.get_matrix()`

---

`oxidized_ga.Graph.edit_edge(start, end, weight)`

---

`oxidized_ga.Graph.edit_edges(edges_to_edit)`

---

`oxidized_ga.Graph.add_vertex(value)`

---

`oxidized_ga.Graph.add_vertices(vertices)`

---

`oxidized_ga.Graph.add_vertices_by_value(values)`

---

`oxidized_ga.Graph.update_node_value(node, new_value)`

---

`oxidized_ga.Graph.get_all_edges()`

---

`oxidized_ga.Graph.is_end_node(node)`

---

`oxidized_ga.Graph.get_connected_component(start_node, disconnected_nodes)`

---

`oxidized_ga.Graph.get_all_neighbors(node)`

---

`oxidized_ga.Graph.replace_subgraph(subgraph, start_node, other_node)`

---

`oxidized_ga.Graph.mutate_types(rules)`

---

`oxidized_ga.Graph.mutate_edges(rules, edges)`

---

`oxidized_ga.Graph.mutate_vertices(add_types, delete_types)`

---

`oxidized_ga.Graph.mutate_weights(lower, upper)`

---

`oxidized_ga.Graph.create_random(size_range, edge_prob, edge_range, values)`

---

### Graph Mutate

---

`oxidized_ga.mutate(graph, r_mut, reps=None, type_rules, vertex_rules, edge_rules, weight_rules, possible_edges)`

---

### Graph Isolate Subgraph

---

`oxidized_ga.isolate_random_subgraph(graph, return_vertices)`

---

### Graph Crossover

---

`oxidized_ga.crossover(graph1, graph2, r_cross)`

---

### Graph Genetic Algorithm

---

`oxidized_ga.graph_genetic_algo(objective_func, n_iter, n_pop, r_cross, r_mut, k, size_range, initialization_prob, settings, parallel, reps, edge_range, values, type_rules, vertex_rules, edge_rules,  weight_rules, possible_edges)`

---

## To Do

- Create `oxidized_ga.Graph.delete_vertex` function
- Wrap `oxidized_ga.graph_genetic_algo` in python function, and add default parameters on Rust side of the code through `pyo3` attribute
- Create tree and undirected graph classes
