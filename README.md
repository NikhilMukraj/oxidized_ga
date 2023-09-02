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

`oxidized_ga.BitString.length()`

Returns the length of the given BitString

---

`oxidized_ga.BitString.string`

Returns the string stored within the class

---

### Decode

---

`oxidized_ga.decode(bitstring, lower_bounds, upper_bounds, n_bits)`

Given a BitString, decodes the binary into usable float values (using the number of bits provided) and scales it to the given bounds

- `bitstring`: A bitstring represented as a string (`ga.BitString.string`)
- `lower_bounds`: (1D numpy float32 array, one element for each value decode should return) the lower bound to scale the output of decode to
- `upper_bounds`: (1D numpy float32 array, one element for each value decode should return) the upper bound to scale the output of decode to
- `n_bits`: The (positive) integer number of bits each value within the bitstring takes up

---

### Genetic Algorithm

---

`genetic_algo(objective_func, lower_bounds, upper_bounds, n_bits, n_iter, n_pop, r_cross, r_mut, k, settings, workers, total_n_bits, print_output)`

Performs the genetic algorithm given an objective function

- `objective_func`: A python function that has the arguments `(string, lower_bounds, upper_bounds, n_bits, settings)` that the BitStrings will be scored on
- `lower_bounds`: (1D numpy float32 array, one element for each value decode should return) the lower bound to scale the output of decode to
- `upper_bounds`: (1D numpy float32 array, one element for each value decode should return) the upper bound to scale the output of decode to
- `n_bits`: The (positive) integer number of bits each value within the BitString takes up
- `n_iter`: The (positive) integer number of iterations to perform the genetic algorithm for
- `n_pop`: The (positive, even) integer size of the population of BitStrings to evaluate
- `r_cross`: The (float) chance of performing crossover
- `r_mut`: The (float) chance of a `0` or `1` flipping during mutation
- `k`: The (positive integer) number of individuals to pull from while performing each tournament selection
- `settings`: (optional, defaults to {}) any secondary parameters to be passed to the objective function
- `workers`: (optional, defaults to 0) the level of parallelism to use
  - `0`: No parallelism
  - `1`: Paralleism only during creation of mutations and execution of crossover functions
  - `>=2`: Parallelism during mutation and crossover as well as the amount of threads to use while executing the objective functions
- `total_n_bits`: (optional, defaults to None) the total amount of bits the BitStrings should be
- `print_output`: (optional, defaults to None, automatically print output) boolean as whether or not to print information about the genetic algorithm while running

---

## Neural Networks

### Activation

---

Activation Types: `ReLU`, `Sigmoid`, `Tanh`, `Swish`, `ELU`

---

`oxidized_ga.Activation.calculate(x)`

Applies the given activation function on a given value

`x`: A numerical value

---

### Layer

---

`oxidized_ga.Layer(size, weights, biases, activation)`

Create a layer class given the shape of the layer, the weight values, the biases, and an activation

`size`: The (positive) integer value specifying the shape of the layer
`weights`: A 2D numpy float32 array that contains the weight specifications
`biases`: A 1D numpy float32 array that contains the bias specifications
`activation`: A `oxidized_ga.Activation` type

---

`oxidized_ga.Layer.set_weights(new_weights)`

Resets the given weights with a set of weights

`new_weights`: A 2D numpy float32 array that contains the weight specifications

---

`oxidized_ga.Layer.set_biases(new_biases)`

Resets the given weights with a set of bias

`new_biases`: A 1D numpy float32 array that contains the bias specifications

---

`oxidized_ga.Layer.calculate(inputs)`

Calculate the output of a layer given some inputs

`inputs`: A 1D numpy float32 array that contains the inputs

---

### Neural Network

---

`oxidized_ga.NeuralNetwork(size, layers)`

Creates a NeuralNetwork class given the size of the network and how many layers

- `size`: The (positive) integer number of layers within the neural network
- `layers`: A list of Layer classes to use within the neural network, the output shape of the previous neural network (which should mean that the first dimensions of one layer should be the same as the second dimension of the next layer)

---

`oxidized_ga.NeuralNetwork.predict(inputs)`

Calculates the output of the neural network given some inputs

- `inputs`: A 1D numpy float32 array that contains the input values to use

---

`oxidized_ga.NeuralNetwork.set_layer(index, new_weights, new_biases)`

Resets the weights or biases of a given layer given the new values and the index of the layer

- `index`: The index of the layer to edit
- `new_weights`: An (optional, defaults to `None`) 2D numpy float32 array containing the new weight values that must match the shape of the original weights
- `new_biases`: An (optional, defaults to `None`) 1D numpy float32 array containing the new bias values that must match the shape of the original biases

---

`oxidized_ga.NeuralNetwork.re_init_layer(index, new_layer)`

Re-initializes a given layer with another layer by index

- `index`: A (positive) integer representing the index of the layer to edit
- `new_layer`: A new Layer class that matches the shape of the previous layer's weights and biases

---

`oxidized_ga.NeuralNetwork.add_layer(new_layer)`

Adds a new layer to the NeuralNetwork class given a Layer class

- `new_layer`: A new layer which weights' second dimensions match the first dimensions of the last layer's weights

---

### Architecture Explanation

```python
  {
    index_number_1: [[oxidized_ga.Activation.type1, oxidized_ga.Activation.type2], [input_shape_1, input_shape_2]],
    index_number_2: [[oxidized_ga.Activation.type2, oxidized_ga.Activation.type3, oxidized_ga.Activation.type4], [input_shape_2, input_shape_3]],
    ...
  }
```

The index number specifies where the place the layer, and chooses an activation type from the list provided, and uses the input shapes to determine the shape of the weights and biases

---

### Neural Network Get Bits

---

`oxidized_ga.neural_network_get_bits(precision, activation_precision, architecture)`

Returns the amount of bits that a neural network will take up given the precision of the weights and biases, the precision of the representation of the activation functions, and the architecture

- `precision`: The number of bits each weight or bias will take up
- `activation_precision`: The number of bits each activation function will take up
- `architecture`: A dictionary storing a representation of the neural network (see [architecture explanation](#architecture-explanation))

---

### Neural Network Decode

---

`oxidized_ga.neural_network_decode(bitstring, lower_bounds, upper_bounds, precision, activation_precision, architecture)`

Decodes a BitString into a NeuralNetwork given the bounds, the precision, the activation's precision, and the architecture

- `bitstring`: A bitstring represented as a string (`ga.BitString.string`)
- `lower_bounds`: (1D numpy float32 array, one element for each value decode should return) the lower bound to scale the output of decode to
- `upper_bounds`: (1D numpy float32 array, one element for each value decode should return) the upper bound to scale the output of decode to
- `precision`: The number of bits each weight or bias will take up
- `activation_precision`: The number of bits each activation function will take up
- `architecture`: A dictionary storing a representation of the neural network (see [architecture explanation](#architecture-explanation))

---

### Neural Network Genetic Algorithm

---

`oxidized_ga.neural_net_genetic_algo(objective_func, lower_bounds, upper_bounds, precision, activation_precision, architecture, n_iter, n_pop, r_cross, r_mut, k, settings, workers, print_output)`

- `objective_func`: A python function that has the arguments `(string, lower_bounds, upper_bounds, n_bits, settings)` that the BitStrings will be scored on
- `lower_bounds`: (1D numpy float32 array, one element for each value decode should return) the lower bound to scale the output of decode to
- `upper_bounds`: (1D numpy float32 array, one element for each value decode should return) the upper bound to scale the output of decode to
- `precision`: The number of bits each weight or bias will take up
- `activation_precision`: The number of bits each activation function will take up
- `n_iter`: The (positive) integer number of iterations to perform the genetic algorithm for
- `n_pop`: The (positive, even) integer size of the population of BitStrings to evaluate
- `r_cross`: The (float) chance of performing crossover
- `r_mut`: The (float) chance of a `0` or `1` flipping during mutation
- `k`: The (positive integer) number of individuals to pull from while performing each tournament selection
- `settings`: (optional, defaults to {}) any secondary parameters to be passed to the objective function
- `workers`: (optional, defaults to 0) the level of parallelism to use
  - `0`: No parallelism
  - `1`: Paralleism only during creation of mutations and execution of crossover functions
  - `>=2`: Parallelism during mutation and crossover as well as the amount of threads to use while executing the objective functions
- `total_n_bits`: (optional, defaults to None) the total amount of bits the BitStrings should be
- `print_output`: (optional, defaults to None, automatically print output) boolean as whether or not to print information about the genetic algorithm while running

---

## Graphs

### Graph

---

`oxidized_ga.Graph(nodes)`

Create a (directed) Graph class with no edges given a dictionary of node and value key-value pairs

- `nodes`: A dictionary of nodes (integers >=0) and their values (integers >=0)

---

`oxidized_ga.Graph.get_nodes()`

Returns a dictionary of the nodes and their values

---

`oxidized_ga.Graph.reset_with_nodes(new_nodes)`

Resets the Graph class with a new set of nodes

- `new_nodes`: A dictionary of nodes (integers >=0) and their values (integers >=0)

---

`oxidized_ga.Graph.get_matrix()`

Returns an adjacency matrix with the nodes and their edges

---

`oxidized_ga.Graph.edit_edge(start, end, weight)`

Edits the weight value of a given edge using the start and end nodes

- `start`: Start node (unsigned integer)
- `end`: End node (unsigned integer)
- `weight`: Weight value (float32)

---

`oxidized_ga.Graph.edit_edges(edges_to_edit)`

Edits a list of edges using the start node, end node, and weight provided

- `edges_to_edit`: A list of edges to edit in the following format:

```python
[
  [start1, end1, weight1],
  [start2, end2, weight2],
  [start3, end3, weight3],
  ...
]
```

---

`oxidized_ga.Graph.add_vertex(value)`

Adds a node with a value

- `value`: Unsigned integer or None, if None the value defaults to 0

---

`oxidized_ga.Graph.add_vertices(vertices)`

Adds the given amount of vertices to the Graph with the default

- `vertices`: (unsigned integer or none) number of vertices to add with the default value of zero

---

`oxidized_ga.Graph.add_vertices_by_value(values)`

Adds a node for each value within the input with the value that the input specifies

- `values`: A list of unsigned integers

---

`oxidized_ga.Graph.update_node_value(node, new_value)`

Updates a node with the specified value

- `node`: Unsigned integer specifying which node
- `new_value`: Unsigned integer to update node with

---

`oxidized_ga.Graph.get_all_edges()`

Returns every edge in the Graph in the following format:

```python
[
  [start1, end1, weight1],
  [start2, end2, weight2],
  [start3, end3, weight3],
  ...
]
```

---

`oxidized_ga.Graph.is_end_node(node)`

Returns a boolean specifying whether or not the given node is an end node (connected by only one other node)

- `node`: Unsigned integer specifying where the node is

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
