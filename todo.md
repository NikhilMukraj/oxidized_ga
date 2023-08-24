# todo

- try running with maturin or cargo --release flag to see how much that speeds up operations
  - maturin develop --release
- cleaner objective function errors
  - make sure to include first evaluation of objectives before running algo
  - py.eval("str(e)") on the "e" error variable to return a stacktrace
- handle keyboard interrupt
- graph version: threadpoolexecution of objective functions
- py function for architecture
- py wrappers for convenience
- rewrite underlying graph struct to be in rust and then write a pyclass wrapper
- write underlying rust tree and undirected graph structs
- make sure they implement the different kinds of mutation
- and they implement the different kinds of vertex and edge methods
- store rules for how end nodes mutate within the struct for the tree itself
- anything necessary for mutation in these new structs should be kept within a struct field if they dont match mutate func signatures
