// represent graph as adjacency matrix
// represent ways to possibly mutate through an adjacency list
// 1 : [2, 3]
// 2 : [3, 4]
// 4 : [3]
// key represents initial type, value represents new possible type
// additional hashmaps that dictates what vertices can be deleted or added
// additional hashmap that dictates where edges can be added
//
// graph should be able to be edited by:
// - adding or removing edges
// - adding or removing vertices
// - changing vertex type
// - potentially changing weight value
// 
// changing edges should be done by randomly flipping values adjacency matrix
// changing vertices should be done by adding or removing columns/rows
// changing vertex should follow mutation scheme shown above
// changing weight value could be done by assigning bitstrings to each weight value and reusing code
// 
// a sequence of these changes should be performed in a random order
// - a random amount of actions is chosen
// - a random type is chosen and performed

use std::io::{Error, ErrorKind, Result};
use std::collections::{HashMap, HashSet};
use pyo3::prelude::*;
use pyo3::PyResult;
use pyo3::types::{PyDict, PyList};
use numpy::{PyArray, PyReadonlyArray1};
use ndarray::Dim;
use rand::Rng;
use rand::seq::SliceRandom;
use rayon::prelude::*;
#[path = "./converter/mod.rs"] mod converter;
#[path = "./selector/mod.rs"] mod selector;


#[derive(Clone)]
#[pyclass]
pub struct Graph { 
    nodes: HashMap<usize, usize>, 
    matrix: Vec<Vec<f32>>,
}

fn generate_matrix<K, V, T>(nodes: &HashMap<K, V>) -> Result<Vec<Vec<T>>> 
where
    K: std::clone::Clone,
    T: Default + Clone
{
    let keys: Vec<K> = nodes.keys().cloned().collect();

    let mat_size = keys.len();

    let mut matrix: Vec<Vec<T>> = Vec::with_capacity(mat_size);
    for _ in 0..mat_size {
        matrix.push(vec![T::default(); mat_size]);
    }

    return Ok(matrix);
}

// subclass to make undirected graph through checks to adjacency matrix
// tree subclass (probably subclass of undirected graph)
// need to rewrite with traits
#[pymethods] 
impl Graph {
    #[new]
    #[pyo3(signature = (new_nodes=None))]
    pub fn new(new_nodes: Option<&PyDict>) -> PyResult<Self> {
        let mut new_nodes = match new_nodes {
            Some(nodes_val) => converter::convert_pydict_to_hashmap(&nodes_val)?,
            None => {
                let empty_hashmap: HashMap<usize, usize> = HashMap::new();
                empty_hashmap
            }
        };

        let mut sorted_vec: Vec<_> = new_nodes.iter().collect();
        sorted_vec.sort_by_key(|(&key, _)| key);
        new_nodes = sorted_vec
            .into_iter()
            .map(|(&key, &value)| (key, value))
            .collect();

        let new_matrix: Vec<Vec<f32>> = generate_matrix::<usize, usize, f32>(&new_nodes)?;

        Ok( Graph { nodes: new_nodes, matrix: new_matrix } )
    }

    #[staticmethod]
    pub fn new_from_rust(new_nodes: HashMap<usize, usize>, new_matrix: Vec<Vec<f32>>) -> Result<Self> {
        if new_matrix.len() != new_matrix[0].len() {
            let err_msg = format!("Shape of ({}, {}) is not square", new_matrix.len(), new_matrix[0].len());
            return Err(Error::new(ErrorKind::InvalidInput, err_msg)); 
        }

        return Ok(Graph { nodes: new_nodes, matrix: new_matrix } );
    }

    #[pyo3(signature = ())]
    pub fn get_nodes(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        converter::convert_hashmap_to_pydict(py, &self.nodes)
    }

    #[pyo3(signature = (dictionary))]
    pub fn reset_with_nodes(&mut self, dictionary: &PyDict) -> PyResult<()> {
        self.nodes = converter::convert_pydict_to_hashmap(dictionary)?;
        self.matrix = generate_matrix(&self.nodes)?;
        // maybe change to set nodes where
        // check what nodes are new and add accordingly?

        Ok(())
    }

    #[pyo3(signature = ())]
    pub fn get_matrix<'a >(&self, py: Python<'a >) -> &'a PyArray<f32, Dim<[usize; 2]>> {
        PyArray::from_vec2(py, &self.matrix.clone()).unwrap()
    }

    #[pyo3(signature = (start, end, weight=1.0))]
    pub fn edit_edge(&mut self, start: usize, end: usize, weight: f32) -> Result<()> {
        if !self.nodes.contains_key(&start) {
            let err_msg = format!(r#"Start node of "{}" not found"#, start);
            return Err(Error::new(ErrorKind::InvalidInput, err_msg));
        }
        if !self.nodes.contains_key(&end) {
            let err_msg = format!(r#"End node of "{}" not found"#, end);
            return Err(Error::new(ErrorKind::InvalidInput, err_msg));
        }

        if start == end {
            return Err(Error::new(ErrorKind::InvalidInput, "Start node cannot be same as end node"));
        }

        self.matrix[start][end] = weight;

        Ok(())
    }

    #[pyo3(signature = (edges_to_edit))]
    pub fn edit_edges(&mut self, edges_to_edit: &PyList) -> Result<()> {
        for i in edges_to_edit {
            let start = i.get_item(0)?.extract::<usize>()?;
            let end = i.get_item(1)?.extract::<usize>()?;
            let weight = i.get_item(2)?.extract::<f32>()?;

            self.edit_edge(start, end, weight)?;
        }

        Ok(())
    }

    fn add_row(&mut self, row: Vec<f32>) -> Result<()> {
        if row.len() != self.matrix.len() {
            let err_msg = format!("{} is not the same size of {}", row.len(), self.matrix.len());
            return Err(Error::new(ErrorKind::InvalidInput, err_msg));
        }

        self.matrix.push(row);

        Ok(())
    }

    fn add_column(&mut self, column: Vec<f32>) -> Result<()> {
        if column.len() != self.matrix.len() {
            let err_msg = format!("{} is not the same size of {}", column.len(), self.matrix.len());
            return Err(Error::new(ErrorKind::InvalidInput, err_msg));
        }

        for (i, row) in self.matrix.iter_mut().enumerate() {
            row.extend_from_slice(&[column[i]]);
        }

        Ok(())
    }

    #[pyo3(signature=(value=None))]
    pub fn add_vertex(&mut self, value: Option<usize>) -> Result<()> {
        let value = match value {
            Some(value_usize) => value_usize,
            None => 0
        };

        let max_val = *self.nodes.keys().max().ok_or_else(|| {
            return Error::new(ErrorKind::InvalidInput, "Cannot find maximum node key");
        })?;

        self.nodes.insert(max_val + 1, value);

        let new_size = self.matrix.len();
        self.add_row(vec![0.0; new_size])?;
        self.add_column(vec![0.0; new_size + 1])?;

        Ok(())
    }

    #[pyo3(signature = (vertices))]
    pub fn add_vertices(&mut self, vertices: usize) -> Result<()> {
        for _ in 0..vertices {
            self.add_vertex(None)?;
        }

        Ok(())
    }

    fn add_vertices_by_value_vec(&mut self, values: Vec<usize>) -> Result<()> {
        for i in values {
            self.add_vertex(Some(i))?;
        }

        Ok(())
    }

    #[pyo3(signature = (values))]
    pub fn add_vertices_by_value(&mut self, values: PyReadonlyArray1<usize>) -> Result<()> {
        let values = match values.to_vec() {
            Ok(values_vec) => values_vec,
            Err(_e) => return Err(Error::new(ErrorKind::InvalidInput, "Cannot create vector"))
        };

        Ok(self.add_vertices_by_value_vec(values)?)
    }

    // shift all values greater than deleted node down by 1
    // delete respective row and column
    pub fn delete_vertex(&mut self, node: usize) -> Result<()> {
        if !self.nodes.contains_key(&node) {
            let err_msg = format!(r#"Node "{}" not found"#, node);
            return Err(Error::new(ErrorKind::InvalidInput, err_msg));
        }

        let mut new_nodes: HashMap<usize, usize> = HashMap::new();

        self.nodes.remove(&node);
        for (key, value) in &self.nodes {
            let new_key = if key > &node { key - 1 } else { *key };
            new_nodes.insert(new_key, *value);
        }

        self.nodes = new_nodes;

        self.matrix.remove(node);
        for row in &mut self.matrix {
            row.remove(node);
        }

        Ok(())
    }

    #[pyo3(signature = (vertices))]
    pub fn delete_vertices(&mut self, vertices: PyReadonlyArray1<usize>) -> Result<()> {
        let mut vertices = match vertices.to_vec() {
            Ok(vertices_vec) => vertices_vec,
            Err(_e) => return Err(Error::new(ErrorKind::InvalidInput, "Cannot create vector"))
        };

        vertices.sort_by(|a, b| b.cmp(a));

        for i in vertices {
            self.delete_vertex(i)?;
        }

        Ok(())
    }

    #[pyo3(signature = (node, new_value))]
    pub fn update_node_value(&mut self, node: usize, new_value: usize) -> Result<()> {
        if !self.nodes.contains_key(&node) {
            let err_msg = format!(r#"Cannot find node "{}""#, node);
            return Err(Error::new(ErrorKind::InvalidInput, err_msg));
        }

        self.nodes.entry(node).and_modify(|k| *k = new_value);

        Ok(())
    }

    #[pyo3(signature = ())]
    pub fn get_all_edges(&self) -> Vec<(usize, usize, f32)> {
        let mut edges: Vec<(usize, usize, f32)> = Vec::new();
    
        for (start_node, row) in self.matrix.iter().enumerate() {
            for (end_node, weight) in row.iter().enumerate() {
                if *weight != 0.0 {
                    edges.push((start_node, end_node, *weight));
                }
            }
        }
    
        return edges;
    }

    #[pyo3(signature = (node))]
    pub fn is_end_node(&self, node: usize) -> Result<bool> {
        let row = match self.matrix.get(node) {
            Some(row) => row,
            None => return Err(Error::new(ErrorKind::InvalidInput, "Node not found")),
        };

        let col = self.matrix
            .iter()
            .map(|row| Some(*row.get(node).unwrap()))
            .collect::<Vec<Option<f32>>>();

        let mut count = row
            .iter()
            .filter(|&&weight| weight != 0.0)
            .count();
        count += col
            .iter()
            .flatten()
            .filter(|&&weight| weight != 0.0)
            .count();

        Ok(count <= 1)
    }

    #[pyo3(signature = (start_node, disconnected_nodes))]
    pub fn get_connected_component(&self, start_node: usize, disconnected_nodes: Vec<usize>) -> Result<Vec<usize>> {
        if !self.nodes.contains_key(&start_node) {
            return Err(Error::new(ErrorKind::InvalidInput, "Node not found"));
        }

        Ok(get_connected_component(&self.matrix, &self.nodes, start_node, &disconnected_nodes))
    }

    #[pyo3(signature = (node))]
    pub fn get_all_neighbors(&self, node: usize) -> Result<Vec<usize>> {
        if !self.nodes.contains_key(&node) {
            return Err(Error::new(ErrorKind::InvalidInput, "Node not found"));
        }

        let row = self.matrix.get(node).unwrap();
        let col = self.matrix
            .iter()
            .map(|row| Some(*row.get(node).unwrap()))
            .collect::<Vec<Option<f32>>>();
        
        let row_neighbors: Vec<usize> = row.iter()
            .enumerate()
            .filter(|(_, weight)| **weight != 0.0)
            .map(|(idx, _)| idx)
            .collect();
        
        let column_neighbors: Vec<usize> = col.iter()
            .enumerate()
            .filter(|(_, weight)| **weight != Some(0.0))
            .map(|(idx, _)| idx)
            .collect();
        
        let mut neighbors = row_neighbors;
        neighbors.extend(column_neighbors);
        
        Ok(neighbors)

    }

    // get all edges
    // increment nodes key values by maximum self.nodes key value
    // add vertices by value, values obtained from subgraph.nodes.values
    // write edges obtained by get all edges but add maximum key value
    // write edge from first vertex of subgraph to start_node

    // needs to cut off graph at some node
    // cut off some part of graph connected to start node
    // find existing weight and replace it with graph?
    // if end node use existing code
    // if node with other connections, replace one connection with new graph
    // and delete connected nodes before replacing

    #[pyo3(signature = (subgraph, start_node, other_node=None))]
    pub fn replace_subgraph(
        &mut self, 
        subgraph: &Graph, 
        start_node: usize, 
        other_node: Option<usize>
    ) -> Result<()> {
        if !self.nodes.contains_key(&start_node) {
            return Err(Error::new(ErrorKind::InvalidInput, 
                format!(r#"Cannot find node "{}""#, start_node)
            ));
        }

        let max_node_key = *self.nodes.keys().max().ok_or_else(|| {
            return Error::new(ErrorKind::InvalidInput, "Cannot find maximum node key");
        })?;

        let consider_other_node = match other_node {
            Some(_node_val) => true,
            None => false,
        };

        let mut to_delete: Vec<usize> = Vec::new();
        if consider_other_node {
            let to_ignore = self.get_all_neighbors(start_node)?
                .into_iter()
                .filter(|&i| i != other_node.unwrap())
                .collect();
            to_delete = self.get_connected_component(start_node, to_ignore)?;
        }

        let subgraph_nodes_values: Vec<usize> = subgraph.nodes
            .values()
            .cloned()
            .collect();

        self.add_vertices_by_value_vec(subgraph_nodes_values)?;

        let subgraph_edges = subgraph.get_all_edges();

        if subgraph_edges.len() == 0 {
            return Ok(());
        }

        let add_previous_connection = start_node != 0
            &&
            self.matrix[start_node - 1][max_node_key - 1] != 0.0 
            &&
            !self.is_end_node(start_node).unwrap();

        if consider_other_node {
            to_delete.sort();
            to_delete.reverse();
            for i in to_delete {
                self.delete_vertex(i)?;
            }
        }

        if add_previous_connection {
            self.edit_edge(start_node - 1, max_node_key - 1, 1.0)?;
        }

        for (start, end, weight) in subgraph_edges {
            self.edit_edge(start + start_node, end + start_node, weight)?;
        }

        Ok(())
    }

    // HashMap<usize, Vec<usize>>
    // &PyDict
    #[pyo3(signature = (rules))]
    pub fn mutate_types(&mut self, rules: &PyDict) -> Result<()>{
        let keys_to_choose = self.nodes
            .keys()
            .cloned()
            .collect::<Vec<usize>>();
        let vertex_to_mutate = keys_to_choose
            .choose(&mut rand::thread_rng())
            .unwrap();

        let current_val = self.nodes.get(&vertex_to_mutate).unwrap();

        // this should always return a valid value because it was checked above
        let new_type_set = match rules.get_item(current_val) {
            Some(types_to_use) => types_to_use,
            None => return Err(Error::new(ErrorKind::InvalidInput, 
                format!("Rule not found for type {}", current_val)
            )) 
        };

        let new_type_set = new_type_set.extract::<Vec<usize>>()?;

        let new_type = new_type_set.choose(&mut rand::thread_rng());

        match new_type {
            Some(type_value) => {
                self.update_node_value(*vertex_to_mutate, *type_value)?;
            },
            None => {}
        };

        Ok(())
    }

    // adjacency lists to represent what vertices can connect to what other vertices
    // 0 : [] // cannot connect to any other vertices
    // 1 : [2, 3]
    // 2 : [] // can only be connected in one direction
    // 3 : [1, 2]
    // get a vector of vertices
    // choose one and remove it from the vector
    // choose another but only choose one of a connectable valid type
    // if corresponding connecting types is empty then just return
    // HashMap<usize, Vec<usize>>
    // &PyDict
    // Vec<f32>
    #[pyo3(signature = (rules, edges=None))]
    pub fn mutate_edges(&mut self, rules: &PyDict, edges: Option<PyReadonlyArray1<f32>>) -> Result<()> {
        let vertices: Vec<usize> = self.nodes
            .keys()
            .cloned()
            .collect();
        let start_node = *vertices
            .choose(&mut rand::thread_rng())
            .unwrap();

        let vertices: Vec<usize> = vertices
            .into_iter()
            .filter(|i| i != &start_node)
            .collect(); // remove start node

        let valid_types = match rules.get_item(self.nodes.get(&start_node).unwrap()) {
            Some(type_val) => type_val.extract::<Vec<usize>>()?,
            None => {
                let err_msg = format!(r#"No associated rule with node {}"#, 
                    self.nodes.get(&start_node).unwrap());
                return Err(Error::new(ErrorKind::InvalidInput, err_msg));
            }
        };

        let vertices: Vec<_> = vertices
            .into_iter()
            .filter(|i| valid_types.contains(self.nodes.get(i).unwrap()))
            .collect();

        if vertices.len() == 0 {
            return Ok(()); // no edge can be made
        }

        let end_node = *vertices
            .choose(&mut rand::thread_rng())
            .unwrap();

        // change to pick from range
        // *edges_vec.choose(&mut rand::thread_rng()).unwrap()
        let weight = match edges {
            Some(edges_vec) => {
                if edges_vec.len() != 2 {
                    return Err(Error::new(ErrorKind::InvalidInput, "Edges vector is not size 2"));
                }

                let mut rng = rand::thread_rng();
                rng.gen_range(
                    edges_vec
                        .get_item(0)
                        .unwrap()
                        .extract::<f32>()?
                        ..=
                    edges_vec.get_item(1)
                        .unwrap()
                        .extract::<f32>()?
                )
            },
            None => 1.0
        };

        self.edit_edge(start_node, end_node, weight)?;

        Ok(())
    }

    // randomly choose to add or delete
    // check before hand if both vectors are non empty
    // if one is empty just use the other one 
    // if both are empty dont do anything
    // choose nodes that are of valid types
    // add or delete accordingly
    // Vec<usize>
    // &PyReadonlyArray1<usize>
    #[pyo3(signature = (add_types, delete_types))]
    pub fn mutate_vertices(
        &mut self, 
        add_types: PyReadonlyArray1<usize>, 
        delete_types: PyReadonlyArray1<usize>
    ) -> Result<()> {
        let mut rng = rand::thread_rng();
        // let random_bool: bool = rng.gen_bool(0.5)

        let add: bool;
        if add_types.len() != 0 && delete_types.len() != 0 {
            if rng.gen_bool(0.5) {
                add = true;
            } else {
                add = false;
            }
        } else if add_types.len() != 0 && delete_types.len() == 0 {
            add = true;
        } else if add_types.len() == 0 && delete_types.len() != 0 {
            add = false;
        } else {
            return Ok(());
        }

        if add {
            self.add_vertex(
                Some(
                    *add_types
                        .to_vec()
                        .unwrap()
                        .choose(&mut rand::thread_rng())
                        .unwrap()
                )
            )?;

            // get index and randomly choose one
        } else {
            let type_to_delete = delete_types.to_vec().unwrap();

            let type_to_delete = type_to_delete
                .choose(&mut rand::thread_rng())
                .unwrap();

            let vertex_to_delete = self.nodes
                .keys()
                .cloned()
                .collect::<Vec<usize>>()
                .into_iter()
                .filter(|i| self.nodes.get(i) == Some(type_to_delete))
                .collect::<Vec<usize>>();

            let vertex_to_delete = vertex_to_delete.choose(&mut rand::thread_rng());

            match vertex_to_delete {
                Some(vertex) => { self.delete_vertex(*vertex)?; },
                None => {}
            }
        }

        Ok(())
    }

    // choose from self.get_all_edges()
    // change one
    #[pyo3(signature = (lower, upper))]
    pub fn mutate_weights(&mut self, lower: f32, upper: f32) -> Result<()> {
        let possible_edges = self.get_all_edges();

        if possible_edges.len() == 0 {
            return Ok(());
        }

        if lower >= upper {
            return Err(Error::new(ErrorKind::InvalidInput, "Lower bound must be less than upper bound"));
        }

        let mut rng = rand::thread_rng();
        let (start, end, _) = possible_edges
            .choose(&mut rng)
            .unwrap();
        
        let weight = rng.gen_range(lower..=upper);
        self.edit_edge(*start, *end, weight)?;

        Ok(())
    }

    // if values are some
    // add by vertice using a random choice from that set of values
    // randomly add 1.0s for edges unless some
    #[staticmethod]
    #[pyo3(signature = (size_range, prob, edge_range=None, values=None))]
    pub fn create_random(
        size_range: (usize, usize), 
        prob: f32,
        edge_range: Option<(f32, f32)>, 
        values: Option<Vec<usize>>
    ) -> Result<Graph> {
        let values = match values {
            Some(values_vec) => values_vec,
            None => vec![]
        };

        let mut nodes: HashMap<usize, usize> = HashMap::new();
        
        let mut rng = rand::thread_rng();

        if size_range.0 == 0 {
            return Err(Error::new(ErrorKind::InvalidInput, "Size range lower bound must be greater than 0"));
        }

        let size: usize;

        if size_range.0 == size_range.1 {
            size = size_range.0;
        } else if size_range.0 < size_range.1{
            size = rng.gen_range(size_range.0..=size_range.1);
        } else {
            return Err(Error::new(ErrorKind::InvalidInput, "Size lower bound must be equal to or less than upper bound"));
        }

        for i in 0..size {
            if values.len() != 0 {
                nodes.insert(i, values[rng.gen_range(0..values.len())]);
            } else {
                nodes.insert(i, 0);
            }
        }

        let matrix = generate_matrix::<usize, usize, f32>(&nodes)?;
        let mut graph = Graph::new_from_rust(nodes, matrix)?;

        for i in 0..size {
            for j in 0..size {
                if rng.gen::<f32>() <= prob && i != j {
                    match edge_range {
                        Some(edge_range_val) => { 
                            let entry = rng.gen_range(edge_range_val.0..=edge_range_val.1);
                            graph.matrix[i][j] = entry;
                        },
                        None =>  { graph.matrix[i][j] = 1.0; },
                    };
                }
            }
        }

        Ok(graph)
    }
}

#[pyfunction]
#[pyo3(signature = (graph, return_vertices=false))]
pub fn isolate_random_subgraph(graph: &Graph, return_vertices: bool) -> (Graph, Option<HashSet<usize>>) {
    let num_vertices = graph.matrix.len();

    // randomly select a subset of vertices to include in the subgraph
    let mut rng = rand::thread_rng();
    let mut selected_vertices: Vec<usize> = (0..num_vertices).collect();

    if num_vertices > 1 {
        let lower_selected = rng.gen_range(0..=num_vertices-1);
        let num_selected = rng.gen_range(lower_selected+1..=num_vertices);

        // selected_vertices.truncate(num_selected);
        selected_vertices = selected_vertices[lower_selected..num_selected].to_vec();
    } else {
        selected_vertices = vec![0];
    }

    let selected_set: HashSet<usize> = selected_vertices.clone().into_iter().collect();

    // extract the submatrix corresponding to the selected vertices
    let sub_matrix: Vec<Vec<f32>> = graph.matrix
        .iter()
        .enumerate()
        .filter(|(i, _)| selected_set.contains(i))
        .map(
            |(_, row)| row
                .iter()
                .enumerate()
                .filter(|(j, _)| selected_set.contains(j)).map(|(_, &val)| val)
                .collect()
        )
        .collect();

    let mut sub_nodes: HashMap<usize, usize> = HashMap::new();
    for (n, i) in selected_vertices.iter().enumerate() {
        sub_nodes.insert(n, *graph.nodes.get(&i).unwrap());
    }

    match return_vertices {
        true => return (Graph::new_from_rust(sub_nodes, sub_matrix).unwrap(), Some(selected_set)),
        false => return (Graph::new_from_rust(sub_nodes, sub_matrix).unwrap(), None),
    };
}

fn get_connected_component(
    matrix: &Vec<Vec<f32>>, 
    nodes: &HashMap<usize, usize>, 
    start_node: usize,
    disconnected_nodes: &Vec<usize>
) -> Vec<usize> {
    let mut visited: Vec<bool> = vec![false; nodes.len()];
    let mut connected_component: Vec<usize> = Vec::new();

    depth_first_search(matrix, start_node, &mut visited, &mut connected_component, disconnected_nodes);

    connected_component
}

fn depth_first_search(
    matrix: &Vec<Vec<f32>>, 
    node: usize, 
    visited: &mut Vec<bool>, 
    connected_component: &mut Vec<usize>,
    disconnected_nodes: &Vec<usize>
) {
    visited[node] = true;
    connected_component.push(node);

    for (adj_node, weight) in matrix[node].iter().enumerate() {
        if *weight != 0.0 && !visited[adj_node] && !disconnected_nodes.contains(&adj_node) {
            depth_first_search(matrix, adj_node, visited, connected_component, disconnected_nodes);
        }
    }
}

// swap randomly chosen node with a given subgraph
#[pyfunction]
pub fn crossover(parent1: &Graph, parent2: &Graph, r_cross: f32) -> Result<(Graph, Graph)> {
    let mut clone1 = parent1.clone();
    let mut clone2 = parent2.clone();

    let mut rng_thread = rand::thread_rng();

    if rng_thread.gen::<f32>() > r_cross {
        return Ok((clone1, clone2));
    }

    let (sub_graph1, vertices1) = isolate_random_subgraph(&clone1, true);
    let (sub_graph2, vertices2) = isolate_random_subgraph(&clone2, true);

    let mut vertices1: Vec<_> = vertices1
        .unwrap()
        .iter()
        .cloned()
        .collect();
    vertices1.sort();
    let mut vertices2: Vec<_> = vertices2
        .unwrap()
        .iter()
        .cloned()
        .collect();
    vertices2.sort();

    let crossover_point1 = vertices1.clone().into_iter().min().ok_or_else(|| {
        return Error::new(ErrorKind::InvalidInput, "Cannot find minimum vertices");
    })?;
    let crossover_point2 = vertices2.clone().into_iter().min().ok_or_else(|| {
        return Error::new(ErrorKind::InvalidInput, "Cannot find minimum vertices");
    })?;

    if clone1.is_end_node(crossover_point1).unwrap() || crossover_point1 == 0 { 
        match clone1.replace_subgraph(&sub_graph2, crossover_point1, None) {
            Ok(_result) => {},
            Err(e) => { return Err(Error::new(ErrorKind::Other, format!("clone 1 end case error: {}", e.to_string()))) },
        };
    } else {
        // get other node, replace in correct direction
        // vertices1.get(1).cloned()
        match clone1.replace_subgraph(&sub_graph2, crossover_point1 - 1, Some(crossover_point1)) {
            Ok(_result) => {},
            Err(e) => { return Err(Error::new(ErrorKind::Other, format!("clone 1: {}", e.to_string()))) },
        };
    }

    if clone2.is_end_node(crossover_point2).unwrap() || crossover_point2 == 0 { 
        match clone2.replace_subgraph(&sub_graph1, crossover_point2, None) {
            Ok(_result) => {},
            Err(e) => { return Err(Error::new(ErrorKind::Other, format!("clone 2 end case error: {}", e.to_string()))) },
        };
    } else {
        // get other node, replace in correct direction
        // vertices2.get(1).cloned()
        match clone2.replace_subgraph(&sub_graph1, crossover_point2 - 1, Some(crossover_point2)) {
            Ok(_result) => {},
            Err(e) => { return Err(Error::new(ErrorKind::Other, format!("clone 2 error: {}", e.to_string()))) },
        };
    }

    return Ok((clone1, clone2));
}

// https://web.mit.edu/rust-lang_v1.25/arch/amd64_ubuntu1404/share/doc/rust/html/book/first-edition/traits.html

// https://pyo3.rs/main/trait_bounds

// https://stackoverflow.com/questions/60540328/how-can-i-add-functions-with-different-arguments-and-return-types-to-a-vector

// enum RuleSet {
//     TypeRule(fn(&mut Graph, &HashMap<usize, Vec<usize>>)),
//     VertexRule(fn(&mut Graph, &Vec<usize>, &Vec<usize>)),
//     EdgeRule(fn(&mut Graph, &HashMap<usize, Vec<usize>>, Option<Vec<f32>>)),
//     WeightRule(fn(&mut Graph, f32, f32)),
// }

// if signatures between the same rules vary scrap this

enum RuleSet {
    TypeRule,
    VertexRule,
    EdgeRule,
    WeightRule,
}

#[pyfunction]
#[pyo3(signature = (
    graph, 
    r_mut, 
    *,
    reps=None,
    type_rules=None, 
    vertex_rules=None, 
    edge_rules=None, 
    weight_rules=None, 
    possible_edges=None
))]
// reps: Option<usize>,
// HashMap<usize, Vec<usize>>
// Vec<usize>, Vec<usize>
// HashMap<usize, Vec<usize>>
// Vec<f32>
pub fn mutate(
    mut graph: Graph,
    r_mut: f32,
    reps: Option<usize>,
    type_rules: Option<&PyDict>,
    vertex_rules: Option<(PyReadonlyArray1<usize>, PyReadonlyArray1<usize>)>, 
    edge_rules: Option<&PyDict>,
    weight_rules: Option<(f32, f32)>,
    possible_edges: Option<PyReadonlyArray1<f32>>,
) -> Result<Graph> {
    let mut rng_thread = rand::thread_rng();

    let reps = match reps { 
        Some(reps_val) => reps_val,
        None => 1,
    };

    // place all the rules into a vector
    // randomly select one of these rulesets to execute
    // use match to match rules with function

    let mut rule_sets: Vec<_> = Vec::new();

    match &type_rules {
        Some(_type_rule_set) => { rule_sets.push(RuleSet::TypeRule); },
        None => {}
    }

    match &vertex_rules {
        Some(_vertex_rule_set) => { rule_sets.push(RuleSet::VertexRule); },
        None => {}
    }

    match &edge_rules {
        Some(_edge_rule_set) => { rule_sets.push(RuleSet::EdgeRule); },
        None => {}
    }

    match &weight_rules {
        Some(_weight_rule_set) => { rule_sets.push(RuleSet::WeightRule); },
        None => {}
    }

    if rule_sets.len() == 0 {
        return Err(Error::new(ErrorKind::InvalidInput, "Must have at least one rule set"));
    }

    for _ in 0..reps {
        if rng_thread.gen::<f32>() <= r_mut {
            let mutation_type = rule_sets
                .choose(&mut rng_thread)
                .unwrap();

            match mutation_type {
                RuleSet::TypeRule => { graph.mutate_types(type_rules.unwrap())?; },
                RuleSet::VertexRule => { 
                    let (add_types, delete_types) = vertex_rules.clone().unwrap();
                    graph.mutate_vertices(
                        add_types, 
                        delete_types
                    )?;
                },
                RuleSet::EdgeRule => { graph.mutate_edges(edge_rules.unwrap(), possible_edges.clone())?; },
                RuleSet::WeightRule => { 
                    let (lower, upper) = weight_rules.unwrap();

                    graph.mutate_weights(
                        lower, 
                        upper
                    )?; 
                },
            }
        }
    }
    
    Ok(graph)
}

fn use_func(
    obj: &PyAny, 
    graph: Graph,
    settings: &PyDict
) -> PyResult<f32> {
    if !obj.is_callable() {
        return Err(Error::new(ErrorKind::InvalidInput, "Not callable").into());
    }
    
    let out = obj.call((graph, settings), None)?;
    let out: f32 = out.extract()?; 

    return Ok(out);
}

#[pyfunction]
#[pyo3(signature = (
    *,
    objective_func,
    n_iter,
    n_pop,
    r_cross,
    r_mut,
    k,
    size_range,
    initialization_prob,
    settings,
    parallel,
    reps,
    edge_range,
    values,
    type_rules, 
    vertex_rules, 
    edge_rules, 
    weight_rules,
    possible_edges,
))]
pub fn graph_genetic_algo(
    objective_func: &PyAny,
    n_iter: usize, 
    n_pop: usize, 
    r_cross: f32,
    r_mut: f32, 
    k: usize, 
    size_range: (usize, usize),
    initialization_prob: f32,
    settings: &PyDict,
    parallel: bool,
    reps: Option<usize>,
    edge_range: Option<(f32, f32)>, 
    values: Option<Vec<usize>>,
    type_rules: Option<&PyDict>,
    vertex_rules: Option<(PyReadonlyArray1<usize>, PyReadonlyArray1<usize>)>, 
    edge_rules: Option<&PyDict>,
    weight_rules: Option<(f32, f32)>,
    possible_edges: Option<PyReadonlyArray1<f32>>,
) -> Result<(Graph, f32, Vec<Vec<f32>>)> {
    if n_pop % 2 != 0 {
        return Err(Error::new(ErrorKind::InvalidInput, "n_pop must be even for crossover"))
    }

    let pop: Result<Vec<Graph>> = (0..n_pop)
        .map(|_x| Graph::create_random(size_range, initialization_prob, edge_range, values.clone()))
        .collect();

    let mut pop = match pop {
        Ok(pop_value) => pop_value,
        Err(e) => return Err(e)
    };

    // let (mut best, mut best_eval) = (&pop[0], objective(&pop[0], &bounds, n_bits, &settings));
    let f = |graph: Graph, settings: &PyDict| { use_func(objective_func, graph, settings) };

    let mut best = pop[0].clone();
    let mut best_eval = match f(pop[0].clone(), settings) {
        Ok(best_eval_result) => best_eval_result,
        Err(e) => return Err(Error::new(ErrorKind::Other, e)),
    };

    let mut all_scores = Vec::<Vec<f32>>::new();

    for gen in 0..n_iter {
        println!("gen: {}", gen + 1);
        // calculate scores
        let scores_results: &PyResult<Vec<f32>> = &pop
            .iter() 
            .map(|p| f(p.clone(), settings))
            .collect();

        // have option to execute this through threadpoolexecutor and pool the results
        // Option<usize> for worker parameter
        // if none do this
        // if some use usize to specify amount of workers

        // https://github.com/PyO3/pyo3/issues/1485

        // dump contents to json
        // pass json to command
        // overhead problems?
        // open various processes and dump them to file and have process constantly read it
        // have process state all processes finished
        
        // check if objective failed anywhere
        let scores = match scores_results {
            Ok(scores_results) => scores_results,
            Err(e) => return Err(Error::new(ErrorKind::Other, e.to_string())),
        };

        all_scores.push(scores.clone());

        for i in 0..n_pop {
            if scores[i] < best_eval {
                best = pop[i].clone();
                best_eval = scores[i];
                println!("score: {}", &scores[i]);
            }
        }

        // generate next population
        let selected: Vec<Graph> = if parallel {
            let new_graphs = (0..n_pop)
                .into_par_iter()
                .map(|_| selector::selection(&pop, &scores, k))
                .collect();

            new_graphs
        } else {
            let new_graphs = (0..n_pop)
                .map(|_| selector::selection(&pop, &scores, k))
                .collect();

            new_graphs
        };

        // let children: Vec<Graph> = if parallel {
        //     let children_vec = (0..n_pop)
        //         .into_par_iter()
        //         .step_by(2)
        //         .flat_map(|i| {
        //             let new_children = match crossover(&selected[i], &selected[i + 1], r_cross) {
        //                 Ok(child_val) => child_val, 
        //                 Err(e) => return Err(e)
        //             };

        //             Ok(vec![new_children.0, new_children.1])
        //         })
        //         .flatten()
        //         .map(|mut child| {
        //             mutate(child, r_mut, reps,
        //             type_rules, vertex_rules, edge_rules,
        //             weight_rules, possible_edges)?;

        //             Ok(child)
        //         })
        //         .collect()?;

        //     children_vec
        // } else {
        //     let mut children_vec: Vec<Graph> = Vec::new();
        //     for i in (0..n_pop).step_by(2) {
        //         let new_children = crossover(&selected[i], &selected[i+1], r_cross)?;
        //         for mut child in vec![new_children.0, new_children.1] {
        //             mutate(child, r_mut, reps,
        //             type_rules, vertex_rules, edge_rules,
        //             weight_rules, possible_edges);
        //             children_vec.push(child);
        //         }
        //     }

        //     children_vec
        // }; 

        let mut children: Vec<Graph> = Vec::new();
        for i in (0..n_pop).step_by(2) {
            let new_children = match crossover(&selected[i], &selected[i+1], r_cross) {
                Ok(crossover_output) => crossover_output,
                Err(_e) => (selected[i].clone(), selected[i+1].clone())
            };
            for child in vec![new_children.0, new_children.1] {
                let child = mutate(child, r_mut, reps,
                    type_rules, vertex_rules.clone(), edge_rules,
                    weight_rules, possible_edges.clone())?;
                children.push(child);
            }
        }

        pop = children;
    }

    return Ok((best, best_eval, all_scores));
}
