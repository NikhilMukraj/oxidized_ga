use pyo3::prelude::*;
use pyo3::types::PyDict;
use numpy::{PyReadonlyArray1, PyArray1};
use rand::Rng;
use std::io;
use std::io::{Result, Error, ErrorKind};
use rayon::prelude::*;
pub mod neural_net;
pub mod graph;
#[path = "./selector/mod.rs"] mod selector;


#[derive(Clone)]
#[pyclass]
struct BitString {
    #[pyo3(get)]
    string: String
}

#[pymethods]
impl BitString {
    #[new]
    #[pyo3(signature = (new_string = String::from("")))]
    fn new(new_string: String) -> PyResult<Self> {
        let new_bistring = BitString { string: new_string };

        if new_bistring.is_binary() {
            return Ok(new_bistring);
        } else {
            return Err(Error::new(ErrorKind::InvalidInput, "Non binary found").into()); 
        }
    }

    fn is_binary(&self) -> bool {
        self.string.chars().all(|c| c == '0' || c == '1')
    }

    #[pyo3(signature = (new_string))]
    fn set(&mut self, new_string: String) -> io::Result<()> {
        // check after initalization

        self.string = new_string;

        if self.is_binary() {
            return Ok(());
        } else {
            return Err(Error::new(ErrorKind::Other, "Non binary found"));
        }
    }

    #[pyo3(signature = ())]
    fn length(&self) -> i32 {
        self.string.len() as i32
    }
}

fn rust_decode(
    bitstring: &str, 
    lower_bounds: Vec<f32>, 
    upper_bounds: Vec<f32>,
    n_bits: i32
) -> Result<Vec<f32>> {
    // decode for non variable length
    // for variable length just keep bounds consistent across all
    // determine substrings by calculating string.len() / n_bits
    if lower_bounds.len() != upper_bounds.len() {
        return Err(Error::new(ErrorKind::Other, "Upper bounds length does not match lower bounds length"));
    }
    if lower_bounds.len() != bitstring.len() / n_bits as usize {
        return Err(Error::new(ErrorKind::Other, "Bounds length does not match n_bits"));
    }
    if bitstring.len() % n_bits as usize != 0 {
        return Err(Error::new(ErrorKind::Other, "String length is indivisible by n_bits"));
    }

    let maximum = i32::pow(2, n_bits as u32) as f32 - 1.;
    let mut decoded_vec = vec![0.; lower_bounds.len()];

    let n_bits = n_bits as usize;
    for i in 0..lower_bounds.len() {
        let (start, end) = (i * n_bits, (i * n_bits) + n_bits);
        let substring = &bitstring[start..end];

        let mut value = match i32::from_str_radix(substring, 2) {
            Ok(value_result) => value_result as f32,
            Err(_e) => return Err(Error::new(ErrorKind::Other, "Non binary found")),
        };
        // using unwrap because whether the index is valid is calculated above
        value = value * (upper_bounds.get(i).unwrap() - lower_bounds.get(i).unwrap()) / maximum + lower_bounds.get(i).unwrap();

        decoded_vec[i] = value;
    }

    return Ok(decoded_vec);
}

#[pyfunction]
fn decode(
    bitstring: &str, 
    lower_bounds: PyReadonlyArray1<f32>, 
    upper_bounds: PyReadonlyArray1<f32>,
    n_bits: i32
) -> Result<Vec<f32>> {
    let lower_bounds = match lower_bounds.to_vec() {
        Ok(lower_bounds_vec) => lower_bounds_vec,
        Err(_e) => return Err(Error::new(ErrorKind::InvalidInput, "Cannot convert lower bounds to vector")),
    };
    let upper_bounds = match upper_bounds.to_vec() {
        Ok(upper_bounds_vec) => upper_bounds_vec,
        Err(_e) => return Err(Error::new(ErrorKind::InvalidInput, "Cannot convert upper bounds to vector")),
    };
    
    match rust_decode(bitstring, lower_bounds, upper_bounds, n_bits) {
        Ok(decoded_vec) => return Ok(decoded_vec),
        Err(e) => return Err(Error::new(ErrorKind::InvalidInput, e.to_string())),
    }
}

#[pyfunction]
fn use_func(
    obj: &PyAny, 
    bitstring: &BitString, 
    lower_bounds: &PyArray1<f32>, 
    upper_bounds: &PyArray1<f32>,
    n_bits: i32, 
    settings: &PyDict
) -> PyResult<f32> {
    if !obj.is_callable() {
        return Err(Error::new(ErrorKind::InvalidInput, "Not callable").into());
    }
    
    let out = obj.call((&bitstring.string, lower_bounds, upper_bounds, n_bits, settings), None)?;
    let out: f32 = out.extract()?; 
    return Ok(out);
}

fn create_random_string(length: usize) -> BitString {
    let mut rng_thread = rand::thread_rng(); 
    let mut random_string = String::from("");
    for _ in 0..length {
        if rng_thread.gen::<f32>() <= 0.5 {
            random_string.push('0');
        } else {
            random_string.push('1');
        }
    }

    return BitString {string: random_string};
}

fn crossover(parent1: &BitString, parent2: &BitString, r_cross: f32) -> (BitString, BitString) {
    let mut rng_thread = rand::thread_rng(); 
    let (mut clone1, mut clone2) = (parent1.clone(), parent2.clone());

    if rng_thread.gen::<f32>() <= r_cross {
        let end_point = parent1.length();
        let crossover_point = rng_thread.gen_range(1..end_point); // change for variable length
        
        // c1 = p1[:pt] + p2[pt:]
		// c2 = p2[:pt] + p1[pt:]

        let string1 = format!("{}{}", &parent1.string[0..crossover_point as usize], &parent2.string[crossover_point as usize..]);
        let string2 = format!("{}{}", &parent2.string[0..crossover_point as usize], &parent1.string[crossover_point as usize..]);

        clone1.set(string1).expect("Error setting bitstring");
        clone2.set(string2).expect("Error setting bitstring");
    }

    return (clone1, clone2);
}

fn mutate(bitstring: &mut BitString, r_mut: f32) {
    let mut rng_thread = rand::thread_rng(); 
    for i in 0..bitstring.length() as usize {
        let do_mut = rng_thread.gen::<f32>() <= r_mut;

        // does in place bit flip if do_mut
        if do_mut && bitstring.string.chars().nth(i).unwrap() == '1' {
            bitstring.string.replace_range(i..i+1, "0"); 
        } else if do_mut && bitstring.string.chars().nth(i).unwrap() == '0' {
            bitstring.string.replace_range(i..i+1, "1");
        } 
    }
}

#[pyfunction]
#[pyo3(signature = (
    objective_func,
    lower_bounds,
    upper_bounds,
    n_bits,
    n_iter,
    n_pop,
    r_cross,
    r_mut,
    k,
    settings,
    parallel,
    total_n_bits=None,
    print_output=None
))]
fn genetic_algo(
    objective_func: &PyAny,
    lower_bounds: &PyArray1<f32>, 
    upper_bounds: &PyArray1<f32>, 
    n_bits: i32, 
    n_iter: usize, 
    n_pop: usize, 
    r_cross: f32,
    r_mut: f32, 
    k: usize, 
    settings: &PyDict,
    parallel: Option<(&PyAny, usize)>,
    total_n_bits: Option<usize>,
    print_output: Option<bool>
) -> Result<(BitString, f32, Vec<Vec<f32>>)> {
    if n_pop % 2 != 0 {
        return Err(Error::new(ErrorKind::InvalidInput, "n_pop must be even for crossover"))
    }
    
    let mut pop: Vec<BitString> = match total_n_bits {
        Some(total_val) => {
            (0..n_pop)
                .map(|_x| create_random_string(total_val))
                .collect()
        },
        None => {
            (0..n_pop)
                .map(|_x| create_random_string(n_bits as usize * lower_bounds.len()))
                .collect()
        }
    };
    // let (mut best, mut best_eval) = (&pop[0], objective(&pop[0], &bounds, n_bits, &settings));
    let f = |bitstring: &BitString, 
             lower_bounds: &PyArray1<f32>,
             upper_bounds: &PyArray1<f32>, 
             n_bits: i32, 
             settings: &PyDict| 
        { use_func(objective_func, bitstring, lower_bounds, upper_bounds, n_bits, settings) };

    let mut best = pop[0].clone();
    let mut best_eval = match f(&pop[0], lower_bounds, upper_bounds, n_bits, settings) {
        Ok(best_eval_result) => best_eval_result,
        Err(e) => { 
            let err_msg = format!("\nObjective function error: {}", e.to_string());
            return Err(Error::new(ErrorKind::Other, err_msg));
        },
    };

    let mut all_scores = Vec::<Vec<f32>>::new();

    let workers = match parallel {
        Some(parallel_tuple) => parallel_tuple.1,
        None => 0 
    };

    let print_output = match print_output {
        Some(print_output_bool) => print_output_bool,
        None => true
    };

    for gen in 0..n_iter {
        if print_output { println!("gen: {}", gen + 1); };
        // calculate scores
        let scores: Vec<f32> = match workers {       
            2.. => {
                let (parallel_func, workers) = parallel.unwrap();
                let bitstrings: Vec<_> = pop
                    .iter()
                    .map(|i| i.string.clone())
                    .collect();

                let out = parallel_func.call((objective_func, workers, bitstrings, 
                    (lower_bounds, upper_bounds, n_bits, settings)), None)?;
                
                let out = match out.extract::<Vec<f32>>() {
                    Ok(out_val) => out_val,
                    Err(e) => { 
                        let err_msg = format!("\nObjective function error: {}", e.to_string());
                        return Err(Error::new(ErrorKind::Other, err_msg));
                    }
                };

                out
            }
            _ => {
                    let out: &PyResult<Vec<f32>> = &pop
                        .iter() 
                        .map(|p| f(p, lower_bounds, upper_bounds, n_bits, settings))
                        .collect();

                    let out = match out {
                        Ok(out_val) => out_val,
                        Err(e) => { 
                            let err_msg = format!("\nObjective function error: {}", e.to_string());
                            return Err(Error::new(ErrorKind::Other, err_msg));
                        }
                    };

                    out.to_vec()
                }
        };
        // check if there is an error if so then return an error

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
        // let scores = match scores_results {
        //     Ok(scores_results) => scores_results,
        //     Err(e) => return Err(Error::new(ErrorKind::Other, e.to_string())),
        // };

        all_scores.push(scores.clone());

        for i in 0..n_pop {
            if scores[i] < best_eval {
                best = pop[i].clone();
                best_eval = scores[i];
                if print_output { println!("new string: {}, score: {}", &pop[i].string, &scores[i]); };
            }
        }

        // generate next population
        let selected: Vec<BitString> = match workers {
            1.. => {
                    let new_strings = (0..n_pop)
                    .into_par_iter()
                    .map(|_| selector::selection(&pop, &scores, k))
                    .collect();

                new_strings
            },
            _ => {
            let new_strings = (0..n_pop)
                .map(|_| selector::selection(&pop, &scores, k))
                .collect();

            new_strings
            }
        };

        let children: Vec<BitString> = match workers {
            1.. => {
                let children_vec = (0..n_pop)
                .into_par_iter()
                .step_by(2)
                .flat_map(|i| {
                    let new_children = crossover(&selected[i], &selected[i + 1], r_cross);
                    vec![new_children.0, new_children.1]
                })
                .map(|mut child| {
                    mutate(&mut child, r_mut);
                    child
                })
                .collect();

                children_vec
            },
            _ => {
                let mut children_vec: Vec<BitString> = Vec::new();
                for i in (0..n_pop).step_by(2) {
                    let new_children = crossover(&selected[i], &selected[i+1], r_cross);
                    for child in vec![new_children.0, new_children.1].iter_mut() {
                        mutate(child, r_mut);
                        children_vec.push(child.clone());
                    }
                }

                children_vec
            }
        };

        pop = children;
    }

    return Ok((best, best_eval, all_scores));
}

// use 8 bits to represent the activation function and spread 5 states as equally as possible
#[pyfunction]
fn neural_network_get_bits(
    precision: usize, 
    activation_precision: usize, 
    architecture: &PyDict
) -> PyResult<(usize, usize)> {
    let mut total_neuron_vals = 0;
    let mut activation_vals = 0;

    for i in architecture.values() {
        for j in i.get_item(0)?.iter()? {
            match j?.extract::<neural_net::Activation>() {
                Ok(_item) => { continue },
                Err(e) =>  { return Err(Error::new(ErrorKind::InvalidInput, e.to_string()).into()); },
            };
        }

        let mut activations = i.get_item(0)?.len()?;
        if activations == 1 {
            activations = 0;
        } else {
            activations = 1;
        }

        activation_vals += activation_precision * activations;

        let outer_shape = i.get_item(1)?.get_item(0)?.extract::<usize>()?;
        let inner_shape = i.get_item(1)?.get_item(1)?.extract::<usize>()?;

        let layer_vals = outer_shape * inner_shape + outer_shape;
        total_neuron_vals += precision * layer_vals;
    }

    return Ok((total_neuron_vals, activation_vals));
}

fn reshape_array(array: Vec<f32>, shape: &Vec<usize>) -> Vec<Vec<f32>> {
    let num_rows = shape[0];
    let num_cols = shape[1];
    
    array
        .chunks(num_cols)
        .take(num_rows)
        .map(|chunk| chunk.to_vec())
        .collect()
}

#[pyfunction]
fn neural_network_decode(
    bitstring: &BitString, 
    lower_bounds: f32, // PyReadonlyArray1<f32>, 
    upper_bounds: f32, // PyReadonlyArray1<f32>,
    n_bits: i32, 
    activation_precision: usize,
    architecture: &PyDict
) -> PyResult<neural_net::NeuralNetwork> {
    let (neuron_vals, activation_vals) = match neural_network_get_bits(n_bits as usize, activation_precision, architecture) {
        Ok(get_bits_val) => get_bits_val,
        Err(e) => return Err(e),
    };

    if bitstring.string.len() != neuron_vals + activation_vals {
        let err_msg = format!(
            "{} length of bitstring does not match required amount of {}", 
            bitstring.string.len(),
            neuron_vals + activation_vals
        );
        return Err(Error::new(ErrorKind::InvalidInput, err_msg).into())
    }

    // repeat negative lower bounds and upper bounds for length of neural_vals
    let lower_bounds_arr = vec![lower_bounds; neuron_vals / n_bits as usize];
    let upper_bounds_arr = vec![upper_bounds; neuron_vals / n_bits as usize];
    let decoded_vec = rust_decode(
        &bitstring.string[0..neuron_vals], 
        lower_bounds_arr, 
        upper_bounds_arr, 
        n_bits
    )?;

    let mut start_index = 0;
    let mut raw_weights_biases_vec: Vec<Vec<f32>> = Vec::new();
    let chunk_sizes: Vec<usize> = architecture.values()
        .iter()
        .map(|i| {
            let outer_shape = i.get_item(1).unwrap().get_item(0).unwrap().extract::<usize>().unwrap();
            let inner_shape = i.get_item(1).unwrap().get_item(1).unwrap().extract::<usize>().unwrap();

            let layer_vals = outer_shape * inner_shape + outer_shape;
            layer_vals
        })
        .collect();

    for &size in &chunk_sizes {
        let end_index = start_index + size;
        let chunk = &decoded_vec[start_index..end_index];
        start_index = end_index;

        raw_weights_biases_vec.push(chunk.to_vec());
    }
    
    // deconstruct weights and biases into weights and biases layer by layer
    // decode activation as well

    let mut layers: Vec<neural_net::Layer> = Vec::new(); 
    let mut activation_index = neuron_vals;
    for (neurons, value) in raw_weights_biases_vec.iter().zip(architecture.values().iter()) {
        // reshape and push into layers vec
        let shape: Vec<usize> = value.get_item(1).unwrap()
            .iter()
            .expect("Cannot iterate over shape")
            .map(|i| i.expect("Cannot access item").extract::<usize>().expect("Cannot extract value from shape"))
            .collect();
        let raw_weights = &neurons[0..(shape[0] * shape[1])];
        let weights = reshape_array(raw_weights.to_vec(), &shape);
        let biases = &neurons[(shape[0] * shape[1])..]; 

        // decode activation function
        // use activation precision and tally up how many bits until the current index

        let current_len = value.get_item(0).unwrap().len().unwrap();
        let mut get_activation = 0;
        if current_len != 1 {
            let activation_chunk = &bitstring.string[(activation_index)..(activation_index + activation_precision)];
            get_activation = rust_decode(
                activation_chunk, 
                vec![0.0], 
                vec![(current_len - 1) as f32], 
                activation_precision as i32
            ).unwrap()[0] as usize;
            
            activation_index += activation_precision;
        }

        layers.push(neural_net::Layer::new_with_vec(
            shape[0], 
            weights, 
            biases.to_vec(), 
            value
                .get_item(0)
                .unwrap()
                .get_item(get_activation)
                .unwrap()
                .extract::<neural_net::Activation>()
                .unwrap() // neural_net::Activation::ReLU
        )?);
    }

    return Ok(neural_net::NeuralNetwork::rust_new(layers.len(), layers)?);
}

// #[pyfunction]
// fn execute_python_code<'a>(py: Python<'a >, x: i32, code: &'a str) -> PyResult<Vec<f32>> {
//     let locals = PyDict::new(py);
//     locals.set_item("x", x)?;

//     // Execute the Python code
//     py.run(code, Some(locals), None)?;

//     let result = locals
//         .get_item("result")
//         .unwrap()
//         .extract::<Vec<f32>>()?;

//     // let values: Vec<f32> = result.extract(py)?;

//     Ok(result)
// }

#[pymodule]
#[pyo3(name = "oxidized_ga")]
fn oxidized_ga(_py: Python, m: &PyModule) -> PyResult<()> {
    // python3 -m venv .venv
    // source .venv/bin/activate
    // pip install -U pip maturin
    m.add_function(wrap_pyfunction!(decode, m)?)?;
    m.add_function(wrap_pyfunction!(genetic_algo, m)?)?;

    m.add_class::<BitString>()?;

    m.add_class::<neural_net::Activation>()?;
    m.add_class::<neural_net::Layer>()?;
    m.add_class::<neural_net::NeuralNetwork>()?;

    m.add_function(wrap_pyfunction!(neural_network_get_bits, m)?)?;
    m.add_function(wrap_pyfunction!(neural_network_decode, m)?)?;

    m.add_class::<graph::Graph>()?;
    m.add_function(wrap_pyfunction!(graph::isolate_random_subgraph, m)?)?;
    m.add_function(wrap_pyfunction!(graph::crossover, m)?)?;
    m.add_function(wrap_pyfunction!(graph::mutate, m)?)?;
    m.add_function(wrap_pyfunction!(graph::graph_genetic_algo, m)?)?;

    Ok(())
}
