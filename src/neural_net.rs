// use ndarray to vectorize operations
use ndarray::Dim;
// use numpy::NotContiguousError;
use numpy::{PyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::PyResult;
use std::io::{Result, Error, ErrorKind};
#[path = "./converter/mod.rs"] mod converter;
// mod converter;


#[derive(Clone)]
#[pyclass]
pub enum Activation {
    ReLU,
    Sigmoid,
    Tanh,
    Swish,
    ELU,
}

#[pymethods]
impl Activation {
    #[pyo3(signature = (x))]
    pub fn calculate(&self, x: f32) -> f32 {
        match self {
            Activation::ReLU => self.relu(x),
            Activation::Sigmoid => self.sigmoid(x),
            Activation::Tanh => self.tanh(x),
            Activation::Swish => self.swish(x),
            Activation::ELU => self.elu(x),
        }
    }

    #[pyo3(signature = (x))]
    pub fn relu(&self, x: f32) -> f32 {
        if x > 0.0 { x } else { 0.0 }
    }

    #[pyo3(signature = (x))]
    pub fn sigmoid(&self, x: f32) -> f32 {
        1.0 / (1.0 + (-1.0 * x).exp())
    }

    #[pyo3(signature = (x))]
    pub fn tanh(&self, x: f32) -> f32 {
        x.tanh()
    }

    #[pyo3(signature = (x))]
    pub fn swish(&self, x: f32) -> f32 {
        x / (1.0 + (-1.0 * x).exp())
    }

    #[pyo3(signature = (x))]
    pub fn elu(&self, x: f32) -> f32 {
        if x > 0.0 { x } else { x.exp() - 1.0 }
    }
}

#[derive(Clone)]
#[pyclass]
pub struct Layer {
    #[pyo3(get, set)]
    size: usize,
    weights: Vec<Vec<f32>>,
    #[pyo3(get)]
    weights_shape: Vec<usize>,
    biases: Vec<f32>,
    #[pyo3(get)]
    biases_shape: usize,
    #[pyo3(get, set)]
    activation_type: Activation,
}

fn dot_product<'a, T>(a: &'a Vec<T>, b: &'a Vec<T>) -> T
where
    T: std::ops::Mul<Output = T> + std::iter::Sum<T> + Copy
{
    a.iter().zip(b.iter()).map(|(x, y)| *x * *y).sum()
}

#[pymethods]
impl Layer {
    // should be wrapped in python so it only takes numpy arrays 
    // should also be wrapped to cache get_weights and get_biases answers
    #[new]
    #[pyo3(signature = (new_size, new_weights, new_biases, new_activation))]
    pub fn new( 
        new_size: usize, 
        new_weights: PyReadonlyArray2<f32>, 
        new_biases: PyReadonlyArray1<f32>, 
        new_activation: Activation
    ) -> PyResult<Self> {
        let weights_shape = new_weights.shape().to_vec();
        let biases_shape = new_biases.len();
        if biases_shape != weights_shape[0] || biases_shape != new_size {
            return Err(Error::new(ErrorKind::InvalidData, "Length of all vectors and size do not match").into());
        } else {
            return Ok(
                Layer {
                    size: new_size,
                    weights: converter::to_vec2(new_weights)?,
                    weights_shape: weights_shape,
                    biases: new_biases.to_vec()?,
                    biases_shape: biases_shape,
                    activation_type: new_activation,
                }
            );
        }
    }

    // https://users.rust-lang.org/t/rust-support-overloading-constructor/4345/11
    #[staticmethod]
    pub fn new_with_vec(
        new_size: usize, 
        new_weights: Vec<Vec<f32>>, 
        new_biases: Vec<f32>, 
        new_activation: Activation
    ) -> Result<Self> {
        let weights_shape = vec![new_weights.len(), new_weights[0].len()];
        let biases_shape = new_biases.len();
        if biases_shape != weights_shape[0] || biases_shape != new_size {
            return Err(Error::new(ErrorKind::InvalidData, "Length of all vectors and size do not match"));
        } else {
            return Ok(
                Layer {
                    size: new_size,
                    weights: new_weights,
                    weights_shape: weights_shape,
                    biases: new_biases,
                    biases_shape: biases_shape,
                    activation_type: new_activation,
                }
            );
        }
    }

    pub fn set_weights_vec(&mut self, arr: Vec<Vec<f32>>) -> Result<()> {
        if vec![arr.len(), arr[0].len()] != self.weights_shape {
            let err_msg = format!("Shape does not match original of ({}, {})", 
                                  self.weights_shape[0], self.weights_shape[1]);
            return Err(Error::new(ErrorKind::Other, err_msg))
        }

        self.weights = arr;

        Ok(())
    }

    #[pyo3(signature = (arr))]
    pub fn set_weights(&mut self, arr: PyReadonlyArray2<f32>) -> PyResult<()> {
        // maybe check if weights are between -1 and 1
        // use range to check maybe
        match converter::to_vec2(arr) {
            Ok(weight_vector) => {
                self.set_weights_vec(weight_vector)?;
            },
            Err(e) => return Err(e.into()),
        }


        Ok(())
    }

    #[pyo3(signature = ())]
    pub fn get_weights<'a>(&self, py: Python<'a >) -> &'a PyArray<f32, Dim<[usize; 2]>> {
        // makes sure lifetime exists as long as python is running and class is alive
        // self.weights.clone().into_pyarray(py).to_owned()
        PyArray::from_vec2(py, &self.weights.clone()).unwrap()
    }

    pub fn set_biases_vec(&mut self, arr: Vec<f32>) -> Result<()> {
        if arr.len() != self.biases_shape {
            let err_msg = format!("Shape does not match original of {}", self.biases_shape);
            return Err(Error::new(ErrorKind::Other, err_msg).into());
        }

        self.biases = arr;

        Ok(())
    }

    #[pyo3(signature = (arr))]
    pub fn set_biases(&mut self, arr: PyReadonlyArray1<f32>) -> PyResult<()> {
        // maybe check if biases are beween -1 and 1
        match arr.to_vec() {
            Ok(bias_vector) => {
                self.set_biases_vec(bias_vector)?;
            },
            Err(e) => return Err(e.into()),
        }

        Ok(())
    }

    #[pyo3(signature = ())]
    pub fn get_biases<'a>(&self, py: Python<'a >) -> &'a PyArray<f32, Dim<[usize; 1]>> {
        // makes sure lifetime exists as long as python is running and class is alive
        // self.weights.clone().into_pyarray(py).to_owned()
        PyArray::from_vec(py, self.biases.clone())
    }

    pub fn calculate_vec(&self, inputs: Vec<f32>) -> Result<Vec<f32>> {
        if inputs.len() != self.weights_shape[1] {
            let err_msg = format!("Input vector shape does not match weights input shape of {}", 
                                  self.weights_shape[1]);
            return Err(Error::new(ErrorKind::Other, err_msg).into());
        }

        let mut ans = vec![0.; self.weights_shape[0]];
        for (n, (bias, weight)) in self.biases.iter().zip(&self.weights).enumerate()  {
            ans[n] = dot_product(&inputs, weight) + bias;
        }

        ans = ans.iter().map(|x| self.activation_type.calculate(*x)).collect(); 
        return Ok(ans);
    }

    #[pyo3(signature = (inputs))]
    pub fn calculate<'a >(&self, py: Python<'a >, inputs: PyReadonlyArray1<f32>) -> PyResult<&'a PyArray1<f32>> {
        let inputs_vec = match inputs.to_vec() {
            Ok(new_inputs_vec) => new_inputs_vec,
            Err(e) => return Err(e.into()),
        };
        
        return Ok(PyArray1::from_vec(py, self.calculate_vec(inputs_vec)?));
    }
}

#[pyclass]
pub struct NeuralNetwork {
    #[pyo3(get)]
    size: usize,
    #[pyo3(get)]
    layers: Vec<Layer>,
}

#[pymethods]
impl NeuralNetwork {
    #[staticmethod]
    pub fn rust_new(new_size: usize, new_layers: Vec<Layer>) -> Result<Self> {
        if new_layers.len() != new_size {
            let err_msg = format!("Size of {} does not match layers length of {}",
                                  new_size, new_layers.len());
            return Err(Error::new(ErrorKind::Other, err_msg).into());
        }

        for i in 1..new_layers.len() {
            if new_layers[i].weights_shape[1] != new_layers[i-1].weights_shape[0] {
                let err_msg = format!("Layer at {} with output shape of {} does not match layer at {} of weight shape {}",
                                      i, new_layers[i].weights_shape[1], i-1, new_layers[i-1].weights_shape[0]);
                return Err(Error::new(ErrorKind::Other, err_msg).into());
            }
        }

        return Ok(
            NeuralNetwork { 
                size: new_size, 
                layers: new_layers, 
            }
        );
    }

    #[new]
    #[pyo3(signature = (new_size, new_layers))]
    pub fn new(new_size: usize, new_layers: Vec<Layer>) -> PyResult<Self> {
        Ok(NeuralNetwork::rust_new(new_size, new_layers)?)
    }

    pub fn rust_predict(&self, inputs: Vec<f32>) -> Result<Vec<f32>> {
        let mut ans = inputs;

        for i in &self.layers {
            ans = match i.calculate_vec(ans) {
                Ok(output) => output,
                Err(e) => return Err(e),
            };
        }

        return Ok(ans);
    }

    #[pyo3(signature = (inputs))]
    pub fn predict<'a >(&self, py: Python<'a >, inputs: PyReadonlyArray1<f32>) -> PyResult<&'a PyArray1<f32>> {
        let mut ans = match inputs.to_vec() {
            Ok(inputs_vec) => inputs_vec,
            Err(e) => return Err(e.into()),
        };

        // for i in &self.layers {
        //     ans = match i.calculate_vec(ans) {
        //         Ok(output) => output,
        //         Err(e) => return Err(Error::new(ErrorKind::Other, e.to_string()).into()),
        //     };
        // }

        ans = self.rust_predict(ans)?;

        return Ok(PyArray1::from_vec(py, ans));
    }

    // check if option is allowed in pyo3, option for none for weights or biases
    // use * to deliminate keyword only args in function signature
    pub fn rust_set_layer(
        &mut self, 
        layer_index: usize,
        new_weights: Option<Vec<Vec<f32>>>, 
        new_biases: Option<Vec<f32>>
    ) -> Result<()> {
        if layer_index >= self.size {
            let err_msg = format!("{} is larger than size of {}", layer_index, self.size);
            return Err(Error::new(ErrorKind::InvalidInput, err_msg));
        }

        // match new_weights {
        //     Some(weights_vec) => { self.layers[layer_index].set_weights_vec(weights_vec)?; },
        //     None => {},
        // }

        if let Some(new_weights) = new_weights {
            self.layers[layer_index].set_weights_vec(new_weights)?;
        }

        // match new_biases {
        //     Some(biases_vec) => { self.layers[layer_index].set_biases_vec(biases_vec)?; },
        //     None => {},
        // }

        if let Some(new_biases) = new_biases {
            self.layers[layer_index].set_biases_vec(new_biases)?;
        }

        Ok(())
    }

    #[pyo3(signature = (layer_index, *, new_weights=None, new_biases=None))]
    pub fn set_layer(
        &mut self, 
        layer_index: usize,
        new_weights: Option<PyReadonlyArray2<f32>>,
        new_biases: Option<PyReadonlyArray1<f32>>
    ) -> PyResult<()> {
        let mut weights_vec = None;

        if let Some(new_weights) = new_weights {
            match converter::to_vec2(new_weights) {
                Ok(new_weights_vec) => { weights_vec = Some(new_weights_vec); },
                Err(e) => { return Err(e.into()); },
            }
        }

        let mut biases_vec = None;

        if let Some(new_biases) = new_biases {
            match new_biases.to_vec() {
                Ok(new_biases_vec) => { biases_vec = Some(new_biases_vec); },
                Err(e) => { return Err(e.into()); },
            }
        }

        self.rust_set_layer(layer_index, weights_vec, biases_vec)?;

        Ok(())
    }

    pub fn rust_re_init_layer(&mut self, layer_index: usize, new_layer: Layer) -> Result<()>{
        if layer_index >= self.size {
            let err_msg = format!("{} is larger than size of {}", layer_index, self.size);
            return Err(Error::new(ErrorKind::InvalidInput, err_msg));
        }

        if new_layer.weights_shape != self.layers[layer_index].weights_shape {
            let err_msg = format!("({}, {}) does not match original size of ({}, {})", 
                                  new_layer.weights_shape[0],
                                  new_layer.weights_shape[1], 
                                  self.layers[layer_index].weights_shape[0],
                                  self.layers[layer_index].weights_shape[1]);
            return Err(Error::new(ErrorKind::InvalidInput, err_msg));
        }

        if new_layer.biases_shape != self.layers[layer_index].biases_shape {
            let err_msg = format!("{} does not match original size of {}", 
                                  new_layer.biases_shape, 
                                  self.layers[layer_index].biases_shape);
            return Err(Error::new(ErrorKind::InvalidInput, err_msg));
        }

        self.layers[layer_index] = new_layer;

        Ok(())
    }

    #[pyo3(signature = (layer_index, new_layer))]
    pub fn re_init_layer(&mut self, layer_index: usize, new_layer: Layer) -> PyResult<()> {
        self.rust_re_init_layer(layer_index, new_layer)?;
        
        Ok(())
    }

    pub fn rust_add_layer(&mut self, new_layer: Layer) -> Result<()> {
        // check if previous layer has correct sizing
        if self.size != 0 && new_layer.weights_shape[1] != self.layers[self.size-1].weights_shape[0] {
            let err_msg = format!("({}, {}) does not match previous size of ({}, {})", 
                                  new_layer.weights_shape[0],
                                  new_layer.weights_shape[1], 
                                  self.layers[self.size-1].weights_shape[0],
                                  self.layers[self.size-1].weights_shape[1]);
            return Err(Error::new(ErrorKind::InvalidInput, err_msg)); 
        }

        self.layers.push(new_layer);
        self.size += 1;

        Ok(())
    }

    #[pyo3(signature = (new_layer))]
    pub fn add_layer(&mut self, new_layer: Layer) -> PyResult<()> {
        self.rust_add_layer(new_layer)?;

        Ok(())
    }
}
