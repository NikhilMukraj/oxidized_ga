use numpy::{PyArray2, PyReadonlyArray2};
use std::collections::HashMap;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::PyResult;
use std::io::{Error, ErrorKind};


#[allow(dead_code)]
fn vec_to_2d_vec<T>(data: Vec<T>, shape: Vec<usize>) -> Vec<Vec<T>> where T: Clone {
    let inner_dim = shape.last().copied().unwrap_or(0);
    data.chunks(inner_dim).map(|chunk| chunk.to_vec()).collect()
}

#[allow(dead_code)]
pub fn to_vec2(arr: PyReadonlyArray2<f32>) -> Result<Vec<Vec<f32>>, Error> {
    let shape = arr.shape().to_vec();
    let downcasted = match arr.downcast::<PyArray2<f32>>() {
        Ok(downcasted_vec) => downcasted_vec,
        Err(_e) => return Err(Error::new(ErrorKind::Other, "Cannot downcast into Vec<f32>")),
    };
    let contig_vec = match downcasted.to_vec() {
        Ok(new_vec) => new_vec,
        Err(_e) => return Err(Error::new(ErrorKind::Other, "Cannot downcast into Vec<f32>")),
    };

    return Ok(vec_to_2d_vec(contig_vec, shape));
}

#[allow(dead_code)]
pub fn convert_pydict_to_hashmap<'a, K, V>(pydict: &'a PyDict) -> PyResult<HashMap<K, V>>
where
    K: FromPyObject<'a> + std::hash::Hash + Eq,
    V: FromPyObject<'a>,
{
    let mut hashmap: HashMap<K, V> = HashMap::new();

    for pair in pydict.items().iter() {
        let key_value = pair.get_item(0).unwrap().extract::<K>()?;
        let value_value = pair.get_item(1).unwrap().extract::<V>()?;
        hashmap.insert(key_value, value_value);
    }

    return Ok(hashmap);
}

#[allow(dead_code)]
pub fn convert_hashmap_to_pydict<'a, K, V>(py: Python<'_>, hashmap: &HashMap<K, V>) -> PyResult<Py<PyDict>>
where
    K: ToPyObject + Eq + std::hash::Hash,
    V: ToPyObject,
{
    let pydict = PyDict::new(py);

    for (key, value) in hashmap {
        pydict.set_item(key, value)?;
    }

    return Ok(pydict.into())
}