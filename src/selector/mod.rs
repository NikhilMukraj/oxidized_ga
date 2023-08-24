use rand::Rng;


#[allow(dead_code)]
pub fn selection<T>(pop: &Vec<T>, scores: &Vec<f32>, k: usize) -> T 
where
    T: Clone
{
    // default should be 3
    let mut rng_thread = rand::thread_rng(); 
    let mut selection_index = rng_thread.gen_range(1..pop.len());

    let indices = (0..k-1)
        .into_iter()
        .map(|_x| rng_thread.gen_range(1..pop.len()));

    // performs tournament selection to select parents
    for i in indices {
        if scores[i] < scores[selection_index] {
            selection_index = i;
        }
    }

    return pop[selection_index].clone();
}
