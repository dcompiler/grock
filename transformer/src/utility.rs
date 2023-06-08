//functions used by multiple modules

use ndarray::{Array2, Axis};

fn softmax(mut input: Array2<f32>){
    input = input.map(|x| x.exp());
    let sum = input.sum();
    input = input.map(|x| x / sum);
}

fn linearize(){}

fn dropout(p: float, mut x:Array2<f32>) ->Array2<f32>{
    for i in x.iter_mut(){
        if rand::random() < p{
            *i = 0.0;
        }
        *i = i / (1.0 - p);
    }
    return x;
}

fn std_dev(x: Array2<f32>, axis : usize)->f32{
    mean = x.mean_axis(Axis(axis));
    let mut tempsum = 0;
    for i in x.axis_iter(Axis(axis)){
        tempsum += (i - mean).pow(2);
    }
    return (tempsum / x.size(axis)).sqrt();
}