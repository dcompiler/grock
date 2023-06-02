//other stages such as embedd, positional encoding, add + norm stage, etc

const MAX_LEN: usize = 6;


use ndarray::{Array, Array1, Array2};
use crate::decoder;

fn embed(input: String) -> Vec<Array1000<f32>>{ //rewrite to struct init run structure
    let output: Vec<Array1000<f32>> = Vec::new();
    for token in String.split_whitespace(){
        match token {
            "0" => output.push_back(gen_array(0)),// create match fn, store values in dict. for better generalizability
            "1" => output.push_back(gen_array(1)),
            other => output.push_back(gen_array(2)),
        }
    }
    return output; //mult all by sqrt(d_model)
}

fn gen_array(element: i32) -> Array<f32, D>{
    let mut output: Array<f32, D> = Array::zeros(decoder::D_MODEL); // dimension, d_model
    output[element] = 1.0;
    return output;
}

pub struct PosEncoder {
    pe: Array2<f32>,
}

fn init_pe(){
 let  mut pe : Array2<f32> = Array2::zeros((MAX_LEN,decoder::D_MODEL));
    let  pos = Array1::range(0.,MAX_LEN as f32,1.);
    let  temp = Array1::range(0.,decoder::D_MODEL as f32,2.);
    let  divterm = temp.map(|x|x * -( 10000.0.log(1.exp() / decoder::D_MODEL as f32).exp()));

}

fn run_pe(){

}

struct LayerNorm{}

fn init_ln(){

}

fn run_ln(){

}



//fn add + norm stage

