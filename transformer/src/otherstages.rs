//other stages such as embedd, positional encoding, add + norm stage, etc

const MAX_LEN: usize = 6;


use ndarray::{Array, Array1, Array2, Axis, Dimension, s};
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
    return output; //IMPORTANT mult all by sqrt(d_model), .lut(x)??
}

fn gen_array(element: i32) -> Array<f32, D>{
    let mut output: Array<f32, D> = Array::zeros(decoder::D_MODEL); // dimension, d_model
    output[element] = 1.0;
    return output;
}

pub struct PosEncoder {
    pe:  [Array2<f32>; 1],
}

fn init_pe()-> PosEncoder{
    PosEncoder{ pe: [Array2::zeros((MAX_LEN,decoder::D_MODEL))]};
    let  mut pe : Array2<f32> = Array2::zeros((MAX_LEN,decoder::D_MODEL));
    let  pos = Array1::range(0.,MAX_LEN as f32,1.);
    let  temp = Array1::range(0.,decoder::D_MODEL as f32,2.);
    let  divterm = temp.map(|x|x * -( 10000.0.log(1.exp() / decoder::D_MODEL as f32).exp()));
    for posterm in pe.iter_mut(){
        for i in pe.iter_mut(){
            if i % 2 == 0 {
                *i = (pos[posterm] * divterm[i]).sin();
            }
            else{
                *i = (pos[posterm] * divterm[i]).cos();
            }
        }
    }
    PosEncoder.pe = [pe]; //unsqueeze(0)
    PosEncoder
    //Harvard calls register_buffer, which makes [pe] accessible from anywhere in the model
    //This is part of the nn.module library in pyTorch
    //Will implement something similar as calls to pe are needed
}


impl PosEncoder{
    fn run_pe(mut x: Array2<f32>) -> Array2<f32>{
         x = x + this.pe.slice(s![..,.x.size(1),..]);
        return Utility::dropout(x);
    }
}

struct LayerNorm{
    a_2 : Array2<f32>,
    b_2 : Array2<f32>,
    eps : f32,

}

fn init_ln(features: Box<dyn Dimension>) -> LayerNorm { //features is layers.size in Harvard. using box dyn bec
    features2 = features.clone();
    LayerNorm{
        a_2: Array2::ones(features),
        b_2: Array2::zeros(features2),
        eps: 0.000001,
    }
}

impl LayerNorm{
    fn run_ln(mut x: Array2<f32>){
        mean = x.mean_axis(Axis(-1));
        std = Utility::std_axis(x, Axis(-1));
        return this.a_2 * (x.map(|x| x-mean)) / (std + this.eps) + this.b_2;
    }
}




//fn add + norm stage

