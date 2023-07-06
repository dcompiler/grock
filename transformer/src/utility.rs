//functions used by multiple modules


use tch::{Device, Kind, Tensor};
use tch::Kind::Float;

pub(crate) fn softmax(mut input: Tensor) ->Tensor{
    input = input.exp();
    let sum = input.sum(Float);
    input = input / sum;
    input
}

pub(crate) fn linear_transform(mut input: &Tensor, weights: Tensor, bias: Tensor) -> Tensor{
    input.dot(&weights) + &bias
}

pub(crate) fn dropout(p: f32, mut x:Tensor) ->Tensor{
    for i  in 0..x.size()[1]{
        if rand::random::<f32>() < p{
            x = x.narrow(1,i,1).fill_(0.0);
        }
    }
    x = x * Tensor::from(1.0 / (1.0 - p));
    return x;
}

/*
pub(crate) fn std_dev(x: Tensor, axis : usize) ->f32{
    let mut mean = 0.0;
    for i in 0..x.size()[axis]{
        mean += x.narrow(axis as i64, i, 1).sum(Kind::Float).int64_value(&[0]);
    }
    let mean = x.mean(Kind::Float)[axis].int64_value(&[0]);
   // let mean = x.mean_dim(Option::from(vec![&axis]), false, Float).int64_value(&[0]);
    let mut tempsum: f32 = 0.0;
    for i in x[axis].iter(){
        tempsum += (i - mean).pow(2);
    }
    return (tempsum / x.size()[axis] as f32).sqrt();
}

 */

pub(crate) fn xavier_gen(input_dim: usize, output_dim: usize) -> Tensor{
    let mut weights = Tensor::rand(vec![input_dim as i64, output_dim as i64], (Kind::Float, Device::Cpu));
    weights = weights * Tensor::from((6.0 / (input_dim +output_dim) as f32).sqrt());
    return weights;
    
}

pub(crate) fn kl_div_loss(input: &Tensor, target: Tensor) -> Tensor {
    let softmax_input = input.softmax(-1, Kind::Float);
    let log_softmax_input = softmax_input.log();
    let kl_div = softmax_input * (log_softmax_input - target);
    kl_div.sum(Kind::Float)
}