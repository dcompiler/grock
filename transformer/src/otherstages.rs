//other stages such as embedd, positional encoding, add + norm stage, etc

pub (crate) const MAX_LEN: usize = 6;


use tch::{Device, IndexOp, Kind};
use crate::{decoder, utility};
use crate::decoder::D_MODEL;
use tch:: Tensor;
use tch::Kind::Float;
use crate::mha::DROPOUT;


pub(crate) struct Embedder {
}


impl Embedder{
    pub(crate) fn forward(&mut self, input: std::string::String) -> Tensor{
        let mut output= Tensor::zeros(vec![MAX_LEN as i64, decoder::D_MODEL as i64], (Kind::Float, Device::Cpu));
        for token in input.split_whitespace(){
            match token {
                /*
                "0" => output.push(gen_array(0)),// todo: create match fn, store values in dict. for better generalizability. this is dependent on data set
                "1" => output.push(gen_array(1)),
                _ => output.push(gen_array(2)),

                 */
                &_ => {}
            }
        }
         unsafe { return output * Tensor::from((D_MODEL as f32).sqrt()); }
    }

}

fn gen_array(element: i64) -> Tensor{
    let mut output = Tensor::zeros(vec![D_MODEL as i64], (Float, Device::Cpu));
    output = output.narrow(0, element, 1).fill_(1.0);
    return output;
}

pub struct PosEncoder {
    pe:  Tensor ,
}

pub(crate) fn init_pe() -> PosEncoder{
    let mut posencoder = PosEncoder{pe : Tensor::zeros(vec![MAX_LEN as i64, decoder::D_MODEL as i64], (Kind::Float, Device::Cpu)) };

    let  mut pe : Tensor = Tensor::zeros(vec![MAX_LEN as i64, decoder::D_MODEL as i64], (Kind::Float, Device::Cpu));

    let  mut pos = Tensor::f_arange_start_step(0.,MAX_LEN as f64,1.0, (Kind::Float, Device::Cpu)).unwrap().unsqueeze(1);
    let  temp = Tensor::f_arange_start_step(0.,decoder::D_MODEL as f64,2.0, (Kind::Float, Device::Cpu)).unwrap();
    let  divterm = (temp *  Tensor::from(-10000_f32.log(10.0) / D_MODEL as f32)).exp();
    for posterm in 0..pe.size()[1]{
        if posterm % 2 == 0 {
            pe.narrow(1, posterm, pe.size()[1]).copy_(&(&pos * &divterm).sin());
        } else {
            pe.narrow(1, posterm, pe.size()[1]).copy_(&(&pos * &divterm).cos());
        }
    }

    posencoder.pe = pe.unsqueeze(0);
    posencoder
    //Harvard calls register_buffer, which makes [pe] accessible from anywhere in the model
    //This is part of the nn.module library in pyTorch
    //Will implement something similar as calls to pe are needed
}


impl PosEncoder{
    pub(crate) fn forward(&self, mut x: Tensor) -> Tensor{
        let len = x.size()[1];
         x = x + self.pe.i((.., ..1));
        return utility::dropout(DROPOUT, x);
    }
}

pub(crate) struct LayerNorm{
    a_2 : Tensor,
    b_2 : Tensor,
    eps : f32,

}

pub(crate) fn init_ln(features: usize) -> LayerNorm { //features is layers.size in Harvard. using box dyn bec //todo:features
    LayerNorm{
        a_2: Tensor::ones(vec![features as i64], (Kind::Float, Device::Cpu)),
        b_2: Tensor::zeros(vec![features as i64], (Kind::Float, Device::Cpu)),
        eps: 0.000001,
    }
}


impl LayerNorm{
    pub(crate) fn forward(&mut self, x: Tensor) -> Tensor {
        let mean = x.mean_dim(-1, true, Float);
        let std = x.std_dim(-1, true, true);
        return &self.a_2 * (x - Tensor::from(mean)) * Tensor::from(1 / (std +  Tensor::from(self.eps))) + &self.b_2;
    }
    }

pub (crate) struct SoftMax{}
impl SoftMax{
    pub(crate) fn forward (&self, x: Tensor) -> Tensor{
        return utility::softmax(x);
    }
}

