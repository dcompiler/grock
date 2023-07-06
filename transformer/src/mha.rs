//multihead attention, masked multihead attention, attention


use crate::{ utility};
use crate::decoder;
use tch::{Device, Kind, Tensor};
use crate::decoder::{D_MODEL, N_HEADS};
use crate::otherstages::MAX_LEN;
use rand::prelude::*;
pub(crate) const DROPOUT : f32 = 0.1;


pub(crate) struct MHA{
    d_model: usize,
    d_k: usize,
    heads: Vec<Head>,
    oweights: Tensor,
    masked: bool,

}



struct Head {
    qweights: Tensor,
    kweights: Tensor,
    vweights: Tensor,
}

pub(crate) fn init_head()->Head{
    Head {
        qweights: utility::xavier_gen(MAX_LEN, decoder::D_MODEL),
        kweights: utility::xavier_gen(MAX_LEN, decoder::D_MODEL),
        vweights: utility::xavier_gen(MAX_LEN, decoder::D_MODEL),
    }
}

pub(crate) fn init_mha(dropout: f32, masked: bool) ->MHA{
        assert_eq!(decoder::D_MODEL % decoder::N_HEADS, 0, "d_model must be divisible by n_heads");
       let mut mha = MHA{
            d_model: decoder::D_MODEL,
            d_k: (decoder::D_MODEL / decoder::N_HEADS), //Harvard uses div floor, but we assert the remainder is 0 so I don't think this is necessary
            heads: vec![],
            oweights: utility::xavier_gen(MAX_LEN * N_HEADS, decoder::D_MODEL),
            masked
        };
    for _ in 0..N_HEADS{
        mha.heads.push(init_head());
    }
    mha

}

impl MHA{
    pub(crate) fn forward(&self, input: Tensor, ) -> Tensor{
       let mut output : Tensor = Default::default();
        for head in &self.heads{
            output = tch::Tensor::concatenate(&[output, head.forward(&input, self.masked)], 0);
        }

        /*
        //concat old code for reference
        x = x.transpose(1,2).contiguous().view(nbatch, -1, self.d_model*&self.heads);
        */

        //linearize
       output.dot(&self.oweights)

    }

}

impl Head{
    fn forward(&self, x: &Tensor, masked: bool) ->Tensor{

        let query = x.dot(&self.qweights);
        let key = x.dot(&self.kweights);
        let value = x.dot(&self.vweights);

       let mut mask = Tensor::zeros(&[MAX_LEN as i64, D_MODEL as i64], (Kind::Float, Device::Cpu));
        if masked  { //15% random masking
            let mut rng = rand::thread_rng();
            for i in 0..mask.size()[1]{
                let n: f64  = rng.gen();
                if n < 0.15 {
                    mask  = mask.narrow(1, i, 1).fill_(-65536);
                }
            }

            mask = mask.unsqueeze(1);
        }

        //nbatch = query.size();

        //project q,k,v
        /*
        let (query, key, value): (Array2<f32>, Array2<f32>, Array2<f32>) =
            [
                query, key, value
            ]
                .iter()
                .map(|( x)| {
                    let mut linear_result = Utility::linear_transform(&x, &weights, &bias);
                    let mut reshaped_result = Array2::zeros((nbatch, linear_result.shape()[1], self.heads, self.d_k));

                    for (mut dest, src) in reshaped_result.axis_iter_mut(ndarray::Axis(1)).zip(linear_result.axis_iter(ndarray::Axis(1))) {
                        dest.assign(&src);
                    }

                    reshaped_result.swap_axes(1, 2);
                    reshaped_result
                }).unzip();
        */


        //apply attention
         attention(query, key, value, mask)

    }
}






fn attention(query: Tensor, key:Tensor, value:Tensor, mask: Tensor) ->Tensor{
    let d_k = query.size()[1];
    let mut scores = query.dot(&key.transpose(0,1));
    scores = scores+(&mask); //add -inf to masked positions, 0 to others
    let scale = Tensor::from(1.0 / (d_k as f32).sqrt());
    scores = scores *scale;
    scores = utility::softmax(scores); //apply to each row
    if DROPOUT>0.0 {
        scores = utility::dropout(DROPOUT, scores);
    }
    scores.dot(&value)
}