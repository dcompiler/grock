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
    pub(crate) heads: Vec<Head>,
    pub(crate) oweights: Tensor,
    masked: bool,
    pub(crate) qweights: Tensor,
    pub(crate) kweights: Tensor,
    pub(crate) vweights: Tensor,

}



struct Head {
    pub(crate) qproj: Tensor,
    pub(crate) kproj: Tensor,
    pub(crate) vproj: Tensor,
}

pub(crate) fn init_head()->Head{
    Head {
        qproj: utility::xavier_gen(MAX_LEN, decoder::D_MODEL).set_requires_grad(true),
        kproj: utility::xavier_gen(MAX_LEN, decoder::D_MODEL).set_requires_grad(true),
        vproj: utility::xavier_gen(MAX_LEN, decoder::D_MODEL).set_requires_grad(true),
    }
}

pub(crate) fn init_mha( masked: bool) ->MHA{
        assert_eq!(decoder::D_MODEL % decoder::N_HEADS, 0, "d_model must be divisible by n_heads");
       let mut mha = MHA{
            d_model: decoder::D_MODEL,
            d_k: (decoder::D_MODEL / decoder::N_HEADS), //Harvard uses div floor, but we assert the remainder is 0 so I don't think this is necessary
            heads: vec![],
            kweights : utility::xavier_gen(D_MODEL, D_MODEL/N_HEADS).set_requires_grad(true),
            qweights : utility::xavier_gen(D_MODEL, D_MODEL/N_HEADS).set_requires_grad(true),
            vweights : utility::xavier_gen(D_MODEL, D_MODEL/N_HEADS).set_requires_grad(true),
            oweights: utility::xavier_gen(MAX_LEN * N_HEADS, decoder::D_MODEL).set_requires_grad(true),
            masked
        };
    for _ in 0..N_HEADS{
        mha.heads.push(init_head());
    }
    mha

}

impl MHA{
    pub(crate) fn forward(&self, input: &Tensor, ) -> Tensor{
       let mut output : Tensor = Default::default();
        for head in &self.heads{
            output = tch::Tensor::concatenate(&[output, head.forward(input,&self.qweights, &self.kweights, &self.vweights, self.masked)], 0);
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
    fn forward(&self, x: &Tensor, qweights: &Tensor, kweights: &Tensor, vweights: &Tensor , masked: bool,) ->Tensor{

        let query = x.dot(&(&self.qproj * qweights));
        let key = x.dot(&(&self.kproj * kweights));
        let value = x.dot(&(&self.vproj * vweights));

       let mut mask = Tensor::zeros(&[MAX_LEN as i64, D_MODEL as i64], (Kind::Float, Device::Cpu));
        if masked  { //15% random masking
            let mut rng = rand::thread_rng();
            for i in 0..mask.size()[1]{
                let n: f64  = rng.gen();
                if n < 0.15 {
                    mask  = mask.narrow(1, i, 1).fill_(-65536); //-inf essentialy, the real -inf value causes errors
                }
            }

            mask = mask.unsqueeze(1);
        }

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