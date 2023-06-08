//multihead attention, masked multihead attention, attention


use ndarray::Array2;
use crate::Utility;
use crate::decoder;

struct MHA{
    heads :usize,
    d_model: usize,
    dropout: f32,
    d_k: usize,
}

fn init_mha(dropout: f32) ->MHA{
        assert_eq!(decoder::D_MODEL % decoder::N_HEADS, 0, "d_model must be divisible by n_heads");
        MHA{
            heads: decoder::N_HEADS,
            d_model: decoder::D_MODEL,
            dropout,
            d_k: (decoder::D_MODEL / decoder::N_HEADS), //Harvard uses div floor, but we assert the remainder is 0 so I don't think this is necessary
        }

}

impl MHA{
    fn run_mha(query: Array2<f32>, key: Array2<f32>, value: Array2<f32>, mask: Option<Array2<f32>>) { //mask parameter
    if mask != None {
        //mask.unsqueeze(1) adds a dimension to the mask
        //need to understand how mask works in harvard code before implementing
    }
        nbatches = query.size(0);

        //project q,k,v

        //apply attention

        //concat, linearize


    }

}





fn attention(query: Array2<f32>, key:Array2<f32>, value:Array2<f32>, mask:Option<Array2<f32>>, dropout: bool )->Array2<f32>{
    let d_k = query.size(-1);
    let mut scores = query.dot(&key.t);
    if mask!= None {
        scores = scores.dot(&mask);
    }
    scores = scores / (d_k as f32).sqrt();
    scores = Utility::softmax(scores); //apply to each row
    if dropout {
        scores = Utiliy::dropout(0.1, scores);
    }
    scores.dot(&value)
}