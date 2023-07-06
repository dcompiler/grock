

struct Batch<T> {
    src: Vec<T>,
    src_mask: Vec<Vec<bool>>,
    trg: Option<Vec<T>>,
    trg_y: Option<Vec<T>>,
    trg_mask: Option<Vec<Vec<bool>>>,
    ntokens: usize,
}

impl<I32: PartialEq + Clone + Copy> Batch<I32> {
    fn init_batch(src: Vec<I32>, trg: Option<Vec<I32>>, pad: I32) -> Self {
        let src_mask = src.iter().map(|&val| val!=pad).collect::<Vec<bool>>(); //mask is true if pad false if not
        let src_mask = vec![src_mask]; //unsqueeze

        let (trg, trg_y, trg_mask, ntokens) = if let Some(trg) = trg {
            let trg_y = trg[..trg.len() - 1].to_vec(); //todo: what is trg_y?
            let trg_mask = Self::make_std_mask(&trg, pad);
            let ntokens = trg_y.iter().filter(|&val| *val != pad).count(); //count number of non pad tokens

            (Some(trg), Some(trg_y), Some(trg_mask), ntokens)
        } else {
            (None, None, None, 0)
        };

        Self {
            src,
            src_mask,
            trg,
            trg_y,
            trg_mask,
            ntokens,
        }
    }

    fn make_std_mask(tgt: &[I32], pad: I32) -> Vec<Vec<bool>> {
        let tgt_mask = tgt.iter().map(|&val| val != pad).collect::<Vec<bool>>();
        let tgt_mask = vec![tgt_mask];

        // todo: subsequent_mask implementation is required here

        tgt_mask
    }
}

fn run_batch(){}

fn ca_mask(sequence_length: usize) -> Vec<Vec<f32>> { //casual attention mask
    let mut mask = vec![vec![0.0; sequence_length]; sequence_length];

    for i in 0..sequence_length {
        for j in (i + 1)..sequence_length {
            mask[i][j] = f32::NEG_INFINITY;
        }
    }

    mask
}

fn make_std_mask(){}


use std::time::{Instant};
use crate::decoder::*;

fn run_epoch(data_iter: &mut dyn Iterator<Item =  Batch<f32>>, //iterate through data
                       decoder: &mut Decoder,) -> f32

{
    let mut start = Instant::now();
    let mut total_tokens = 0;
    let mut total_loss = 0.0;
    let mut tokens = 0;

    for (i, batch) in data_iter.enumerate() {
        let out = decoder.forward(batch.src, batch.trg.as_ref().unwrap(),
                                &batch.src_mask, batch.trg_mask.as_ref());
        let loss = 0.0;
        //todo: let loss = decoder::compute_loss(out, &batch.trg_y, batch.ntokens);
        total_loss += loss;
        total_tokens += batch.ntokens;
        tokens += batch.ntokens;
        if i % 50 == 1 {
            let elapsed = start.elapsed().as_secs_f32();
            println!("Epoch Step: {} Loss: {} tokens per Sec: {}",
                     i, loss / batch.ntokens as f32, tokens as f32 / elapsed);
            start = Instant::now();
            tokens = 0;
        }
        //optimzer::backprop
    }
    total_loss / total_tokens as f32
}