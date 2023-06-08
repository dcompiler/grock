// for structuring decoder, function calls from other files, batching
use std::collections::VecDeque;
use std::os::raw::c_float;
use ndarray::{Array, Array6, Array1};


use crate::mha;

 pub(crate) const N_PARAMS: usize = 100; //number of learnable weights/parameters
 const N_LAYERS: usize = 2; //number of layers in decoder
pub(crate) const N_HEADS: usize = 4; //number of heads in multihead attention
pub(crate) const D_MODEL: usize = 128; //  dimensionality of hidden state, size of vector used to represent each token in input sequence
 const D_HEAD: usize = 32; //dimensionality of each attention head, size of sub vector used within atten. head
 const BATCH_SIZE: usize = 1;
 const LEARNING_RATE: f32 = 0.0001;

struct Decoder{}

fn init_decoder(){}

fn run_decoder(){
 /*
read input
embedd input
encode input (positional encoding)

repeat nx: (decoder layer)
    masked mha
    add + norm
    mha
    add+norm
    fnn
    add+norm

linearize
softmax

 */
}


struct DecoderLayer{}

fn init_decoder_layer(){} //initialize (number heads, layers, widths, vocab size/length, etc.)

fn run_decoder_layer(){}

struct SublayerConnection{}

fn init_sublayer_connection(){}

fn run_sublayer_connection(){}

//connect layers/something to handle inputs and outputs




struct Batch{}

fn init_batch(){}

fn run_batch(){}

fn subsequent_mask(){}

fn make_std_mask(){}




