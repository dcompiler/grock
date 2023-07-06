// for structuring decoder, function calls from other files, batching
use tch::Tensor;


use crate::{mha, otherstages};
use crate::fnn::{FNN, init_fnn};
use crate::otherstages::{Embedder, MAX_LEN,  SoftMax};

pub(crate) const N_PARAMS: usize = 100; //number of learnable weights/parameters
 const N_LAYERS: usize = 2; //number of layers in decoder
pub(crate) const N_HEADS: usize = 4; //number of heads in multihead attention
pub(crate) const D_MODEL: usize = 128; //  dimensionality of hidden state, size of vector used to represent each token in input sequence
 const D_HEAD: usize = 32; //dimensionality of each attention head, size of sub vector used within atten. head
 const BATCH_SIZE: usize = 1;
 const LEARNING_RATE: f32 = 0.0001;

pub (crate) trait Forward {
    fn forward(&self, input: Tensor)->Tensor;
}

pub (crate) struct Decoder{
        inputlayer: InputLayer,
      internallayers: Vec<DecoderLayer>,
    outputlayer: OutputLayer,
    params: Vec<Parameter>,
}

struct Parameter{
    t: usize,
    fmoment: Tensor,
    smoment: Tensor,
    weight: Box<Tensor>,
    next: Option<Box<Vec<Parameter>>>,
    prev: Option<Box<Vec<Parameter>>>,
}

struct InputLayer{
    embedd: Embedder,
    pos_encoder: otherstages::PosEncoder,
}

struct OutputLayer{
    softmax: SoftMax,
    layernorm: otherstages::LayerNorm,
}

struct DecoderLayer{
    mmha: mha::MHA,
    mha: mha::MHA,
    fnn: FNN,
    norm1: otherstages::LayerNorm,
    norm2: otherstages::LayerNorm,
    norm3: otherstages::LayerNorm,
}


fn init_decoder() -> Decoder{
   let mut decoder = Decoder{
       inputlayer: InputLayer {
          embedd: Embedder{},
          pos_encoder: otherstages::init_pe() },
      internallayers: vec![],
      outputlayer: OutputLayer {
          layernorm: otherstages::init_ln(D_MODEL),
          softmax: SoftMax{} },
       params: vec![],
  };
    for _ in 0..N_LAYERS{
        decoder.internallayers.push(init_decoder_layer());
    }

     decoder
}



fn init_decoder_layer() -> DecoderLayer {
    DecoderLayer{
        mmha: mha::init_mha(0.1, true),
        mha: mha::init_mha(0.1, false),
        fnn: init_fnn(MAX_LEN ), // confirm max_len is correct
        norm1: otherstages::init_ln(D_MODEL), //confirm d_model is correct
        norm2: otherstages::init_ln(D_MODEL),
        norm3: otherstages::init_ln(D_MODEL),
    }
}


fn init_params(decoder: Decoder) -> Vec<Parameter>{
    let params = Vec::new();
    params
    //todo ------------------------------------------------
    //first embed
    //connect embed to mmha
    //do decoder layer
    //if is another layer connect to next layer
    //else connect to final output param
    //todo: how to connect single output to many in heads? and how to pass through the concat stage?
}

impl InputLayer{
    fn forward(&mut self, mut input: String) -> Tensor{
        let mut output = self.embedd.forward(input);
        output = self.pos_encoder.forward(output);
        output
    }
}


impl DecoderLayer{
    fn forward(&mut self, mut input: Tensor)-> Tensor{
        input = self.mmha.forward(input);
        input = self.norm1.forward(input);
        input = self.mha.forward(input);
        input = self.norm2.forward(input);
        input = self.fnn.forward(input);
        input = self.norm3.forward(input);
        input
    }
}

impl OutputLayer{
    fn forward(&mut self, mut input: Tensor) -> Tensor{
        input = self.layernorm.forward(input);
        input = self.softmax.forward(input);
        input
    }
}

impl Decoder{
    pub(crate) fn forward(&self, src: Vec<f32>, src_mask: &Vec<f32>, trg: &Vec<Vec<bool>>, trg_mask: Option<&Vec<Vec<bool>>>,){
        //todo ------------------------------------------------
    }
}
fn run_decoder(mut decoder: Decoder, mut y: String) -> Tensor{

    //batch, batchtrg, batchmask, batchtrgmask, all training things

    let mut x = decoder.inputlayer.forward(y);
    for i in 0..N_LAYERS{
        x = decoder.internallayers[i].forward(x);
    }
    x = decoder.outputlayer.forward(x);

    x
}
















//currently unused


/*


struct SublayerConnection{
    prev: Option<T>,
    next: Option<T>,
    prev_output: Vec<Option<T>>,
    next_input: Vec<Option<T>>,
}

fn init_sublayer_connection(prev: T, next: T)-> SublayerConnection{
    SublayerConnection{
        prev,
        next,
        prev_output: Vec::prev.Forward(),
        next_input: Vec::new(),
    }
}
*/


/* ex of sublayer connection usage
    embed2posenc = SublayerConnection{
        prev: Decoder::embedd,
        next: Decoder::pos_encoder,
        prev_output: Vec::new(),
        next_input: Vec::new(),
    };

    // more connections, consider if this is the best way to link objects or if keeping i,k,v, etc. in the same object is better

 */



fn run_sublayer_connection(){}

//connect layers/something to handle inputs and outputs










