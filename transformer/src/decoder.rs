use std::io::empty;
use std::ops::Deref;
// for structuring decoder, function calls from other files, parameters
use tch::{Device, Kind, Tensor};
use crate::{mha, otherstages, optimizer, fnn};
use crate::fnn::{ NeuralNetwork};
use crate::optimizer::NoamOpt;
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

pub (crate) struct Decoder<'a>{
        inputlayer: InputLayer,
      internallayers: Vec<DecoderLayer>,
    outputlayer: OutputLayer,
    params: Vec<&'a Parameter<'a>>,
}

pub(crate) struct Parameter<'a> {
    t: usize,
    rate: f64,
    fmoment: Tensor,
    smoment: Tensor,
    pub(crate) weight: &'a Tensor,
    next: Vec<&'a Parameter<'a>>,
    prev: Vec<&'a Parameter<'a>>,
}

struct InputLayer{
    embedder: Embedder,
    pos_encoder: otherstages::PosEncoder,
}

struct OutputLayer{
    softmax: SoftMax,
    layernorm: otherstages::LayerNorm,
}

struct DecoderLayer{
    mmha: mha::MHA,
    mha: mha::MHA,
    fnn: NeuralNetwork,
    norm1: otherstages::LayerNorm,
    norm2: otherstages::LayerNorm,
    norm3: otherstages::LayerNorm,
}


fn init_decoder<'a>() -> Decoder<'a>{
   let mut decoder = Decoder{
       inputlayer: InputLayer {
          embedder: otherstages::init_embedder(),
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
        mmha: mha::init_mha( true),
        mha: mha::init_mha(false),
        fnn: fnn::init_fnn(),
        norm1: otherstages::init_ln(D_MODEL), //confirm d_model is correct
        norm2: otherstages::init_ln(D_MODEL),
        norm3: otherstages::init_ln(D_MODEL),
    }
}


fn init_params(decoder: Decoder) -> Vec<Parameter>{
    let mut params = Vec::new();
    let mut ewp = init_param(&decoder.inputlayer.embedder.weight);
    params.push(ewp);
    let mut lcount =1;
    for layer in decoder.internallayers{
            let mut fop = init_param(&layer.mmha.oweights); //concat matrix
            let mut fq = init_param(&layer.mmha.qweights);
            let mut fk = init_param(&layer.mmha.kweights);
            let mut fv = init_param(&layer.mmha.vweights);
            params.push(fop);
            params.push(fq);
            params.push(fk);
            params.push(fv);
            for mmhahead in layer.mmha.heads {
                let mut fqp = init_param(&mmhahead.qproj); //todo: check lifetime won't expire for these dawgs
                let mut fkp = init_param(&mmhahead.kproj);
                let mut fvp = init_param(&mmhahead.vproj);
                params.push(fqp);
                params.push(fkp);
                params.push(fvp);
            }
        let mut sop = init_param(&layer.mha.oweights); //concat matrix
        let mut sq = init_param(&layer.mha.qweights);
        let mut sk = init_param(&layer.mha.kweights);
        let mut sv = init_param(&layer.mha.vweights);
        params.push(sop);
        params.push(sq);
        params.push(sk);
        params.push(sv);
        for mhahead in layer.mha.heads {
            let mut sqp = init_param(&mhahead.qproj);
            let mut skp = init_param(&mhahead.kproj);
            let mut svp = init_param(&mhahead.vproj);
            params.push(sqp);
            params.push(skp);
            params.push(svp);
        }
        let mut w1p = init_param(&layer.fnn.w1);
        let mut w2p = init_param(&layer.fnn.w2);
        let mut b1p = init_param(&layer.fnn.b1);
        let mut b2p = init_param(&layer.fnn.b2);
        params.push(w1p);
        params.push(w2p);
        params.push(b1p);
        params.push(b2p);

    }


    params
    //todo ------------------------------------------------
    //implement output param for loss calculation
    //fix return error on params
}

fn init_param<'a>(weight: &'a tch::Tensor) ->Parameter<'a>{
     Parameter{
         t:0,
         rate: 0.0,
         fmoment: Tensor::zeros(vec![MAX_LEN as i64, D_MODEL as i64], (Kind::Float, Device::Cpu) ), //todo: how to initialize first/second moments
         smoment: Tensor::zeros(vec![MAX_LEN as i64, D_MODEL as i64], (Kind::Float, Device::Cpu) ),
         weight,
         next: Vec::new(),
         prev: Vec::new()
     }
}

impl<'a> Parameter<'a >{

    pub(crate) fn update(&mut self, rate: &f64, opt: &mut NoamOpt){
        self.rate = *rate;
        let mut gradient = self.weight.grad();
        self.fmoment = &self.fmoment * Tensor::from(opt.beta1) + Tensor::from(1.0-opt.beta1) * &gradient;
        self.fmoment = &self.fmoment / Tensor::from(1.0-opt.beta1.powf(self.t as f64)); //bias correction
        self.smoment = &self.smoment * Tensor::from(opt.beta2) + Tensor::from(1.0-opt.beta2) * (&gradient.pow(&Tensor::from(2)));
        self.smoment = &self.smoment / Tensor::from(1.0-opt.beta2.powf(self.t as f64)); //bias correction
        self.weight = &((self.weight) - Tensor::from(self.rate) * &self.fmoment / (self.smoment.sqrt() + opt.eps)); //todo: fix
        self.t = &self.t + 1;
    }
}

impl InputLayer{
    fn forward(&mut self, mut input: String) -> Tensor{
        let mut output = self.embedder.forward(input);
        output = self.pos_encoder.forward(output);
        output
    }
}


impl DecoderLayer{
    fn forward(&mut self, mut input: Tensor)-> Tensor{
        input = &input + self.mmha.forward(&input);
        input = self.norm1.forward(input);
        input = &input + self.mha.forward(&input);
        input = self.norm2.forward(input);
        input = &input + self.fnn.forward(&input);
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

impl<'a> Decoder<'a>{
    pub(crate) fn forward(&self, src: Vec<f32>, src_mask: &Vec<f32>, trg: &Vec<Vec<bool>>, trg_mask: Option<&Vec<Vec<bool>>>,){
        //todo ------------------------------------------------
        //delete this stub?


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



//fn run_sublayer_connection(){}

//connect layers/something to handle inputs and outputs


//resnet gradient descent?? what is resnet? how to do residuals in transformer









