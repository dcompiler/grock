//feed Forward neural network implementation

use tch::{Device, Kind, Tensor};
use crate::decoder::D_MODEL;
use crate::mha::DROPOUT;
use crate::otherstages::MAX_LEN;
use crate::utility;

const D_FF: usize = 512; //dimensionality of feed Forward network

struct Example{
    input: Tensor,
    output: Tensor,
}

//todo: fix many borrowing/lifetime issues with this network.. may have too much sharing & need to be reworked


struct Connection{
    src: &'static mut Neuron,
    dst: &'static mut Neuron,
    weight: Tensor,
    bias: Tensor,
}

enum NeuronType {
    Input,
    Activation,
    Hidden,
    Output,
}

struct Neuron{
    outgoing_connections: Vec<&'static mut Connection>,
    incoming_connections: Vec<&'static mut Connection>,
    neuron_type: NeuronType,
    output: &'static mut Tensor,
}

struct Layer {
    neurons: Vec<Neuron>
}

pub (crate) struct FNN{
     layers:[Option<Layer>; 5],
}

 fn init_fnn_helper() -> FNN{
     FNN{
         layers: [None, None, None, None, None], //since bert uses 2 layer fnn + input layer + output layer + activation layer in the middle
     }
}

fn init_connection(mut src: &'static mut Neuron, mut dst: &'static mut Neuron, weight: Tensor, bias: Tensor) -> Connection {
 Connection {
        src,
        dst,
        weight,
        bias
    }
}

fn init_neuron(neuron_type: NeuronType) -> Neuron{
    Neuron{
        outgoing_connections: Vec::new(),
        incoming_connections: Vec::new(),
        output:  & mut  Tensor::zeros(vec![MAX_LEN as i64, D_MODEL as i64], (Kind::Float, Device::Cpu)),
        neuron_type,
    }
}

pub (crate) fn init_fnn(ninputs: usize) -> FNN{ //assumes 2 hidden layers, 1 activation layer, 1 input layer, 1 output layer
    let mut input = Layer{ neurons: Vec::new()};
    for _ in 0..ninputs{
    input.neurons.push(init_neuron(NeuronType::Input));
    }
    let mut h1 = Layer{ neurons: Vec::new()};
    for _ in 0..D_FF{
        h1.neurons.push(init_neuron(NeuronType::Hidden));
    }
    let mut relul = Layer{ neurons: Vec::new()};
    for _ in 0..D_FF{
        relul.neurons.push(init_neuron(NeuronType::Activation));
    }
    let mut h2= Layer{ neurons: Vec::new()};
    for _ in 0..D_FF{
        h2.neurons.push(init_neuron(NeuronType::Hidden));
    }
    let mut output = Layer{ neurons: Vec::new()};
    output.neurons.push(init_neuron(NeuronType::Output));

    let mut fnn = init_fnn_helper(); //should be 2 for bert

    fnn.layers[0] = Option::from(input);
    fnn.layers[1] = Option::from(h1);
    fnn.layers[2] = Option::from(relul);
    fnn.layers[3] = Option::from(h2);
    fnn.layers[4] = Option::from(output);

    for l in 0..5 {
        for  i in fnn.layers[l].unwrap().neurons.iter_mut(){
            for  j in fnn.layers[l+1].unwrap().neurons.iter_mut(){
                let mut connection = init_connection( i, j, Tensor::zeros(vec![D_FF as i64, D_MODEL as i64], (Kind::Float, Device::Cpu)), Tensor::zeros(vec![D_FF as i64, D_MODEL as i64], (Kind::Float, Device::Cpu)));
                connection.src.outgoing_connections.push(&mut connection);
                connection.dst.incoming_connections.push(&mut connection);
            }
        }
    }

    fnn
}

impl FNN {

    pub(crate) fn forward(&self, input: Tensor) -> Tensor{
        //todo: driver for input, use train_fnn or have it train with the rest of the model?
        return Tensor::zeros(vec![D_FF as i64, D_MODEL as i64], (Kind::Float, Device::Cpu));
    }

    fn train_fnn(&mut self, mut examples: Vec<Example>, alpha: f32, epochs: i32){ //example is f32 since input vectors have been embedded
        self.initialize_weights_biases();
        for i in 0..epochs{ //until end of epochs, stopping criteria
            print!("epoch: {}", i);
            for j in examples.iter_mut(){
                self.propogate( j);
                //self.backprop(j, alpha);
            }
        }
    println!("training complete");
    }


    fn propogate(&mut self, mut example: &mut Example){
        for i in self.layers[0].unwrap().neurons.iter_mut(){
            i.output = &mut example.input;
        }
        for i in 1..3{
            for j in self.layers[i].unwrap().neurons.iter_mut(){
                  j.forward(&example.input);
            }
        }
        //todo: check propogate through network
    }

    /* IMPLEMENT BACKPROP AS PART OF TRAINING/OPTIMIZING
    fn backprop(& self, mut example: &Example, alpha: f32){
       let output_error = self.error_prime( self.layers[4].unwrap(), &example.output);
        for l in 4..1.iter(){ //iterate through layers
            for i in self.layers[l].unwrap().neurons.iter(){ //iterate through neurons in layer
                for j in i.incoming_connections.iter(){ //iterate through incoming connections in each layer
                    match i.neuron_type{
                        NeuronType::Input => (),
                        NeuronType::Activation => (),
                        NeuronType::Hidden => {
                            j.weight = j.weight - Tensor::from(alpha) * j.src.output * output_error;
                            j.bias = j.bias - Tensor::from(alpha) * output_error;
                        }
                        NeuronType::Output => {
                            j.weight = j.weight - Tensor::from(alpha) * j.src.output * output_error;
                            j.bias = j.bias - Tensor::from(alpha) * output_error;
                        }
                    }
                }
            }
        }
    }

     */

    fn initialize_weights_biases(&self){
        for layer in self.layers.iter(){
            if self.layers.iter().next().is_none() {
                break;
            }
            for neuron in layer.unwrap().neurons.iter(){
                for connections in neuron.outgoing_connections.iter(){
                    connections.weight = (Tensor::rand(vec![MAX_LEN as i64, D_MODEL as i64], (Kind::Float, Device::Cpu)) * Tensor::from(2.0-1.0 )) *Tensor::from((6.0 / (D_FF+D_MODEL) as f32).sqrt()) ;//xavier initialization
                    connections.bias = Tensor::zeros(vec![D_FF as i64, D_MODEL as i64], (Kind::Float, Device::Cpu));
                }
            }
        }
    }

    fn run_fnn(input: Tensor, weights: Tensor, biases: Tensor ){
        let mut l1 = input.dot(&weights)+ biases;
        l1 = l1.relu();
        l1 = utility::dropout(DROPOUT, l1);
        //todo
        //apply d_model -> d_ff linear transformation
        //apply relu
        //apply dropout
        //apply d_ff -> d_model linear transformation
        //todo: residual
    }

    fn calculate_error(mut layer: Layer,   exoutput: &Tensor) -> Tensor{ //loss function
        let mut error = layer.neurons[0].output.subtract(exoutput);
         error.pow(&Tensor::from(2.0))
    }

    fn error_prime(layer: Layer, exoutput: Tensor) -> Tensor{ //derivative of error
        let mut error = layer.neurons[0].output.subtract(&exoutput);
         error/exoutput.size()[0]
    }
}

impl Neuron{
    fn forward(&mut self, input: &Tensor){
        match self.neuron_type{
            NeuronType::Input => (),
            NeuronType::Activation => {
                self.output = &mut self.output.relu();
            }
            NeuronType::Hidden => {
                self.output = &mut utility::linear_transform(input, Default::default(), Default::default()) //todo: implement linear transform weights
            }
            NeuronType::Output => (),
        }
    }
}



fn relu_prime(x:f32 )-> f32 { if x>0.0 {1.0} else {0.0} }


