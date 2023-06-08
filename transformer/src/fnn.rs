//feed forward neural network implementation

use ndarray::{Array, Array2};

const D_FF: usize = 512; //dimensionality of feed forward network

struct Connection{
    src: Neuron,
    dst: Neuron,
    weight: f32,
}

enum NeuronType {
    Input,
    Activation,
    Hidden,
    Output,
}

struct Neuron{
    outgoing_connections: Vec<Connection>,
    incoming_connections: Vec<Connection>,
    neuron_type: NeuronType,
}

struct Layer {
    neurons: Vec<Neuron>
}

struct FNN{
     layers:Array<Layer, D>,
}

 fn init_fnn_helper() -> FNN{
     FNN{
         layers: Array::zeros((5)), //since bert uses 2 layer fnn + input layer + output layer + activation layer in the middle
     }
}

fn init_Connection(mut src: Neuron, mut dst: Neuron, weight: f32) -> &mut Connection{
    Connection{
        src,
        dst,
        weight,
    };
    src.outgoing_connections.push(Connection).expect("src outgoing connections push failed");
    dst.incoming_connections.push(Connection).expect("dst incoming connections push failed");
    Connection
}

fn init_neuron(neuron_type: NeuronType) -> Neuron{
    Neuron{
        outgoing_connections: Vec::new(),
        incoming_connections: Vec::new(),
        neuron_type,
    }
}

fn init_fnn(ninputs: f32) -> FNN{ //assumes 2 hidden layers, 1 activation layer, 1 input layer, 1 output layer
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


   //input layer
    *fnn.layers[0] : input;
    *fnn.layers[1] : h1;
    *fnn.layers[2] : relul;
    *fnn.layers[3] : h2;
    *fnn.layers[4] : output;


    for l in 0..5{
        for i in *ffn.layers[l].iter(){
            for j in *ffn.layers[l+1].iter(){
                init_Connection(i, j, 0.0);
            }
        }
    }


    fnn
}

impl fnn{
    fn train_fnn(examples: Vec<f32>, alpha: f32, epochs: i32){ //example is f32 since input vectors have been embedded
        initialize_weights();
        for i in 0..epochs{
            print!("epoch: {}", i);
            for j in examples.iter(){
                train(j, alpha)
            }
        }
    println!("training complete");
    }

    fn train(example: f32, alpha: f32){
        propogate(example);
        backprop(example, alpha);
        //update weights in these fcts
    }

    fn propogate(example: f32){
        //propogate through network
    }

    fn backprop(example: f32, alpha: f32){
        //backpropogate through network
    }

    fn initialize_weights(){
        //initialize weights
        //initialize biases
    }

    fn run_fnn(input: Array2<f32>, weights: Array2<f32>, biases: Array2<f32> ){
        let mut l1 = input.dot(&weights)+ biases;
        l1 = relu(l1);
        l1 = Utility::dropout(l1);

        //apply d_model -> d_ff linear transformation
        //apply relu
        //apply dropout
        //apply d_ff -> d_model linear transformation
    }
}



fn relu(x: f32) -> f32{
    x.max(0.0)
}


