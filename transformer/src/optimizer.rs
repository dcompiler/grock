use tch::{ Tensor};
use crate::decoder::D_MODEL;
use crate::utility;


struct LabelSmoothing {
    padding_idx: Vec<i32>,
    smoothing: f32,
    confidence: f32,
    size: usize,
    true_dist: Option<Tensor>, //todo: true_dist
}

impl LabelSmoothing{
    fn init_label_smoothing(size: usize, padding_idx: Vec<i32>, smoothing: f32 ) -> LabelSmoothing{
LabelSmoothing{
            padding_idx,
            smoothing,
            confidence: 1.0-smoothing,
            size,
            true_dist: None,
        }
    }

    fn forward(& mut self, mut x: Tensor, target: Tensor) ->Tensor{
        assert_eq!(x.size()[1], self.size as i64);
        let mut true_dist = target.clone(&Default::default());
        true_dist = true_dist.fill_(self.smoothing as f64 / (self.size as f64 - 2.));
        true_dist = true_dist.scatter_(1, &target.data().unsqueeze(1), &Tensor::from(self.confidence));
        for i in 0..true_dist.size()[1]{
            if self.padding_idx.contains(&(i as i32)){
                true_dist = true_dist.narrow(1, i as i64, true_dist.size()[1]).fill_(0 as f64);
            }
        }

        let nonzero = target.nonzero();
        let mut mask = vec![];
        for i in 0..nonzero.size()[1]{
            let val = nonzero.narrow(1, i, 1).f_int64_value(&[1]);
            if self.padding_idx.contains(&(val.unwrap() as i32)){
                mask.push(i);
            }
        }


        let second = true_dist.copy();
        if !mask.is_empty() {
                self.true_dist = Option::from(true_dist.index_fill_(0, &Tensor::from_slice(&mask), 0.0));
        }
        else{self.true_dist = Option::from(true_dist);} //what is purpose of self.true_dist if we use local?
        utility::kl_div_loss(&x, second)
    }
}


struct NoamOpt {
    param_groups: Vec<Tensor>,
    step: u64,
    warmup: u64,
    factor: f64,
    model_size: u64,
    rate: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
}

impl NoamOpt {
    fn new(model_size: u64, factor: f64, warmup: u64, beta1: f64, beta2: f64, eps: f64) -> Self {
        Self {
            param_groups: vec![],
            step: 0,
            warmup,
            factor,
            model_size,
            rate: 0.0,
            beta1,
            beta2,
            eps,
        }
    }

    fn step(&mut self) {
        self.step += 1;
        let rate = &self.rate(self.step);
        for param in self.param_groups.iter_mut() {
            //param.rate = rate; //todo: update rates function
        }
        self.rate = *rate;
    }

    fn rate(&mut self, step: u64) -> f64 {
       self.factor * (self.model_size as f64).powf(-0.5) * ((step as f64).powf(-0.5).min(self.warmup as f64).powf(-1.5))
    }
    fn backprop(&mut self, loss: Tensor) {
        loss.backward();
        self.step();
    }
}

fn get_std_opt() -> NoamOpt {
   NoamOpt::new(D_MODEL as u64, 2.0, 4000, 0.9, 0.98, 1e-9)
}

//add each weight to param_groups, as a paramater with associated t, 1st moment vector, and second moment vector (gradient??)
//put each paramater in param groups
//have param groups run the optimizer on each
