/*
# mLSTM: Matrix Long Short-Term Memory

This module implements the mLSTM (matrix LSTM) cell and layer as described in the paper:
"xLSTM: Extended Long Short-Term Memory" by Beck et al. (2024).

The mLSTM extends the traditional LSTM by using a matrix memory state and exponential gating,
allowing for enhanced storage capacities and improved performance on long-range dependencies.
*/

use burn::{
    config::Config,
    module::{Module, Param},
    nn::{Dropout, DropoutConfig, Initializer, Linear, LinearConfig},
    tensor::{Tensor, activation, backend::Backend},
};
use num_traits::{FromPrimitive, ToPrimitive};

/// State for mLSTM containing cell matrix and hidden state
#[derive(Clone, Debug)]
pub struct MLstmstate<B: Backend> {
    /// Cell state - matrix of shape [`batch_size`, `hidden_size`, `hidden_size`]
    pub cell: Tensor<B, 3>,
    /// Hidden state - vector of shape [`batch_size`, `hidden_size`]
    pub hidden: Tensor<B, 2>,
}

impl<B: Backend> MLstmstate<B> {
    /// Create a new mLSTM state
    pub const fn new(cell: Tensor<B, 3>, hidden: Tensor<B, 2>) -> Self {
        Self { cell, hidden }
    }
}

/// Configuration for mLSTM
#[derive(Config, Debug)]
pub struct MLstmconfig {
    /// Size of input features
    pub d_input: usize,
    /// Size of hidden state
    pub d_hidden: usize,
    /// Number of layers
    pub num_layers: usize,
    /// Dropout probability
    #[config(default = "0.0")]
    pub dropout: f64,
    /// Weight initializer
    #[config(default = "Initializer::XavierNormal{gain:1.0}")]
    pub initializer: Initializer,
}

impl MLstmconfig {
    /// Initialize a new mLSTM
    pub fn init<B: Backend>(&self, device: &B::Device) -> MLstm<B> {
        let layers = (0..self.num_layers)
            .map(|i| {
                let input_size = if i == 0 { self.d_input } else { self.d_hidden };
                MLstmcell::new(input_size, self.d_hidden, &self.initializer, device)
            })
            .collect();

        MLstm {
            layers,
            dropout_layer: DropoutConfig::new(self.dropout).init(),
            d_input: self.d_input,
            d_hidden: self.d_hidden,
            num_layers: self.num_layers,
            dropout: self.dropout,
        }
    }
}

/// mLSTM layer implementation
#[derive(Module, Debug)]
pub struct MLstm<B: Backend> {
    /// Stack of mLSTM cells
    pub layers: alloc::vec::Vec<MLstmcell<B>>,
    /// Dropout module for inter-layer dropout
    pub dropout_layer: Dropout,
    /// Input size
    pub d_input: usize,
    /// Hidden size
    pub d_hidden: usize,
    /// Number of layers
    pub num_layers: usize,
    /// Dropout probability
    pub dropout: f64,
}

impl<B: Backend> MLstm<B> {
    /// Forward pass through mLSTM consuming and returning states
    ///
    /// # Arguments
    /// * `input_seq` - Input tensor of shape [`batch_size`, `seq_length`, `input_size`]
    /// * `states` - States to consume (will be moved)
    ///
    /// # Returns
    /// * Output tensor of shape [`batch_size`, `seq_length`, `hidden_size`]
    /// * New states
    pub fn forward(
        &self,
        input_seq: &Tensor<B, 3>,
        states: Option<alloc::vec::Vec<MLstmstate<B>>>,
    ) -> (Tensor<B, 3>, alloc::vec::Vec<MLstmstate<B>>)
    where
        <B as Backend>::FloatElem: ToPrimitive + FromPrimitive,
    {
        let device = input_seq.device();
        let [batch_size, seq_length, _] = input_seq.dims();

        // Initialize or consume provided states
        let mut hidden_states = states.unwrap_or_else(|| self.init_hidden(batch_size, &device));

        let mut all_outputs = alloc::vec::Vec::with_capacity(seq_length);

        for t in 0..seq_length {
            let input_t = input_seq
                .clone()
                .slice([0..batch_size, t..(t + 1), 0..self.d_input])
                .squeeze(1);

            let mut layer_input = input_t;

            for (layer_idx, layer) in self.layers.iter().enumerate() {
                // Take ownership of the state using mem::replace
                let old_state = core::mem::replace(
                    &mut hidden_states[layer_idx],
                    MLstmstate::new(
                        Tensor::zeros([batch_size, self.d_hidden, self.d_hidden], &device),
                        Tensor::zeros([batch_size, self.d_hidden], &device),
                    ),
                );

                // Consume the state and get new state back
                let (h_new, new_state) = layer.forward(&layer_input, old_state);

                // Store the new state
                hidden_states[layer_idx] = new_state;

                // Apply dropout between layers (but not after last layer)
                layer_input = if layer_idx < self.num_layers - 1 && self.dropout > 0.0 {
                    self.dropout_layer.forward(h_new)
                } else {
                    h_new
                };
            }

            all_outputs.push(layer_input.unsqueeze_dim(1));
        }

        let output = Tensor::cat(all_outputs, 1);
        (output, hidden_states)
    }

    /// Initialize hidden states
    fn init_hidden(&self, batch_size: usize, device: &B::Device) -> alloc::vec::Vec<MLstmstate<B>> {
        (0..self.num_layers)
            .map(|_| {
                MLstmstate::new(
                    Tensor::zeros([batch_size, self.d_hidden, self.d_hidden], device),
                    Tensor::zeros([batch_size, self.d_hidden], device),
                )
            })
            .collect()
    }
}

/// mLSTM cell implementation with matrix memory
#[derive(Module, Debug)]
pub struct MLstmcell<B: Backend> {
    /// Weight matrix for input to gates
    pub weight_ih: Param<Tensor<B, 2>>,
    /// Weight matrix for hidden to gates
    pub weight_hh: Param<Tensor<B, 2>>,
    /// Bias for gates
    pub bias: Param<Tensor<B, 1>>,
    /// Query projection
    pub w_q: Linear<B>,
    /// Key projection
    pub w_k: Linear<B>,
    /// Value projection
    pub w_v: Linear<B>,
    /// Input size
    pub input_size: usize,
    /// Hidden size
    pub hidden_size: usize,
}

impl<B: Backend> MLstmcell<B> {
    /// Create a new mLSTM cell
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        initializer: &Initializer,
        device: &B::Device,
    ) -> Self {
        // 3 gates: input, forget, output
        // For weight matrix [output_size, input_size]: fan_in=input_size, fan_out=output_size
        let weight_ih = initializer.init_with(
            [3 * hidden_size, input_size],
            Some(input_size),
            Some(3 * hidden_size),
            device,
        );
        let weight_hh = initializer.init_with(
            [3 * hidden_size, hidden_size],
            Some(hidden_size),
            Some(3 * hidden_size),
            device,
        );

        // Initialize biases with specific values for stability
        let mut bias_data = alloc::vec![0.0; 3 * hidden_size];
        for item in bias_data.iter_mut().take(hidden_size) {
            *item = -3.0;
        }
        for item in bias_data.iter_mut().take(2 * hidden_size).skip(hidden_size) {
            *item = -3.0;
        }
        let bias = Tensor::from_floats(bias_data.as_slice(), device);

        let w_q = LinearConfig::new(input_size, hidden_size)
            .with_bias(false)
            .with_initializer(Initializer::XavierUniform { gain: 0.5 })
            .init(device);
        let w_k = LinearConfig::new(input_size, hidden_size)
            .with_bias(false)
            .with_initializer(Initializer::XavierUniform { gain: 0.5 })
            .init(device);
        let w_v = LinearConfig::new(input_size, hidden_size)
            .with_bias(false)
            .with_initializer(Initializer::XavierUniform { gain: 0.5 })
            .init(device);

        Self {
            weight_ih,
            weight_hh,
            bias: Param::from_tensor(bias),
            w_q,
            w_k,
            w_v,
            input_size,
            hidden_size,
        }
    }

    /// Forward pass through mLSTM cell consuming the state
    ///
    /// # Arguments
    /// * `input` - Input tensor [`batch_size`, `input_size`]
    /// * `state` - State to consume (moved)
    ///
    /// # Returns
    /// * New hidden state (for output)
    /// * New complete state
    pub fn forward(
        &self,
        input: &Tensor<B, 2>,
        state: MLstmstate<B>,
    ) -> (Tensor<B, 2>, MLstmstate<B>)
    where
        <B as Backend>::FloatElem: num_traits::ToPrimitive + num_traits::FromPrimitive + Copy,
    {
        // Destructure state to get ownership of tensors
        let MLstmstate { cell, hidden } = state;

        // Compute gates: i, f, o
        let gates = input.clone().matmul(self.weight_ih.val().transpose())
            + hidden.matmul(self.weight_hh.val().transpose())
            + self.bias.val().unsqueeze_dim(0);

        let chunks = gates.chunk(3, 1);
        let i_gate = &chunks[0];
        let f_gate = &chunks[1];
        let o_gate = &chunks[2];

        // Exponential gating with stabilization
        let i = i_gate.clone().clamp(-15.0, 8.0).exp();
        let f = f_gate.clone().clamp(-15.0, 8.0).exp();
        let o = activation::sigmoid(o_gate.clone());

        // Compute query/key/value, scale q/k by sqrt(hidden_size)
        let scale_scalar = (self.hidden_size as f64).sqrt();
        let scale_elem: <B as Backend>::FloatElem =
            num_traits::FromPrimitive::from_f64(scale_scalar)
                .expect("Failed to cast scale to backend float");
        let q = self.w_q.forward(input.clone()) / scale_elem;
        let k = self.w_k.forward(input.clone()) / scale_elem;
        let v = self.w_v.forward(input.clone());

        // Outer product: v ⊗ k^T -> [batch, hidden, hidden]
        let v_unsqueezed = v.unsqueeze_dim(2); // [batch, hidden, 1]
        let k_unsqueezed = k.unsqueeze_dim(1); // [batch, 1, hidden]
        let outer_product = v_unsqueezed.matmul(k_unsqueezed);

        // Update cell state: C = f*C + i*(v ⊗ k^T)
        let f_expanded = f.unsqueeze_dim(2);
        let i_expanded = i.unsqueeze_dim(2);

        let mut c_new = f_expanded * cell + i_expanded * outer_product;

        // --- Soft normalization (backend-generic) ---
        let c_abs_max_elem: <B as Backend>::FloatElem = c_new.clone().abs().max().into_scalar();
        let c_abs_max_f64 = num_traits::ToPrimitive::to_f64(&c_abs_max_elem).unwrap_or(0.0);

        if c_abs_max_f64 > 1e-8 {
            let scale_factor_f64 = 1.0 / (1.0 + c_abs_max_f64 / 10.0);
            let scale_factor_elem: <B as Backend>::FloatElem =
                num_traits::FromPrimitive::from_f64(scale_factor_f64)
                    .expect("Failed to cast scale factor to backend float");

            c_new = c_new * scale_factor_elem;
        }
        // ------------------------------------------

        // Compute hidden state: h = o * (q^T @ C)
        let q_unsqueezed = q.unsqueeze_dim(1);
        let qc = q_unsqueezed.matmul(c_new.clone()).squeeze(1);
        let h_new = o * qc;

        // Return both the hidden state (for output) and new complete state
        let new_state = MLstmstate::new(c_new, h_new.clone());
        (h_new, new_state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Distribution;

    type TestBackend = burn_ndarray::NdArray<f32>;

    #[test]
    fn test_mlstm_forward() {
        let device = Default::default();
        let config = MLstmconfig::new(64, 128, 2).with_dropout(0.1);
        let mlstm = config.init::<TestBackend>(&device);

        let input = Tensor::<TestBackend, 3>::random([4, 10, 64], Distribution::Default, &device);

        let (output, states) = mlstm.forward(&input, None);

        assert_eq!(output.dims(), [4, 10, 128]);
        assert_eq!(states.len(), 2);
        assert_eq!(states[0].hidden.dims(), [4, 128]);
        assert_eq!(states[0].cell.dims(), [4, 128, 128]);
    }

    #[test]
    fn test_mlstm_cell() {
        let device = Default::default();
        let cell = MLstmcell::new(32, 64, &Initializer::XavierNormal { gain: 1.0 }, &device);

        let input = Tensor::<TestBackend, 2>::random([4, 32], Distribution::Default, &device);
        let state = MLstmstate::new(
            Tensor::<TestBackend, 3>::zeros([4, 64, 64], &device),
            Tensor::<TestBackend, 2>::zeros([4, 64], &device),
        );

        let (h_new, new_state) = cell.forward(&input, state);

        assert_eq!(h_new.dims(), [4, 64]);
        assert_eq!(new_state.cell.dims(), [4, 64, 64]);
        assert_eq!(new_state.hidden.dims(), [4, 64]);
    }

    #[test]
    fn test_mlstm_state_reuse() {
        let device = Default::default();
        let config = MLstmconfig::new(32, 64, 1);
        let mlstm = config.init::<TestBackend>(&device);

        let input1 = Tensor::<TestBackend, 3>::random([2, 5, 32], Distribution::Default, &device);
        let input2 = Tensor::<TestBackend, 3>::random([2, 5, 32], Distribution::Default, &device);

        // First forward pass
        let (output1, states) = mlstm.forward(&input1, None);

        // Second forward pass reusing states
        let (output2, _final_states) = mlstm.forward(&input2, Some(states));

        assert_eq!(output1.dims(), [2, 5, 64]);
        assert_eq!(output2.dims(), [2, 5, 64]);
    }
}
