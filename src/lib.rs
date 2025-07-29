mod rng;
use rng::{SimpleRng, shuffle_slice};

pub trait ModelDefinition<D, P> {
    fn gradient_compute(&self, data_sample: &D, parameters: &P) -> Vec<f64>;
    fn update_parameters(&self, parameters: &mut P, gradients: &[f64], learning_rate: f64);
    fn calculate_loss(&self, _dataset: &[D], _parameters: &P) -> Option<f64> {
        None
    }
}

pub struct StochasticGradientDescent {
    learning_rate: f64,
    epochs: usize,
    batch_size: usize,
    shuffle: bool, // Whether to shuffle data each epoch
    rng_seed: u64, // Seed for the internal RNG
}

impl StochasticGradientDescent {
    pub fn new(
        learning_rate: f64,
        epochs: usize,
        batch_size: usize,
        shuffle: bool,
        rng_seed: u64,
    ) -> Self {
        if batch_size == 0 {
            panic!("Batch size cannot be zero.");
        }
        StochasticGradientDescent {
            learning_rate,
            epochs,
            batch_size,
            shuffle,
            rng_seed,
        }
    }

    pub fn optimize<M, D, P>(&self, model: &M, parameters: &mut P, dataset: &[D])
    where
        M: ModelDefinition<D, P>,
        // D: Sync + Send, // Not strictly needed without parallelization yet
        // P: Clone,       // Not strictly needed for this implementation path
    {
        if dataset.is_empty() {
            println!("Warning: Dataset is empty. No optimization will be performed.");
            return;
        }

        let n_samples = dataset.len();
        let effective_batch_size = self.batch_size.min(n_samples); // Ensure batch size isn't too large

        if self.batch_size > n_samples {
            println!(
                "Warning: Batch size ({}) is larger than the dataset size ({}). \
                 Using batch size equal to dataset size ({}).",
                self.batch_size, n_samples, effective_batch_size
            );
        }

        let mut rng = SimpleRng::new(self.rng_seed);
        let mut indices: Vec<usize> = (0..n_samples).collect();
        for epoch in 0..self.epochs {
            if self.shuffle {
                shuffle_slice(&mut indices, &mut rng);
            }
            for batch_start in (0..n_samples).step_by(effective_batch_size) {
                let batch_end = (batch_start + effective_batch_size).min(n_samples);
                if batch_start == batch_end {
                    continue;
                }
                let current_batch_indices = &indices[batch_start..batch_end];
                let mut accumulated_gradients: Option<Vec<f64>> = None;
                for &data_idx in current_batch_indices {
                    let data_sample = &dataset[data_idx];
                    let sample_gradient = model.gradient_compute(data_sample, parameters);

                    if accumulated_gradients.is_none() {
                        accumulated_gradients = Some(sample_gradient);
                    } else {
                        if let Some(acc_grads) = accumulated_gradients.as_mut() {
                            if acc_grads.len() != sample_gradient.len() {
                                panic!(
                                    "Gradient dimension mismatch. Expected {}, got {}. \
                                     Ensure gradient compute consistently returns gradients of the same length.",
                                    acc_grads.len(),
                                    sample_gradient.len()
                                );
                            }
                            for (ag, sg) in acc_grads.iter_mut().zip(sample_gradient.iter()) {
                                *ag += *sg;
                            }
                        }
                    }
                }

                if let Some(mut total_batch_gradients) = accumulated_gradients {
                    if !total_batch_gradients.is_empty() {
                        let num_in_batch = current_batch_indices.len() as f64;
                        if num_in_batch > 0.0 {
                            for grad_component in total_batch_gradients.iter_mut() {
                                *grad_component /= num_in_batch; // Average the gradients
                            }
                        }
                        model.update_parameters(
                            parameters,
                            &total_batch_gradients,
                            self.learning_rate,
                        );
                    }
                }
            }

            if let Some(loss) = model.calculate_loss(dataset, parameters) {
                println!("Epoch: {}, Loss: {:.4}", epoch + 1, loss);
            } else {
                println!("Epoch: {} completed.", epoch + 1);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Debug, Clone)] // Added Clone for easier dataset creation in tests
    struct SampleData {
        features: Vec<f64>,
        label: f64,
    }
    struct LinearRegressionModel {
        num_features: usize,
    }
    impl ModelDefinition<SampleData, Vec<f64>> for LinearRegressionModel {
        fn gradient_compute(&self, data_sample: &SampleData, parameters: &Vec<f64>) -> Vec<f64> {
            if self.num_features == 0 {
                // Model with only bias
                if parameters.len() != 1 {
                    panic!("Parameter length mismatch for bias-only model.");
                }
                let prediction = parameters[0]; // bias
                let error = prediction - data_sample.label;
                return vec![error]; // Gradient for bias
            }

            // +1 for bias
            if parameters.len() != self.num_features + 1 {
                panic!(
                    "Parameter length mismatch in gradient compute. Expected {}, got {}. Features: {}",
                    self.num_features + 1,
                    parameters.len(),
                    self.num_features
                );
            }

            let mut prediction = parameters[0]; // bias
            for i in 0..self.num_features {
                prediction += parameters[i + 1] * data_sample.features[i];
            }
            let error = prediction - data_sample.label;

            let mut gradients = vec![0.0; self.num_features + 1];
            gradients[0] = error; // Gradient for bias
            for i in 0..self.num_features {
                gradients[i + 1] = error * data_sample.features[i]; // Gradient for weight_i
            }
            gradients
        }

        fn update_parameters(
            &self,
            parameters: &mut Vec<f64>,
            gradients: &[f64],
            learning_rate: f64,
        ) {
            if parameters.is_empty() && gradients.is_empty() {
                return; // No parameters to update
            }
            if parameters.len() != gradients.len() {
                panic!(
                    "Parameter and gradient length mismatch in update_parameters. Params: {}, Grads: {}",
                    parameters.len(),
                    gradients.len()
                );
            }
            for i in 0..parameters.len() {
                parameters[i] -= learning_rate * gradients[i];
            }
        }

        fn calculate_loss(&self, dataset: &[SampleData], parameters: &Vec<f64>) -> Option<f64> {
            if dataset.is_empty() {
                return Some(0.0);
            }
            if self.num_features == 0 {
                // Model with only bias
                if parameters.len() != 1 {
                    panic!("Parameter length mismatch for bias-only model in loss calc.");
                }
                let mut total_squared_error = 0.0;
                for sample in dataset {
                    let prediction = parameters[0]; // bias
                    let error = prediction - sample.label;
                    total_squared_error += error * error;
                }
                return Some(total_squared_error / (2.0 * dataset.len() as f64));
            }

            if parameters.len() != self.num_features + 1 {
                return None; // Or panic, indicates a problem
            }
            let mut total_squared_error = 0.0;
            for sample in dataset {
                let mut prediction = parameters[0]; // bias
                for i in 0..self.num_features {
                    prediction += parameters[i + 1] * sample.features[i];
                }
                let error = prediction - sample.label;
                total_squared_error += error * error;
            }
            Some(total_squared_error / (2.0 * dataset.len() as f64)) // Mean Squared Error
        }
    }

    #[test]
    fn test_linear_regression_sgd() {
        // Create a simple dataset: y = 2*x1 + 3*x2 + 1
        let dataset = vec![
            SampleData {
                features: vec![1.0, 1.0],
                label: 6.0,
            },
            SampleData {
                features: vec![2.0, 3.0],
                label: 14.0,
            },
            SampleData {
                features: vec![3.0, 2.0],
                label: 13.0,
            },
            SampleData {
                features: vec![0.0, 0.0],
                label: 1.0,
            },
            SampleData {
                features: vec![5.0, 1.0],
                label: 14.0,
            },
        ];
        let model = LinearRegressionModel { num_features: 2 };
        let mut initial_parameters = vec![0.0, 0.0, 0.0]; // bias, w1, w2
        let sgd = StochasticGradientDescent::new(0.01, 100, 2, true, 12345);
        sgd.optimize(&model, &mut initial_parameters, &dataset);

        println!("Optimized parameters: {:?}", initial_parameters);
        assert!(initial_parameters[0] > 0.5 && initial_parameters[0] < 1.5); // Bias close to 1
        assert!(initial_parameters[1] > 1.5 && initial_parameters[1] < 2.5); // w1 close to 2
        assert!(initial_parameters[2] > 2.5 && initial_parameters[2] < 3.5); // w2 close to 3

        let initial_loss_val = vec![0.0, 0.0, 0.0];
        let initial_loss = model.calculate_loss(&dataset, &initial_loss_val).unwrap();
        let final_loss = model.calculate_loss(&dataset, &initial_parameters).unwrap();
        println!("Initial Loss: {}, Final Loss: {}", initial_loss, final_loss);
        assert!(final_loss < initial_loss);
    }

    #[test]
    fn test_sgd_full_batch_no_shuffle() {
        let dataset = vec![
            SampleData {
                features: vec![1.0],
                label: 2.0,
            }, // y = 2x, bias = 0
            SampleData {
                features: vec![2.0],
                label: 4.0,
            },
        ];
        let model = LinearRegressionModel { num_features: 1 };
        let mut params = vec![0.0, 0.0]; // bias, w1

        // No shuffle, seed doesn't matter but provide one
        let sgd = StochasticGradientDescent::new(0.01, 100, dataset.len(), false, 0);
        sgd.optimize(&model, &mut params, &dataset);
        println!("Full batch optimized parameters: {:?}", params);
        // For y = 2x (bias=0, w1=2)
        assert!(
            params[0].abs() < 1.0,
            "Bias : {} should be very close to 0",
            params[0]
        );
        assert!(
            (params[1] - 2.0).abs() < 1.0,
            "w1 : {} should be very close to 2",
            params[1]
        );
    }

    #[test]
    #[should_panic]
    fn test_zero_batch_size() {
        let _ = StochasticGradientDescent::new(0.01, 100, 0, true, 0);
    }

    #[test]
    fn test_empty_dataset() {
        let dataset: Vec<SampleData> = Vec::new();
        let model = LinearRegressionModel { num_features: 1 };
        let mut params = vec![0.0, 0.0];
        let original_params = params.clone();

        let sgd = StochasticGradientDescent::new(0.01, 10, 1, true, 0);
        sgd.optimize(&model, &mut params, &dataset);

        assert_eq!(
            params, original_params,
            "Parameters should not change for an empty dataset."
        );
        println!("Optimization with empty dataset completed. Parameters unchanged.");
    }

    #[test]
    fn test_model_with_only_bias() {
        // Dataset where y = constant (e.g., y = 3)
        let dataset = vec![
            SampleData {
                features: vec![],
                label: 3.0,
            },
            SampleData {
                features: vec![],
                label: 3.1,
            },
            SampleData {
                features: vec![],
                label: 2.9,
            },
            SampleData {
                features: vec![],
                label: 3.05,
            },
        ];

        let model = LinearRegressionModel { num_features: 0 }; // Only bias
        let mut initial_parameters = vec![0.0]; // Only bias parameter

        let sgd = StochasticGradientDescent::new(0.1, 50, 2, true, 42);
        sgd.optimize(&model, &mut initial_parameters, &dataset);

        println!("Optimized bias-only parameter: {:?}", initial_parameters);
        // The parameter should be close to the average of labels (around 3.0125)
        assert!(initial_parameters.len() == 1);
        assert!(
            (initial_parameters[0] - 3.0125).abs() < 0.1,
            "Bias should be close to the mean of labels."
        );

        let final_loss = model.calculate_loss(&dataset, &initial_parameters).unwrap();
        println!("Final loss for bias-only model: {}", final_loss);
        assert!(final_loss < 0.01); // Expect low loss
    }

    #[test]
    fn test_shuffle_consistency_with_seed() {
        let mut indices1: Vec<usize> = (0..10).collect();
        let mut indices2: Vec<usize> = (0..10).collect();

        let mut rng1 = SimpleRng::new(123);
        let mut rng2 = SimpleRng::new(123);

        shuffle_slice(&mut indices1, &mut rng1);
        shuffle_slice(&mut indices2, &mut rng2);

        assert_eq!(
            indices1, indices2,
            "Shuffle should be consistent for the same seed."
        );

        let mut rng3 = SimpleRng::new(456);
        let mut indices3: Vec<usize> = (0..10).collect();
        shuffle_slice(&mut indices3, &mut rng3);

        assert_ne!(
            indices1, indices3,
            "Shuffle should be different for different seeds."
        );
    }
}
