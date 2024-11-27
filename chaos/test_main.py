import torch
from .main import ChaoticSystemModel

import torch.nn as nn

def test_forward_pass():
  # Parameters for the model
  input_size = 3  # Number of behavior types
  hidden_size = 20  # Hidden size in the RNN
  output_size = 1  # Output: Likelihood of a pivotal state

  # Create the model
  model = ChaoticSystemModel(input_size, hidden_size, output_size)

  # Create dummy input data
  time_steps = 10
  individuals = 5
  data = torch.randn(time_steps, individuals, input_size)

  # Forward pass
  output = model(data)

  # Check the output shape
  assert output.shape == (time_steps, individuals, output_size), "Output shape is incorrect"

def test_training_step():
  # Parameters for the model
  input_size = 3  # Number of behavior types
  hidden_size = 20  # Hidden size in the RNN
  output_size = 1  # Output: Likelihood of a pivotal state

  # Create the model
  model = ChaoticSystemModel(input_size, hidden_size, output_size)

  # Create dummy input data and targets
  time_steps = 10
  individuals = 5
  data = torch.randn(time_steps, individuals, input_size)
  targets = torch.randn(time_steps, individuals, output_size)

  # Loss and optimizer
  criterion = nn.MSELoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

  # Training step
  model.train()
  outputs = model(data)
  loss = criterion(outputs, targets)
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

  # Check that the loss is a scalar
  assert loss.item() > 0, "Loss should be a positive scalar"

def test_model_response_to_perturbation():
  # Parameters for the model
  input_size = 3  # Number of behavior types
  hidden_size = 20  # Hidden size in the RNN
  output_size = 1  # Output: Likelihood of a pivotal state

  # Create the model
  model = ChaoticSystemModel(input_size, hidden_size, output_size)

  # Create dummy input data
  time_steps = 10
  individuals = 5
  data = torch.randn(time_steps, individuals, input_size)
  data_perturbed = data.clone()
  data_perturbed[2, 1, 0] += 0.1  # Small change in behavior for individual 1 at time step 2

  # Evaluate the model's response to the perturbation
  model.eval()
  with torch.no_grad():
    original_output = model(data)
    perturbed_output = model(data_perturbed)

  # Calculate the difference between the outputs
  difference = torch.abs(perturbed_output - original_output)

  # Check that the difference is non-zero
  assert torch.sum(difference).item() > 0, "Difference should be non-zero"

if __name__ == "__main__":
  test_forward_pass()
  test_training_step()
  test_model_response_to_perturbation()
  print("All tests passed!")