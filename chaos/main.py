import torch
import torch.nn as nn
import random
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the Tensor Substrate (Sociological Data)
time_steps = 365  # Number of days
individuals = [
    "Person A", "Person B", "Person C", "Person D", "Person E", "Person F", "Person G", "Person H",
    "Person I", "Person J", "Person K", "Person L", "Person M", "Person N", "Person O", "Person P",
    "Person Q", "Person R", "Person S", "Person T", "Person U", "Person V", "Person W", "Person X",
    "Person Y", "Person Z", "Person AA", "Person AB", "Person AC", "Person AD", "Person AE", "Person AF",
    "Person AG", "Person AH", "Person AI", "Person AJ", "Person AK", "Person AL", "Person AM", "Person AN",
    "Person AO", "Person AP", "Person AQ", "Person AR", "Person AS", "Person AT", "Person AU", "Person AV",
    "Person AW", "Person AX"
]
behaviors = [
    "Post", "Like", "Share", "Comment", "Follow", "Unfollow", "Block", "Unblock",
    "Message", "Reply", "Retweet", "Favorite", "Mention", "Tag", "Upload", "Download",
    "Stream", "Subscribe", "Unsubscribe", "Join", "Leave", "Check-in", "Rate", "Review",
    "Bookmark", "Report", "Mute", "Unmute", "Pin", "Unpin", "React", "Invite", "Accept",
    "Decline", "Edit", "Delete", "Create", "Update", "View", "Search", "Browse", "Purchase",
    "Sell", "Donate", "Volunteer", "Attend", "RSVP", "Cancel", "Save", "Share Story", "Vote"
]

# Create a tensor with random values to simulate sociological data
# Shape: [time_steps, num_individuals, num_behaviors]
data = torch.randn(time_steps, len(individuals), len(behaviors)).to(device)
print("Sociological Data Tensor created.")

# Flatten the tensor data to 2D
data_flattened = data.view(-1, len(behaviors)).cpu().numpy()

# Standardize the data
scaler = StandardScaler()
data_standardized = scaler.fit_transform(data_flattened)

# Apply PCA
pca_components = 10  # Reduce to 10 principal components
pca = PCA(n_components=pca_components)
data_pca = pca.fit_transform(data_standardized)

# Reshape the PCA-transformed data back to 3D tensor format
data_pca_reshaped = torch.tensor(data_pca).view(time_steps, len(individuals), -1).float().to(device)

# (Optional) Visualize the PCA result
plt.figure(figsize=(10, 6))
plt.scatter(data_pca[:, 0], data_pca[:, 1], alpha=0.5)
plt.title("PCA of Sociological Data")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()

# Create the Chaotic System Model
class ChaoticSystemModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=3, dropout=0.3):
        super(ChaoticSystemModel, self).__init__()
        # Multi-layer bidirectional LSTM with dropout
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, dropout=dropout,
                            bidirectional=True, batch_first=True)
        # Fully connected layer
        self.fc = nn.Linear(hidden_size * 2, output_size)  # Multiply by 2 for bidirectional

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out)
        return output

# Parameters for the model
input_size = data_pca_reshaped.shape[2]  # Number of principal components
hidden_size = 50                         # Hidden size in the LSTM
output_size = len(behaviors)             # Number of behaviors
num_layers = 3                           # Number of LSTM layers
dropout = 0.3                            # Dropout rate

# Create the model
model = ChaoticSystemModel(input_size, hidden_size, output_size, num_layers, dropout).to(device)

# Training the Model
# Dummy target values for training
targets = torch.randn(time_steps, len(individuals), output_size).to(device)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
epochs = 1000
for epoch in range(epochs):
    model.train()
    outputs = model(data_pca_reshaped)
    loss = criterion(outputs, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}')

# Introduce Random Perturbations
perturbation_count = 0
while True:
    perturbation_count += 1
    data_perturbed = data_pca_reshaped.clone()
    # Introduce a random perturbation
    random_time = random.randint(0, time_steps - 1)
    random_individual = random.randint(0, len(individuals) - 1)
    random_component = random.randint(0, data_pca_reshaped.shape[2] - 1)
    perturbation_value = random.uniform(0.1, 1.0)
    data_perturbed[random_time, random_individual, random_component] += perturbation_value

    print(f"Perturbation {perturbation_count}: Time Step {random_time}, Individual {individuals[random_individual]}, "
          f"Principal Component {random_component}, Perturbation Value {perturbation_value:.4f}")

    # Evaluate the model's response to the perturbation
    model.eval()
    with torch.no_grad():
        original_output = model(data_pca_reshaped)
        perturbed_output = model(data_perturbed)

    # Calculate the difference between the outputs
    difference = torch.abs(perturbed_output - original_output)

    # Focus on the 'Vote' behavior
    vote_behavior_idx = behaviors.index('Vote')
    difference_vote = difference[:, :, vote_behavior_idx]

    # Find pivotal states where the difference exceeds 0.05
    pivotal_states = (difference_vote > 0.05).nonzero(as_tuple=True)

    # Check if at least 25% of the individuals have a pivotal change in 'Vote'
    unique_individuals = set(pivotal_states[1].tolist())
    print(f"Unique individuals with pivotal change in 'Vote': {len(unique_individuals)}")
    if len(unique_individuals) >= len(individuals) * 0.25:
        break

# Print Pivotal States in Human-Readable Form
print(f"\nPivotal states identified after {perturbation_count} perturbation(s):")
for time_idx, ind_idx in zip(*pivotal_states):
    individual = individuals[ind_idx]
    time_step = time_idx.item()
    print(f"Time Step {time_step}: {individual} had a pivotal change in 'Vote'.")