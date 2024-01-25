

# **Gaussian Mixture Prior VAE Model using MNIST Dataset**
"""

# This is the General Code

"""### **Libraries**"""

# Commented out IPython magic to ensure Python compatibility.
import warnings
warnings.filterwarnings("ignore")

# Debugger
import pdb

import time

# math and numpy
import itertools
import decimal
import mpmath
import math
import numpy as np
import random

# pandas
import pandas as pd
from pandas import DataFrame
from collections import Counter

# matplot
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from IPython import display
display.set_matplotlib_formats("svg")
# %matplotlib inline
import matplotlib.patches as mpatches

# scipy
import scipy.linalg as la
from scipy.stats import entropy
from scipy.stats import norm
from scipy.stats import multivariate_normal
import scipy.io as scio

# sklearn
from sklearn import datasets
from sklearn import metrics
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, accuracy_score
from scipy.optimize import linear_sum_assignment
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler


# torch
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.optim.lr_scheduler import StepLR
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import make_grid
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from torch.distributions.mixture_same_family import MixtureSameFamily
from torch.distributions.categorical import Categorical
from torch.distributions.wishart import Wishart
from torch.distributions.dirichlet import Dirichlet
from torch.distributions import Bernoulli
from torch.utils.data import Dataset
import torchvision.models as models
from torchvision.models import resnet50
from torchvision.datasets import STL10

"""### **Device**"""

# start_time = time.time()
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

"""## **Hyperparameters**"""

# VAE Part

pixel_size=[28, 28]
num_epochs=1000
learning_rate=1e-4
step_size=20
weight_decay=1e-5
input_dim=pixel_size[0]*pixel_size[1]
encoder_hidden_dim_1=500
encoder_hidden_dim_2=500
encoder_hidden_dim_3=2000
latent_dim=10
decoder_hidden_dim_1=2000
decoder_hidden_dim_2=500
decoder_hidden_dim_3=500
output_dim=pixel_size[0]*pixel_size[1]
train_batch_size=100
test_batch_size=100
gamma=0.9


# GMM Part

num_components=10
num_iterations=20
epsilon=1e-10
decimal.getcontext().prec = 28
clustering_method="GMM"
covariance_type="full"


# Set seeds for NumPy, PyTorch, and Python's random module

random_seed=10
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
random.seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# Early Stopping

max_patience=num_epochs+1
best_test_loss=float("inf")
patience_counter=0

# File Name

Dataset_name = "MNIST"
Model = "VADE"
Model_name= f"{Model}_{clustering_method}_{covariance_type}_{Dataset_name}"

"""# **MNIST Dataset**

### **Summary of Dataset**
"""

t = transforms.Compose([transforms.ToTensor()])
data_dir = '~/Desktop/MNIST_Dataset/data'
trainset = datasets.MNIST(data_dir, train=True, download=True, transform=t)
testset = datasets.MNIST(data_dir, train=False, download=True, transform=t)

"""### **Data Loader**"""

train_loader=DataLoader(trainset, batch_size=train_batch_size, shuffle=True)
test_loader=DataLoader(testset, batch_size=test_batch_size, shuffle=False)

"""## **Visualizing Images**"""

with torch.no_grad():
  # retrieve the next batch of data from a DataLoader
  image,target=next(iter(train_loader))
  image=image.cpu()
  image=image.clamp(0,1)
  image=image[:50]
  image=make_grid(image, 10, 5)
  image=image.numpy()
  image=np.transpose(image,(1,2,0))
  plt.imshow(image)
  plt.savefig(f'Random_Image_{Dataset_name}.png')
  plt.close()

"""# **Functions**

### **Clustering Accuracy with Linear Assignment**
"""

def cluster_acc_with_assignment(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)

    new_predicted_labels = np.array([col_ind[i] for i in y_pred])
    accuracy = sum([w[row, col] for row, col in zip(row_ind, col_ind)]) * 1.0 / y_pred.size
    return accuracy, new_predicted_labels

"""# **VADE Model**

### **Encoder Class**
"""

class Encoder(nn.Module):
  def __init__(self):
    super(Encoder, self).__init__()
    self.encoder = nn.Sequential(
        nn.Linear(input_dim, encoder_hidden_dim_1),
        nn.ReLU(),
        nn.Linear(encoder_hidden_dim_1, encoder_hidden_dim_2),
        nn.ReLU(),
        nn.Linear(encoder_hidden_dim_2, encoder_hidden_dim_3),
        nn.ReLU(),
    )
    self.mu_layer = nn.Linear(encoder_hidden_dim_3, latent_dim)
    self.log_var_layer = nn.Linear(encoder_hidden_dim_3, latent_dim)

  def forward(self, x):
    x = self.encoder(x)
    return self.mu_layer(x), self.log_var_layer(x)

"""### **Decoder Class**"""

class Decoder(nn.Module):
  def __init__(self):
    super(Decoder, self).__init__()
    self.decoder = nn.Sequential(
        nn.Linear(latent_dim, decoder_hidden_dim_1),
        nn.ReLU(),
        nn.Linear(decoder_hidden_dim_1, decoder_hidden_dim_2),
        nn.ReLU(),
        nn.Linear(decoder_hidden_dim_2, decoder_hidden_dim_3),
        nn.ReLU(),
        nn.Linear(decoder_hidden_dim_3, output_dim),
        nn.Sigmoid()
    )
  def forward(self, z):
    return self.decoder(z)

"""### **GMM Prior Class**"""

class GMM(nn.Module):
  def __init__(self, M, K):
    super(GMM, self).__init__()
    self.M=M
    self.K=K

    # Distribution Parameters
    self.means=nn.Parameter(torch.randn(self.K, self.M))                          # [K, M]
    self.logvariances=nn.Parameter(torch.randn(self.K, self.M))                   # [K, M]

    # Mixing Weights
    # self.weight=nn.Parameter(torch.rand(self.K))                               # [K]
    self.weights=nn.Parameter(Dirichlet(torch.ones(self.K)).sample())             # [K]


  def get_params(self):
    return self.means, self.logvariances, self.weights


  def log_prob_gmm(self, z):

    num_samples=z.shape[0]

    # Parameters
    mean, logvariance, weight=self.get_params()
    weight=F.softmax(weight, dim=0)

    # log-mixture-of-Gaussians

    z=z.reshape(num_samples, 1, self.M)                                          # [B, 1, M]
    weight=weight.reshape(1, self.K)                                             # [1, K]
    mean=mean.reshape(1, self.K, self.M)                                         # [1, K, M]
    logvariance=logvariance.reshape(1, self.K, self.M)                           # [1, K, M]

    log_weight=torch.log(weight)
    dist_prior=Normal(loc=mean, scale=torch.exp(0.5*logvariance))                # [B, K, M]
    log_normal_probability=dist_prior.log_prob(z).sum(dim=2)                     # [B, K]

    # q(y_i_k)
    q_numerator=torch.exp(log_weight+log_normal_probability)
    q_denominator=torch.sum(q_numerator, dim=1, keepdim=True)                    # [B, 1]
    q_y_i_k= q_numerator/q_denominator                                           # [B, K]
    return q_y_i_k, log_weight, log_normal_probability                           # [B, K], [B]

"""### **VADE Class**"""

class VADE(nn.Module):
  def __init__(self):
    super(VADE, self).__init__()
    self.encoder=Encoder()
    self.decoder=Decoder()
    self.prior=GMM(M=latent_dim, K=num_components)

  def reparameterize(self, mu, log_var):
    if self.training:
      sigma=torch.exp(0.5*log_var)
      eps=torch.randn_like(sigma)
      return mu+eps*sigma
    else:
      return mu

  def forward(self,x):
    x=x.view(-1, input_dim)
    encoder_mu, encoder_logvar=self.encoder(x)
    latent_z=self.reparameterize(encoder_mu, encoder_logvar)
    q_y_ik, prior_log_weight, prior_log_normal_probability=self.prior.log_prob_gmm(latent_z)
    decoder_mu=self.decoder(latent_z)
    return decoder_mu, q_y_ik, prior_log_weight, prior_log_normal_probability, latent_z, encoder_mu, encoder_logvar

"""### **VADE Model and Optimizer**"""

model=VADE().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
print(model)
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('\n \n Number of Estimated Parameters: %d' % num_params)

"""# **Model Training**

### **Backpropagation and Parameter Updates**
"""

Training_Losses=[]
Test_Loss=[]
True_Labels=[]
Latent_embeddings=[]

# GMM Part
Accuracy_GMM=[]
Mean_Accuracy_GMM=[]
STD_Accuracy_GMM=[]
NMI_GMM=[]
ARI_GMM=[]
Posterior_EM=[]
Updated_GMM_Labels=[]

# Posterior Probability
Accuracy_Posterior=[]
NMI_Posterior=[]
ARI_Posterior=[]
Updated_Posterior_Labels=[]
Posterior_NN=[]

print(f"Training and Testing {Model_name} Model: {Dataset_name} Dataset: \n\n")

for epoch in range(num_epochs):

  ## Training Step
  model.train()
  Training_Losses.append(0)
  num_batches=0

  for training_features, _ in train_loader:
    optimizer.zero_grad()
    training_features=training_features.reshape(-1, input_dim)
    training_features=training_features.to(device)

    decoder_mu, q_y_i_k, prior_log_weight, prior_log_normal_probability, latent_z, latent_mu, latent_logvar=model(training_features)
    decoder_mu=decoder_mu.to(device)                                             # [B, K, D]
    batch_size, batch_dim = training_features.shape

    # Encoder Likelihood: [B, M]

    q_z_x= Normal(loc=latent_mu, scale=torch.exp(0.5*latent_logvar))
    log_q_z_x = torch.sum(q_z_x.log_prob(latent_z))

    # GMM Prior Likelihood: [B, K]

    ###################### Soft Clustering #####################################
    index=q_y_i_k==0
    q_y_i_k[index]+=epsilon

    prior_lik=prior_log_weight+prior_log_normal_probability-torch.log(q_y_i_k)
    prior_log_likelihood=q_y_i_k*prior_lik
    log_p_z=torch.sum(prior_log_likelihood)

    ############################################################################


    # Decoder Likelihood: [B, D]

    log_p_x_z= -F.binary_cross_entropy(input=decoder_mu, target=training_features, reduction="sum")


    ############################################################################

    ELBO=(log_p_x_z-log_q_z_x+log_p_z)                                           # [B, K]


    # Total Loss
    Total_Loss = -ELBO

    # Backpropagation
    Total_Loss.backward()
    optimizer.step()
    num_batches+=1
    Training_Losses[-1]+=Total_Loss.item()

  # Update the Learning Rate
  scheduler.step()

  # Average Losses
  Training_Losses[-1]/= num_batches

  ## Evaluation Step
  model.eval()
  Final_Test_Loss=0.0
  number_of_batches=0
  latent_space=[]
  ground_truth_labels=[]


  # Posterior Part
  posterior_probabilities=[]

  # GMM Part
  simulated_gmm_accuracies = []
  simulated_gmm_nmi = []
  simulated_gmm_ari = []
  simulated_predicted_labels_gmm = []
  simulated_posterior_probabilities_gmm = []
  reconstructions = []
  original_image = []


  with torch.no_grad():
    for test_features, test_labels in test_loader:

      test_features=test_features.reshape(-1, input_dim)
      test_features=test_features.to(device)
      test_labels=test_labels.to(device)

      decoder_mu_test, q_y_i_k_test, prior_log_weight_test, prior_log_normal_probability_test, latent_z_test, latent_mu_test, latent_logvar_test=model(test_features)
      decoder_mu_test=decoder_mu_test.to(device)
      batch_size, batch_dim = test_features.shape

      latent_space.append(latent_z_test)
      ground_truth_labels.append(test_labels.cpu().numpy())
      posterior_probabilities.append(q_y_i_k_test)

      # Encoder Likelihood: Shape [B, M]

      q_z_x_test = Normal(loc=latent_mu_test, scale=torch.exp(0.5*latent_logvar_test))
      log_q_z_x_test = torch.sum(q_z_x_test.log_prob(latent_z_test))

      # GMM Prior Likelihood: [B, K]

      ###################### Soft Clustering #####################################
      index=q_y_i_k_test==0
      q_y_i_k_test[index]+=epsilon

      prior_lik_test = prior_log_weight_test + prior_log_normal_probability_test-torch.log(q_y_i_k_test)
      prior_log_likelihood_test=q_y_i_k_test*prior_lik_test
      log_p_z_test=torch.sum(prior_log_likelihood_test)

      ############################################################################

      # Decoder log-Likelihood

      log_p_x_z_test= -F.binary_cross_entropy(input=decoder_mu_test, target=test_features, reduction="sum")


      ######################### Test ELBO  #########################

      ELBO_test=(log_p_x_z_test-log_q_z_x_test+log_p_z_test)

      # Total Loss
      Total_Test_Loss = -ELBO_test
      Final_Test_Loss+= Total_Test_Loss.item()
      number_of_batches+= 1

      reconstructions.append(decoder_mu_test)
      original_image.append(test_features)

    # Average Loss
    Final_Test_Loss/=number_of_batches
    Test_Loss.append(Final_Test_Loss)

    # Latent Space
    latent_space=torch.cat(latent_space, dim=0).detach().cpu().numpy()
    Latent_embeddings.append(latent_space)


    # True Labels
    ground_truth_labels_numpy = np.concatenate(ground_truth_labels)
    True_Labels.append(ground_truth_labels_numpy)

    ################## Clustering using Posterior Probability###################


    # Clustering using Posterior Probabilities (Soft Assignment)

    posterior_probabilities_q_y_i_k=torch.cat(posterior_probabilities, dim=0)
    predicted_labels_posterior=torch.argmax(posterior_probabilities_q_y_i_k, dim=1).cpu().numpy()
    Posterior_NN.append(posterior_probabilities_q_y_i_k)
    nmi_posterior = normalized_mutual_info_score(ground_truth_labels_numpy, predicted_labels_posterior)
    ari_posterior= adjusted_rand_score(ground_truth_labels_numpy, predicted_labels_posterior)
    NMI_Posterior.append(nmi_posterior)
    ARI_Posterior.append(ari_posterior)
    new_accuracy_posterior, new_predicted_labels_posterior = cluster_acc_with_assignment(ground_truth_labels_numpy, predicted_labels_posterior)
    Accuracy_Posterior.append(new_accuracy_posterior)
    Updated_Posterior_Labels.append(new_predicted_labels_posterior)

    ##################Clustering using Gaussian Mixture Model###################

    for i in range(num_iterations):
      # seed = np.random.randint(0, 100)
      gmm = GaussianMixture(n_components=num_components, covariance_type=covariance_type, random_state=i)
      predicted_labels_gmm = gmm.fit_predict(latent_space)
      posterior_probabilities_gmm = gmm.predict_proba(latent_space)
      simulated_posterior_probabilities_gmm.append(posterior_probabilities_gmm)
      nmi_gmm = normalized_mutual_info_score(ground_truth_labels_numpy, predicted_labels_gmm)
      ari_gmm = adjusted_rand_score(ground_truth_labels_numpy, predicted_labels_gmm)
      new_accuracy_gmm, new_predicted_labels_gmm = cluster_acc_with_assignment(ground_truth_labels_numpy,
                                                                                predicted_labels_gmm)
      simulated_gmm_accuracies.append(new_accuracy_gmm)
      simulated_gmm_nmi.append(nmi_gmm)
      simulated_gmm_ari.append(ari_gmm)
      simulated_predicted_labels_gmm.append(new_predicted_labels_gmm)

    average_gmm_accuracy = np.mean(simulated_gmm_accuracies)
    max_gmm_accuracy = np.max(simulated_gmm_accuracies)
    sd_gmm_accuracy = np.std(simulated_gmm_accuracies)
    Accuracy_GMM.append(max_gmm_accuracy)
    Mean_Accuracy_GMM.append(average_gmm_accuracy)
    STD_Accuracy_GMM.append(sd_gmm_accuracy)
    best_accuracy_index = simulated_gmm_accuracies.index(np.max(simulated_gmm_accuracies))
    Updated_GMM_Labels.append(simulated_predicted_labels_gmm[best_accuracy_index])
    Posterior_EM.append(simulated_posterior_probabilities_gmm[best_accuracy_index])
    NMI_GMM.append(simulated_gmm_nmi[best_accuracy_index])
    ARI_GMM.append(simulated_gmm_ari[best_accuracy_index])
    ####################################################################################################################

    # Reconstructed Digits

    if (epoch + 1) == num_epochs:
        reconstructions = torch.cat(reconstructions, dim=0)
        True_labels = np.array(True_Labels[0])
        Predicted_labels = np.array(Updated_GMM_Labels[0])
        Selected_Image = []

        for digits in range(num_components):
            Matching_index = np.where((True_labels == digits) & (Predicted_labels == digits))[0]
            Selected_Reconstruction = reconstructions[Matching_index]
            Selected_Reconstruction = Selected_Reconstruction[:10]
            Selected_Image.append(Selected_Reconstruction)

        Selected_Image = torch.cat(Selected_Image, dim=0)
        Reconstructed_Digits = Selected_Image.view(Selected_Image.size(0), 1, pixel_size[0], pixel_size[1])
        Reconstructed_Digits = Reconstructed_Digits.cpu()
        Reconstructed_Digits = Reconstructed_Digits.clamp(0, 1)
        Reconstructed_Digits = Reconstructed_Digits[: 100]
        Reconstructed_Digits = make_grid(Reconstructed_Digits, 10, 10)
        Reconstructed_Digits = Reconstructed_Digits.numpy()
        Reconstructed_Digits = np.transpose(Reconstructed_Digits, (1, 2, 0))
        plt.imshow(Reconstructed_Digits)
        plt.axis('off')
        plt.savefig(f'Reconstructed_Digits_{Model_name}_{Dataset_name}.png')
        plt.close()

        # Mismatch Digits

        Selected_Mismatch_Image = []

        for digits in range(num_components):
            Mismatching_index = np.where((True_labels == digits) & (Predicted_labels != digits))[0]
            Selected_Reconstruction = reconstructions[Mismatching_index]
            Selected_Reconstruction = Selected_Reconstruction[:10]
            Selected_Mismatch_Image.append(Selected_Reconstruction)

        Selected_Image = torch.cat(Selected_Mismatch_Image, dim=0)
        Reconstructed_Digits = Selected_Image.view(Selected_Image.size(0), 1, pixel_size[0], pixel_size[1])
        Reconstructed_Digits = Reconstructed_Digits.cpu()
        Reconstructed_Digits = Reconstructed_Digits.clamp(0, 1)
        Reconstructed_Digits = Reconstructed_Digits[: 100]
        Reconstructed_Digits = make_grid(Reconstructed_Digits, 10, 10)
        Reconstructed_Digits = Reconstructed_Digits.numpy()
        Reconstructed_Digits = np.transpose(Reconstructed_Digits, (1, 2, 0))
        plt.imshow(Reconstructed_Digits)
        plt.axis('off')
        plt.savefig(f'Reconstructed_Mismatch_Digits_{Model_name}_{Dataset_name}.png')
        plt.close()

        # Original Images (Random)
        original_test_image = torch.cat(original_image, dim=0)
        test_images = original_test_image.view(original_test_image.size(0), 1, pixel_size[0], pixel_size[1])
        test_images = test_images.cpu()
        test_images = test_images.clamp(0, 1)
        test_images = test_images[:50]
        test_images = make_grid(test_images, 10, 5)
        test_images = test_images.numpy()
        test_images = np.transpose(test_images, (1, 2, 0))
        plt.imshow(test_images)
        plt.axis('off')
        plt.savefig(f'Original_Random_Images_{Model_name}_{Dataset_name}.png')
        plt.close()

        # Reconstructed Images (Random)

        Reconstructions = reconstructions.view(reconstructions.size(0), 1, pixel_size[0], pixel_size[1])
        Reconstructions = Reconstructions.cpu()
        Reconstructions = Reconstructions.clamp(0, 1)
        Reconstructions = Reconstructions[: 50]
        Reconstructions = make_grid(Reconstructions, 10, 5)
        Reconstructions = Reconstructions.numpy()
        Reconstructions = np.transpose(Reconstructions, (1, 2, 0))
        plt.imshow(Reconstructions)
        plt.axis('off')
        plt.savefig(f'Reconstructed_Random_Images_{Model_name}_{Dataset_name}.png')
        plt.close()

    if (epoch+1)%5==0:
        print(f"\n Step: [{epoch+1}/{num_epochs}] \t  Training Loss: {Training_Losses[-1]: 0.2f} \t Test Loss: {Final_Test_Loss: 0.2f} \
        \t Accuracy (GMM): {max_gmm_accuracy: 0.2f} \t Accuracy (Posterior): {new_accuracy_posterior: 0.2f} \t NMI (GMM): {simulated_gmm_nmi[best_accuracy_index]: 0.2f} \t NMI (Posterior): {nmi_posterior: 0.2f} \
        \t LR: {scheduler.get_last_lr()[0]:0.6f}")

"""### **Model Parameters**"""

# print(f"Model Parameters: \n\n", model.parameters)
param_list = [*model.parameters()]

"""# **Results**

### **Clustering Accuracy**
"""

print("\n Loss:")
print("-"*50)
print("\n Final Training Loss:", Training_Losses[-1])
print("\n Final Test Loss:", Test_Loss[-1])
print("-"*50)
print("\n Posterior Probability Results:")
print(f"\n Accuracy (Posterior): {Accuracy_Posterior[-1]: 0.3f}")
print(f'\n NMI (Posterior): {NMI_Posterior[-1]: 0.2f}')
print(f'\n ARI (Posterior): {ARI_Posterior[-1]: 0.2f}')
print("-"*50)

print("-"*50)
print("\n GMM Results:")
print(f"\n Clustering accuracy (GMM): {Accuracy_GMM[-1]: 0.3f}")
print(f'\n NMI (GMM): {NMI_GMM[-1]: 0.2f}')
print(f'\n ARI (GMM): {ARI_GMM[-1]: 0.2f}')
print("-"*50)

overall_avg_accuracy_posterior = np.mean(Accuracy_Posterior)
overall_std_accuracy_posterior = np.std(Accuracy_Posterior)

print("Overall Mean Accuracy (Posterior Probability):", overall_avg_accuracy_posterior, "\n")
print("Overall STD Accuracy (Posterior Probability):", overall_std_accuracy_posterior, "\n")

"""### **True and Predicted Labels**"""

print(f"\n True Labels Frequency:\n\n {Counter(True_Labels[-1])}")
print(f"\n Posterior Labels Frequency:\n\n {Counter(Updated_Posterior_Labels[-1])}")
print(f"\n GMM Labels Frequency:\n{Counter(Updated_GMM_Labels[-1])}")

"""# **Graphs**

"""### **Training Losses and Test Losses**"""

plt.figure(figsize=(8, 6))
x=np.linspace(1, len(Training_Losses), len(Training_Losses))
y=np.linspace(1, len(Test_Loss), len(Test_Loss))
plt.plot(x, Training_Losses, label='Training Losses', color="blue", linestyle="solid", linewidth=1)
plt.plot(y, Test_Loss, label='Test Losses', color="green", linestyle="solid", linewidth=1)
plt.xlabel('Epochs')
plt.ylabel('Negative ELBO')
# plt.title('Training and Test Losses')
plt.legend(loc="best")
plt.grid()
plt.savefig(f"Training_Test_Loss_{Model_name}_{Dataset_name}.pdf", format="pdf", bbox_inches="tight")
plt.close()

"""### **Negative ELBO (Training Loss)**"""

x=np.linspace(1, len(Training_Losses), len(Training_Losses))
plt.figure(figsize=(8, 6))
plt.xlabel("Epochs")
plt.ylabel("Negative ELBO (Training Losses)")
plt.plot(x, Training_Losses, color="blue", linestyle="solid", linewidth=1)
plt.grid()
plt.savefig(f"Negative_ELBO_Training_Loss_{Model_name}_{Dataset_name}.pdf", format="pdf", bbox_inches="tight")
plt.close()

"""### **Negative ELBO (Test Loss)**"""

x=np.linspace(1, len(Test_Loss), len(Test_Loss))
plt.figure(figsize=(8, 6))
plt.xlabel("Epochs")
plt.ylabel("Negative ELBO (Test Loss)")
plt.plot(x, Test_Loss, color="blue", linestyle="solid", linewidth=1)
plt.grid()
plt.savefig(f"Negative_ELBO_Test_Loss_{Model_name}_{Dataset_name}.pdf", format="pdf", bbox_inches="tight")
plt.close()

"""### **Accuracy Plot**"""

plt.figure(figsize=(8, 6))
x=np.linspace(1, len(Accuracy_GMM), len(Accuracy_GMM))
y=np.linspace(1, len(Accuracy_Posterior), len(Accuracy_Posterior))
plt.plot(x, Accuracy_GMM, label='Accuracy (GMM)', color="blue", linestyle="solid", linewidth=1)
plt.plot(y, Accuracy_Posterior, label='Accuracy (Posterior)', color="green", linestyle="solid", linewidth=1)
plt.xlabel('Epochs')
plt.ylabel('Clustering Accuracy')
plt.legend(loc="best")
plt.grid()
plt.savefig(f"Accuracy_{Model_name}_{Dataset_name}_GMM_Posterior.pdf", format="pdf", bbox_inches="tight")
plt.close()

"""### **Accuracy (Posterior)**"""

x=np.linspace(1, len(Accuracy_Posterior), len(Accuracy_Posterior))
plt.figure(figsize=(8, 6))
plt.xlabel("Epochs")
plt.ylabel('Clustering Accuracy')
plt.plot(x, Accuracy_Posterior, color="blue", linestyle="solid", linewidth=1)
plt.grid()
plt.savefig(f"Accuracy_{Model_name}_{Dataset_name}_Posterior.pdf", format="pdf", bbox_inches="tight")
plt.close()

"""### **Accuracy (GMM)**"""

x=np.linspace(1, len(Accuracy_GMM), len(Accuracy_GMM))
plt.figure(figsize=(8, 6))
plt.xlabel("Epochs")
plt.ylabel('Clustering Accuracy')
plt.plot(x, Accuracy_GMM, color="blue", linestyle="solid", linewidth=1)
plt.grid()
plt.savefig(f"Accuracy_{Model_name}_{Dataset_name}_GMM.pdf", format="pdf", bbox_inches="tight")
plt.close()

"""# **Confusion Matrix and t-SNE**

### **Confusion Matrix (GMM)**
"""

conf_matrix = confusion_matrix(True_Labels[-1], Updated_GMM_Labels[-1])
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
# plt.title("Confusion Matrix")
plt.savefig(f"Confusion_Matrix_{Model_name}_{Dataset_name}_GMM.pdf", format="pdf", bbox_inches="tight")
plt.close()

"""### **Confusion Matrix (Posterior)**"""

conf_matrix = confusion_matrix(True_Labels[-1], Updated_Posterior_Labels[-1])
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
# plt.title("Confusion Matrix")
plt.savefig(f"Confusion_Matrix_{Model_name}_{Dataset_name}_Posterior.pdf", format="pdf", bbox_inches="tight")
plt.close()

"""### **t-SNE Visualization (GMM)**"""

latent_space_recoded=torch.tensor(Latent_embeddings[-1], dtype=torch.float32)
# tsne = TSNE(n_components=2, perplexity=5, random_state=random_seed)
tsne = TSNE(n_components=2, random_state=random_seed)
latent_tsne = tsne.fit_transform(latent_space_recoded)
plt.figure(figsize=(10, 10))
cmap = plt.cm.get_cmap('rainbow', len(np.unique(Updated_GMM_Labels[-1])))

scatter = plt.scatter(latent_tsne[:, 0], latent_tsne[:, 1], c=Updated_GMM_Labels[-1], cmap='rainbow')
plt.colorbar(scatter)
# plt.title('Visualization of Latent Space with Clusters')
plt.xlabel('Latent Embeddings (1)')
plt.ylabel('Latent Embeddings (2)')
# plt.grid(True)
# Create legend handles for each cluster
legend_handles = [mpatches.Patch(color=cmap(i), label=f'Cluster {i}') for i in np.unique(Updated_GMM_Labels[-1])]
plt.legend(handles=legend_handles, loc='best')

plt.savefig(f"t-SNE_Visualization_{Model_name}_{Dataset_name}_GMM.pdf", format="pdf", bbox_inches="tight")
plt.close()

"""### **t-SNE Visualization (Posterior)**"""

latent_space_recoded=torch.tensor(Latent_embeddings[-1], dtype=torch.float32)
# tsne = TSNE(n_components=2, perplexity=5, random_state=random_seed)
tsne = TSNE(n_components=2, random_state=random_seed)
latent_tsne = tsne.fit_transform(latent_space_recoded)
plt.figure(figsize=(10, 8))
cmap = plt.cm.get_cmap('rainbow', len(np.unique(Updated_Posterior_Labels[-1])))
scatter = plt.scatter(latent_tsne[:, 0], latent_tsne[:, 1], c=Updated_Posterior_Labels[-1], cmap='rainbow')
plt.colorbar(scatter)
# plt.title('Visualization of Latent Space with Clusters')
plt.xlabel('Latent Embeddings (1)')
plt.ylabel('Latent Embeddings (2)')
# plt.grid(True)
legend_handles = [mpatches.Patch(color=cmap(i), label=f'Cluster {i}') for i in np.unique(Updated_Posterior_Labels[-1])]
plt.legend(handles=legend_handles, loc='best')
plt.savefig(f"t-SNE_Visualization_{Model_name}_{Dataset_name}_Posterior.pdf", format="pdf", bbox_inches="tight")
plt.close()

"""## **Compare Reconstructions and Original Images**

### **Original Images**
"""

with torch.no_grad():
  print("Original Image")
  test_images=test_features.view(test_features.size(0), 1, pixel_size[0], pixel_size[1])
  test_images=test_images.cpu()
  test_images=test_images.clamp(0,1)
  test_images=test_images[:50]
  test_images=make_grid(test_images, 10, 5)
  test_images=test_images.numpy()
  test_images=np.transpose(test_images, (1, 2, 0))
  plt.imshow(test_images)
  plt.axis('off')
  plt.savefig(f'Original_Images_{Model_name}_{Dataset_name}.png')
  plt.close()

"""### **Reconstructed Images**"""

with torch.no_grad():
  print("Reconstructed Image")
  reconstructions=decoder_mu_test.view(decoder_mu_test.size(0), 1, pixel_size[0], pixel_size[1])
  reconstructions=reconstructions.cpu()
  reconstructions=reconstructions.clamp(0,1)
  reconstructions= reconstructions[:50]
  reconstructions=make_grid(reconstructions, 10, 5)
  reconstructions=reconstructions.numpy()
  reconstructions=np.transpose(reconstructions, (1,2,0))
  plt.imshow(reconstructions)
  plt.axis('off')
  plt.savefig(f'Reconstructed_Images_{Model_name}_{Dataset_name}.png')
  plt.close()

"""### **Data Generation (Sampling Latent Space)**"""




torch.manual_seed(random_seed)
model.eval()
with torch.no_grad():
  print("Generated Image")
  mu = param_list[-3]
  logvar = param_list[-2]
  sd = torch.exp(0.5 * logvar)
  z = torch.normal(mu.repeat(10, 1), sd.repeat(10, 1)).to(device=device)
  # z=torch.normal(-0.5, 1, size=(100,latent_dim), device=device)
  new_image=model.decoder(z)
  new_image=new_image.view(new_image.size(0), 1, pixel_size[0], pixel_size[1])
  new_image=new_image.cpu()
  new_image=new_image.clamp(0, 1)
  new_image=new_image[: 100]
  new_image=make_grid(new_image, 10, 10)
  new_image=new_image.numpy()
  new_image=np.transpose(new_image,(1, 2, 0))
  plt.imshow(new_image)
  plt.axis('off')
  plt.savefig(f'Generated_Image_{Model_name}_{Dataset_name}_Updated.png')
  plt.close()

print(f"Experiment Completed on Model: {Model_name} and {Dataset_name} Dataset")