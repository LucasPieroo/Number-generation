import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
import random
import matplotlib.pyplot as plt
from io import BytesIO

# Load MNIST data (cache to avoid reloading)
@st.cache_data
def load_mnist():
    mnist = fetch_openml("mnist_784", version=1, as_frame=True)
    x, y = mnist['data'], mnist['target']
    return x, y

x, y = load_mnist()

st.title("MNIST Digit Explorer - Just select the digit you want to see!")
st.text("I was not able to train a model, but I tried at least to create the interface to explore the MNIST dataset.")
# Dropdown for digit selection
selected_digit = st.selectbox("Select a digit (0-9):", list(map(str, range(10))))

# Filter indices for selected digit
indices = np.where(y == selected_digit)[0]

# Randomly select 5 indices
if len(indices) >= 5:
    chosen_indices = random.sample(list(indices), 5)
else:
    chosen_indices = indices[:5]

# Display 5 images in 5 boxes (28x28)
cols = st.columns(5)
for i, idx in enumerate(chosen_indices):
    digit_image = x.iloc[idx].to_numpy().reshape(28, 28)
    fig, ax = plt.subplots(figsize=(2, 2))
    ax.imshow(digit_image, cmap='binary', interpolation='nearest')
    ax.axis('off')
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    cols[i].image(buf.getvalue(), use_container_width =True)