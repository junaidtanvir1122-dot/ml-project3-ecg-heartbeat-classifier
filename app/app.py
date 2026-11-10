import sys; sys.path.insert(0, "src")                       # src/ ke modules import karne ke liye
import numpy as np, pandas as pd, torch, streamlit as st
import matplotlib.pyplot as plt
from model import ECGNet

NAMES = ["N: normal", "S: supraventricular ectopic", "V: ventricular ectopic",
         "F: fusion", "Q: unknown / paced"]

@st.cache_resource                                          # model sirf ek dafa load ho
def load_model():
    m = ECGNet(); m.load_state_dict(torch.load("results/best_model.pt")); return m.eval()

@st.cache_data                                              # test CSV sirf ek dafa load ho
def load_test():
    return pd.read_csv("data/mitbih_test.csv", header=None)

st.title("ECG Heartbeat Classifier")
model, test = load_model(), load_test()
mode = st.radio("Input", ["Test-set row", "Upload CSV (one row, 187 values)"])
true = None
if mode == "Test-set row":
    i = st.number_input("Row number", 0, len(test) - 1, 0)
    x, true = test.iloc[i, :187].values.astype(np.float32), int(test.iloc[i, 187])
else:
    f = st.file_uploader("CSV file", type="csv")
    if f is None: st.stop()                                 # file aane tak aage mat chalo
    v = pd.read_csv(f, header=None).values.flatten()
    if len(v) != 187: st.error(f"Expected 187 values, got {len(v)}"); st.stop()
    x = v.astype(np.float32)

with torch.no_grad():
    p = torch.softmax(model(torch.tensor(x)[None]), dim=1)[0].numpy()   # 5 probabilities

fig, ax = plt.subplots(figsize=(8, 2.5))
ax.plot(np.arange(187) * 8, x)                              # har sample 8 ms (125 Hz)
ax.set_xlabel("time (ms)"); ax.set_ylabel("amplitude"); st.pyplot(fig)

st.subheader(f"Predicted: {NAMES[p.argmax()]}  ({p.max():.1%})")
if true is not None: st.write(f"True class: {NAMES[true]}")
fig2, ax2 = plt.subplots(figsize=(8, 2.5))
ax2.bar(["N", "S", "V", "F", "Q"], p); ax2.set_ylim(0, 1); ax2.set_ylabel("probability")
st.pyplot(fig2)