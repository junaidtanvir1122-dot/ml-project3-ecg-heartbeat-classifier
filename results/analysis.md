# Analysis

**6.1 Always-"N" baseline.** A model that always answers "N" scores 18,118 / 21,892 = 82.76% on the
test set. Our network scores 97.77%, which is better, but the gap between overall accuracy (97.77%)
and balanced accuracy (84.03%) shows that accuracy alone hides how poorly the rare classes S (66.7%)
and F (64.8%) are recognised. When one class makes up 83% of the data, accuracy mostly measures how
well that one class is recognised, so per-class and balanced accuracy must be reported as well.

**6.2 Weakest class.** F has the lowest per-class accuracy (64.81%). Of its 57 misclassified beats,
38 (67%) were predicted as N and 18 as V. A fusion beat is by definition a blend of a normal and a
ventricular beat, so its shape lies between the two and overlaps with them. In addition, F has only
641 training rows (0.7%), so the network saw very few examples of it.

**6.3 Loss curves.** At the best epoch (23) the training loss is 0.0556 and the validation loss is
0.0717, a small gap of about 0.016. After that the training loss keeps falling (0.0492 at epoch 30)
while the validation loss never gets below 0.0717 (0.0721 at epoch 30), so the gap widens to about
0.023. The network is therefore overfitting slightly after epoch 23, but only mildly. It is not
underfitting, since both losses are low and accuracy is about 98%.

**6.4 Initialisation.** With all-zero weights every neuron in a layer computes the same output and
receives the same gradient, so they stay identical forever (symmetry) and the layer acts like a single
neuron; random weights break the symmetry. For a ReLU layer, Var(z) = n * Var(w) * E[a^2], and ReLU
zeroes the negative half of the previous layer's outputs, so E[a^2] = Var(z_prev) / 2. Keeping the
variance the same from layer to layer needs n * Var(w) / 2 = 1, i.e. Var(w) = 2/n and a standard
deviation of sqrt(2/n). Otherwise the signal shrinks or blows up as it passes through the layers.

**6.5 Momentum and weight decay.** v = beta*v + (1-beta)*g is a moving average of the gradients. If two
consecutive gradients point in opposite directions they partly cancel inside v, so the step is small
and the zig-zag is damped (gradients in the same direction add up and speed the update). A weight that
receives no gradient still gets g = lambda * P, so it shrinks towards zero by a factor of about
(1 - alpha*lambda) per update. With alpha = 0.5, lambda = 1e-4 and 18,480 updates this is roughly
e^(-0.92), about 0.4 of its starting value.

**6.6 Rare classes.** Use a class-weighted loss: multiply each row's loss by a weight inversely
proportional to its class frequency (for example n_total / (5 * n_class), about 27 for F and 0.24
for N). A mistake on a rare F beat then costs as much as many mistakes on N, so the gradient no longer
ignores F. The trade-off is that the network will raise more false alarms on N beats.