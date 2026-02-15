import numpy as np

# --- 1. General Experiment Settings ---
SEED = 42
NUM_AGENTS = 30           # N agents
DIMENSION = 10            # Dimension of the optimization variable x
MAX_ITERATIONS = 5000     # T iterations
TOPOLOGY = 'ring'         # 'ring', 'fully_connected', 'random'

# --- 2. Constraint Set (Projection) ---
# The paper assumes x stays within a bounded set Omega
CONSTRAINT_RADIUS = 10.0  # Radius R for Euclidean ball projection

# --- 3. Optimization Schedules (CRITICAL) ---
# Based on Paper Eq (2-3) and Theorem 1 conditions:
# Alpha (step size) must decay: alpha_k = c / (k+1)^a
# Tau (clipping) must grow:     tau_k   = d * (k+1)^b

# Parameter 'a' (Alpha Decay): Must be 0.5 < a <= 1
# Parameter 'b' (Tau Growth):  Must satisfy 0 <= b < a - 0.5
ALPHA_BASE = 0.1          # c
ALPHA_EXP = 0.9           # a (0.9 satisfies the condition)

TAU_BASE = 5.0            # d
TAU_EXP = 0.3             # b (0.3 < 0.9 - 0.5, satisfies the condition)


# ayroz: these functions are currently designed for the original paper, 
# ayroz: I would probably change them into modules later
def get_step_size(k):
    """Calculates alpha_k for iteration k."""
    return ALPHA_BASE / ((k + 1) ** ALPHA_EXP)

def get_clipping_thresh(k):
    """Calculates tau_k for iteration k."""
    return TAU_BASE * ((k + 1) ** TAU_EXP)

# --- 4. Dataset & Heavy-Tail Noise Settings ---
# Switchable Datasets: 'synthetic_regression', 'mnist', 'cifar10'
DATASET_NAME = 'synthetic_regression'

# Switchable Noise Types: 'pareto', 'levy_stable', 'cauchy', 'log_normal'
NOISE_TYPE = 'pareto'

# Heavy Tail Index (delta/alpha):
# 1 < TAIL_INDEX <= 2 means Infinite Variance (Heavy Tail)
# Standard Gaussian corresponds to TAIL_INDEX = 2 (conceptually)
TAIL_INDEX = 1.5   
NOISE_SCALE = 1.0  # Scale parameter for the noise distribution

# --- 5. Algorithm Method Switch ---
# Switch between the baseline and your proposed innovations
# Options: 
#   'standard_dynamic'   (The baseline from the paper: Eq 3)
#   'global_adaptive'    (Your Innovation 1: Federated variance tracking)
#   'soft_biclip'        (Your Innovation 2: Smooth garden shaping)
CLIPPING_METHOD = 'standard_dynamic'