import numpy as np

# Numerical stability constant
EPSILON = 1e-8

def get_clipping_function(method_name):
    """
    Factory function to return the requested clipping method.
    """
    if method_name == 'standard_dynamic':
        return standard_dynamic_clip
    elif method_name == 'global_adaptive':
        return global_adaptive_clip
    elif method_name == 'soft_biclip':
        return soft_biclip
    else:
        raise ValueError(f"Unknown clipping method: {method_name}")

# ==============================================================================
# 1. BASELINE: Standard Dynamic Norm Clipping
# ==============================================================================
def standard_dynamic_clip(g, threshold, state=None):
    """
    Implements the standard clipping from 'Distributed Stochastic Optimization 
    under Heavy-Tailed Noises' (Eq 3).
    
    Logic:
        scale = min(1, tau / ||g||_2)
        g_hat = g * scale
        
    Args:
        g (np.array): Input gradient vector.
        threshold (float): The clipping threshold 'tau_k'.
        state (dict): Unused here, kept for API consistency.
        
    Returns:
        (np.array, dict): Clipped gradient and unchanged state.
    """
    # L2 Norm of the entire gradient vector
    norm_g = np.linalg.norm(g)
    
    # Paper Eq (3): Scaling factor
    scale = min(1.0, threshold / (norm_g + EPSILON))
    
    clipped_g = g * scale
    return clipped_g, state

# ==============================================================================
# 2. INNOVATION 1: Global-Adaptive Coordinate-wise Clipping (G-ACC)
# ==============================================================================
def global_adaptive_clip(g, threshold, state=None):
    """
    Implements Adaptive Coordinate-wise Clipping.
    
    Logic:
        1. Track element-wise variance (v_t) using EMA (like Adam).
        2. Normalize gradient by variance: g_norm = g / (sqrt(v_t) + epsilon).
        3. Clip the *normalized* coordinate to [-threshold, threshold].
        
    Args:
        state (dict): Must contain 'v_t' (variance estimate).
    """
    # Initialize state if first iteration
    if state is None or 'v_t' not in state:
        state = {'v_t': np.zeros_like(g)}
    
    # Hyperparameters for adaptivity (can be moved to config)
    BETA = 0.99  # EMA decay rate
    
    # 1. Update Variance Estimate (EMA of squared gradients)
    # v_t = beta * v_{t-1} + (1-beta) * g^2
    v_t = BETA * state['v_t'] + (1 - BETA) * np.square(g)
    
    # 2. Compute Adaptive Denominator
    denom = np.sqrt(v_t) + EPSILON
    
    # 3. Coordinate-wise Clipping
    # We clip the "preconditioned" gradient g/denom to range [-tau, tau]
    # Then rescale back? No, usually adaptive clipping replaces the gradient.
    # Proposal: "Client clips local gradient element-wise" using v_t.
    
    # Logic: If a coordinate is 100x variance, clip it.
    # We use element-wise min/max logic similar to standard clip but per coordinate
    
    # Effective threshold per coordinate
    # If variance is high, we allow larger gradients. If low, we clip strictly.
    # Or strict AdaGC style: clip(g / denom) * denom is often redundant.
    # Let's use: g_hat = clip(g, -tau*denom, tau*denom)
    
    upper_bound = threshold * denom
    lower_bound = -threshold * denom
    
    clipped_g = np.clip(g, lower_bound, upper_bound)
    
    # Update state
    state['v_t'] = v_t
    
    return clipped_g, state

# ==============================================================================
# 3. INNOVATION 2: Soft-BiClip (Smooth Garden Shaping)
# ==============================================================================
def soft_biclip(g, threshold, state=None):
    """
    Implements 'Soft-BiClip' - a differentiable shaping function.
    
    Logic:
        1. Heavy-Tail Suppression (Upper): Tanh compression.
           Instead of hard clip at tau, we use tau * tanh(g/tau).
           This is linear near 0 and saturates smoothly at tau.
           
        2. Noise Gating (Lower): Sigmoid Soft-Masking.
           Suppress gradients small than 'lower_threshold'.
           
    Args:
        threshold (float): Acts as the saturation point (tau).
    """
    # --- Part A: Smooth Upper Clipping (Tanh) ---
    # As g -> infinity, tanh(g/tau) -> 1, so output -> tau.
    # As g -> 0, tanh(g/tau) -> g/tau, so output -> g (linear).
    upper_shaped = threshold * np.tanh(g / (threshold + EPSILON))
    
    # --- Part B: Smooth Lower Gating (Soft Mask) ---
    # We want to kill signals where |g| < lower_threshold.
    # In BiClip, lower_threshold is often a fraction of upper (e.g., 0.1 * tau).
    lower_threshold = 0.1 * threshold 
    
    # Sigmoid-like gate: 0 if |g| << lower, 1 if |g| >> lower
    # Formula: 1 / (1 + exp(-k * (|g| - lower)))
    # k determines sharpness.
    k_sharpness = 10.0
    magnitude = np.abs(g)
    
    gate = 1.0 / (1.0 + np.exp(-k_sharpness * (magnitude - lower_threshold)))
    
    # Apply both
    final_g = upper_shaped * gate
    
    return final_g, state