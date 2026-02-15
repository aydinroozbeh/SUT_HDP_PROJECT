import numpy as np
import config

def generate_noise(noise_type, shape):
    """
    Factory function to generate zero-mean noise vectors with specific tail properties.
    
    Args:
        noise_type (str): 'pareto', 'cauchy', 'levy_stable', 'gaussian'
        shape (tuple): Shape of the gradient vector (e.g., (10,))
    """
    scale = config.NOISE_SCALE
    
    if noise_type == 'pareto':
        return _generate_symmetric_pareto(shape, config.TAIL_INDEX, scale)
    
    elif noise_type == 'cauchy':
        # Cauchy has undefined mean/variance (Tail index = 1)
        return np.random.standard_cauchy(shape) * scale
    
    elif noise_type == 'gaussian':
        # Baseline for comparison (Light tail)
        return np.random.normal(0, scale, shape)
    
    elif noise_type == 'levy_stable':
        # Generating true Levy-Stable distributions usually requires scipy.
        # We use a robust approximation using the Chambers-Mallows-Stuck method
        # if scipy is not available, or simply wrap scipy.
        return _generate_levy_stable(shape, config.TAIL_INDEX, scale)
    
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")

def _generate_symmetric_pareto(shape, alpha, scale):
    """
    Generates Symmetric Pareto noise.
    
    Paper Context: Used in numerical experiments (Section IV).
    math: p(x) ~ |x|^{-(alpha+1)}
    
    Logic:
    1. Generate Standard Pareto: X ~ Pareto(alpha)  (Values >= 0)
    2. Make it Symmetric: Multiply by random sign {-1, 1}
    3. Result has Expectation E[xi] = 0 (Satisfying Assumption 6)
    """
    # np.random.pareto(a) returns x such that distribution is (1+x)^(-a-1)
    # We add 1 to avoid zeros if necessary, but standard usage is fine.
    pareto_samples = np.random.pareto(alpha, shape)
    
    # Generate random signs [-1, 1]
    signs = np.random.choice([-1, 1], size=shape)
    
    noise = signs * pareto_samples * scale
    return noise

def _generate_levy_stable(shape, alpha, scale):
    """
    Generates alpha-stable noise using Scipy if available, 
    otherwise falls back to Symmetric Pareto (which is in the domain of attraction).
    """
    try:
        from scipy.stats import levy_stable
        # levy_stable.rvs(alpha, beta, loc, scale)
        # alpha: Tail index (0 < alpha <= 2)
        # beta: Skewness (0 for symmetric)
        # loc: Mean (0)
        return levy_stable.rvs(alpha, 0, loc=0, scale=scale, size=shape)
    except ImportError:
        print("Warning: Scipy not found. Falling back to Symmetric Pareto for Levy noise.")
        return _generate_symmetric_pareto(shape, alpha, scale)