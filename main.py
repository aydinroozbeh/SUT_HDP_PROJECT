import numpy as np
import matplotlib.pyplot as plt

# Import custom modules
import config
import clipping_methods
import dataset_loader
import noise_generators

def init_network(num_agents, topology_type='ring'):
    """
    Creates the doubly stochastic communication matrix A.
    Paper Requirement: Matrix A must be doubly stochastic.
    """
    if topology_type == 'ring':
        # Simple ring topology: 1/3 self, 1/3 left, 1/3 right
        A = np.zeros((num_agents, num_agents))
        for i in range(num_agents):
            A[i, i] = 1/3
            A[i, (i-1)%num_agents] = 1/3
            A[i, (i+1)%num_agents] = 1/3
        return A
    # ayroz: Placeholder for other topologies: Fully connected for Federated Learning
    return np.eye(num_agents)

def projection(x, radius):
    """
    Projects x onto the constraint set Omega (Euclidean ball).
    Paper Requirement: P_Omega operator.
    """
    norm = np.linalg.norm(x)
    if norm > radius:
        return x * (radius / norm) # ayroz: scaling back to the unit ball/cube
    return x

def main():
    # --- 1. Setup & Initialization ---
    print(f"Initializing Network: {config.NUM_AGENTS} agents, Topology: {config.TOPOLOGY}")
    
    # Initialize Communication Matrix A
    A = init_network(config.NUM_AGENTS, config.TOPOLOGY)
    
    # Initialize Models (x_i,0)
    # Shape: (Num_Agents, Dimension)
    X = np.zeros((config.NUM_AGENTS, config.DIMENSION)) 
    
    # Initialize Agent States (New!)
    # This stores history like 'v_t' for Adaptive Clipping or 'm_t' for Momentum
    agent_states = [{} for _ in range(config.NUM_AGENTS)]
    
    # Load Data Source
    # Returns a function: get_gradient(agent_id, params)
    get_gradient_batch = dataset_loader.load_data(config.DATASET_NAME)

    # Load Loss Function
    get_loss_func = dataset_loader.load_loss_func(config.DATASET_NAME)

    # Load Clipping Method
    # Returns a function: clip(g, threshold, state) -> (g_new, state_new)
    clip_gradient = clipping_methods.get_clipping_function(config.CLIPPING_METHOD)
    
    # Metrics tracking
    loss_history = []
    

    # SHAYAN: add SLSQP for finding the actuall optimum (without noise)
    # SHAYAN: add figure for log10( (f(y_k) - f(y*)) / (f(y_0) - f(y*)) )

    print(f"Starting Optimization Loop using method: {config.CLIPPING_METHOD}...")

    # --- 2. Main Optimization Loop ---
    for k in range(config.MAX_ITERATIONS):
        
        # --- Parameter Scheduling ---
        # Paper Requirement: alpha decreases, tau increases
        alpha_k = config.get_step_size(k)      # Learning rate
        tau_k = config.get_clipping_thresh(k)  # Clipping threshold
        
        # --- Step A: Consensus (Mixing) ---
        # Eq (3): v_{i,k} = Sum( A_ij * x_{j,k} )
        V = A @ X 
        
        # --- Step B: Stochastic Gradient Generation ---
        raw_gradients = np.zeros_like(V)
        
        for i in range(config.NUM_AGENTS):
            # 1. Get True Gradient at v_{i,k}
            true_grad = get_gradient_batch(agent_id=i, params=V[i])
            
            # 2. Generate Heavy-Tailed Noise (Pareto/Levy)
            noise = noise_generators.generate_noise(config.NOISE_TYPE, shape=true_grad.shape)
            
            # 3. Add Noise
            raw_gradients[i] = true_grad + noise

        # --- Step C: Robust Clipping (The Innovation) ---
        # Eq (3): g_hat = Clip(g, tau)
        # We pass the persistent state to allow adaptive methods to track variance
        clipped_gradients = np.zeros_like(raw_gradients)
        
        for i in range(config.NUM_AGENTS):
            # Apply clipping and update the agent's internal state (e.g., v_t)
            clipped_g, new_state = clip_gradient(
                g=raw_gradients[i], 
                threshold=tau_k, 
                state=agent_states[i]
            )
            clipped_gradients[i] = clipped_g
            agent_states[i] = new_state

        # --- Step D: Descent & Projection ---
        # Eq (2): x_{i,k+1} = Projection( v_{i,k} - alpha_k * g_hat )
        X_next = np.zeros_like(X)
        for i in range(config.NUM_AGENTS):
            update = V[i] - alpha_k * clipped_gradients[i]
            X_next[i] = projection(update, config.CONSTRAINT_RADIUS)
            
        X = X_next

        # --- Monitoring ---
        if k % 100 == 0:
            # 1. Calculate Global Loss (Average of all agents)
            total_loss = 0.0
            for i in range(config.NUM_AGENTS):
                total_loss += get_loss_func(agent_id=i, params=X[i])
            avg_loss = total_loss / config.NUM_AGENTS
            
            loss_history.append(avg_loss)
            
            # Optional: Still track param norm if you want debug info
            avg_norm = np.mean(np.linalg.norm(X, axis=1))
            
            print(f"Iter {k}: Alpha={alpha_k:.4f}, Tau={tau_k:.2f}, Loss={avg_loss:.4f}, Norm={avg_norm:.2f}")
    
    # --- 3. Results ---
    print("Optimization Finished.")
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, linewidth=2)
    plt.title(f"Convergence: {config.CLIPPING_METHOD} ({config.DATASET_NAME})")
    plt.xlabel("Iterations (x100)")
    plt.ylabel("Global Loss (Error)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.yscale('log')  # Log scale usually looks better for convergence
    plt.show()

if __name__ == "__main__":
    main()