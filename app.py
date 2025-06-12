import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from scipy.stats import multivariate_normal
from sklearn.datasets import make_classification
import seaborn as sns

# Set page configuration
st.set_page_config(
    page_title="Gaussian Discriminant Analysis Explained",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

class GDA:
    """
    Gaussian Discriminant Analysis implementation from scratch
    """
    def __init__(self):
        self.phi = None
        self.mu0 = None
        self.mu1 = None
        self.sigma = None
        
    def fit(self, X, y):
        """
        Fit the GDA model using Maximum Likelihood Estimation
        
        Parameters:
        X: Feature matrix (n_samples, n_features)
        y: Target vector (n_samples,) with binary labels (0 or 1)
        """
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        m = len(y)  # number of training examples
        
        # Estimate φ (class prior probability)
        self.phi = np.mean(y)
        
        # Estimate μ₀ and μ₁ (class means)
        self.mu0 = np.mean(X[y == 0], axis=0)
        self.mu1 = np.mean(X[y == 1], axis=0)
        
        # Estimate Σ (shared covariance matrix)
        X0_centered = X[y == 0] - self.mu0
        X1_centered = X[y == 1] - self.mu1
        
        self.sigma = (X0_centered.T @ X0_centered + X1_centered.T @ X1_centered) / m
        
        return self
    
    def predict_proba(self, X):
        """
        Predict class probabilities using Bayes rule
        """
        X = np.array(X)
        
        # Calculate likelihoods using multivariate Gaussian
        if self.sigma is not None and self.mu0 is not None and self.mu1 is not None and self.phi is not None:
            likelihood_0 = multivariate_normal.pdf(X, self.mu0, self.sigma)
            likelihood_1 = multivariate_normal.pdf(X, self.mu1, self.sigma)
            
            # Apply Bayes rule
            prior_0 = 1 - self.phi
            prior_1 = self.phi
        else:
            raise ValueError("Model not fitted yet")
        
        posterior_0 = likelihood_0 * prior_0
        posterior_1 = likelihood_1 * prior_1
        
        # Normalize
        total = posterior_0 + posterior_1
        prob_0 = posterior_0 / total
        prob_1 = posterior_1 / total
        
        return np.column_stack([prob_0, prob_1])
    
    def predict(self, X):
        """
        Make predictions using argmax
        """
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)

def generate_sample_data(n_samples=200, random_state=42):
    """
    Generate 2D sample data for two classes
    """
    np.random.seed(random_state)
    
    # Class 0: centered around (-2, -2)
    mu0 = np.array([-2, -2])
    cov0 = np.array([[2, 0.5], [0.5, 1.5]])
    X0 = np.random.multivariate_normal(mu0, cov0, n_samples//2)
    y0 = np.zeros(n_samples//2)
    
    # Class 1: centered around (2, 2)
    mu1 = np.array([2, 2])
    cov1 = np.array([[1.5, -0.3], [-0.3, 2]])
    X1 = np.random.multivariate_normal(mu1, cov1, n_samples//2)
    y1 = np.ones(n_samples//2)
    
    # Combine
    X = np.vstack([X0, X1])
    y = np.hstack([y0, y1])
    
    # Shuffle
    indices = np.random.permutation(len(y))
    return X[indices], y[indices]

def plot_data_and_decision_boundary(X, y, gda_model=None, title="Data Visualization"):
    """
    Plot 2D data with optional decision boundary
    """
    fig = go.Figure()
    
    # Plot class 0
    class_0_mask = y == 0
    fig.add_trace(go.Scatter(
        x=X[class_0_mask, 0],
        y=X[class_0_mask, 1],
        mode='markers',
        name='Class 0',
        marker=dict(color='blue', size=8, opacity=0.7)
    ))
    
    # Plot class 1
    class_1_mask = y == 1
    fig.add_trace(go.Scatter(
        x=X[class_1_mask, 0],
        y=X[class_1_mask, 1],
        mode='markers',
        name='Class 1',
        marker=dict(color='red', size=8, opacity=0.7)
    ))
    
    if gda_model is not None:
        # Create decision boundary
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                            np.linspace(y_min, y_max, 100))
        
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        Z = gda_model.predict_proba(grid_points)[:, 1]
        Z = Z.reshape(xx.shape)
        
        # Add contour for decision boundary
        fig.add_trace(go.Contour(
            x=np.linspace(x_min, x_max, 100),
            y=np.linspace(y_min, y_max, 100),
            z=Z,
            contours=dict(start=0.5, end=0.5, size=0),
            line=dict(color='green', width=3),
            showscale=False,
            name='Decision Boundary'
        ))
        
        # Add class means
        fig.add_trace(go.Scatter(
            x=[gda_model.mu0[0]],
            y=[gda_model.mu0[1]],
            mode='markers',
            name='μ₀',
            marker=dict(color='darkblue', size=15, symbol='star')
        ))
        
        fig.add_trace(go.Scatter(
            x=[gda_model.mu1[0]],
            y=[gda_model.mu1[1]],
            mode='markers',
            name='μ₁',
            marker=dict(color='darkred', size=15, symbol='star')
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Feature 1",
        yaxis_title="Feature 2",
        showlegend=True,
        width=700,
        height=500
    )
    
    return fig

def main():
    st.title("📊 Gaussian Discriminant Analysis (GDA) Explained")
    st.markdown("---")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    section = st.sidebar.radio(
        "Choose a section:",
        ["Theory & Concepts", "Mathematical Formulation", "Implementation", "Interactive Demo", "Gaussian Density Visualization", "Step-by-Step Example"]
    )
    
    if section == "Theory & Concepts":
        st.header("🎯 What is Gaussian Discriminant Analysis?")
        
        st.markdown("""
        **Gaussian Discriminant Analysis (GDA)** is a generative classification algorithm that models the distribution 
        of features for each class using multivariate Gaussian distributions. Unlike logistic regression (which is discriminative), 
        GDA models the joint probability distribution P(x,y).
        
        ### Key Concepts:
        
        **🔹 Generative vs. Discriminative Models:**
        - **Generative**: Models P(x|y) and P(y), then uses Bayes rule to find P(y|x)
        - **Discriminative**: Directly models P(y|x)
        
        **🔹 Assumptions:**
        1. Features follow a multivariate Gaussian distribution for each class
        2. All classes share the same covariance matrix (homoscedastic)
        3. Features are continuous
        
        **🔹 When to use GDA:**
        - When the Gaussian assumption is reasonable
        - When you have limited training data (GDA can be more efficient than logistic regression)
        - When you need to generate new samples from each class
        """)
        
        # Visual comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Generative Approach (GDA)")
            st.markdown("""
            1. Model P(x|y=0) and P(x|y=1) using Gaussians
            2. Model P(y) using Bernoulli distribution
            3. Use Bayes rule: P(y|x) = P(x|y)P(y) / P(x)
            """)
        
        with col2:
            st.subheader("Discriminative Approach (Logistic Regression)")
            st.markdown("""
            1. Directly model P(y|x) using sigmoid function
            2. Learn decision boundary directly
            3. No assumption about feature distribution
            """)
    
    elif section == "Mathematical Formulation":
        st.header("📐 Mathematical Formulation")
        
        st.subheader("Model Assumptions")
        st.latex(r"""
        \begin{align}
        y &\sim \text{Bernoulli}(\phi) \\
        x|y=0 &\sim \mathcal{N}(\mu_0, \Sigma) \\
        x|y=1 &\sim \mathcal{N}(\mu_1, \Sigma)
        \end{align}
        """)
        
        st.markdown("Where:")
        st.markdown("- φ is the class prior probability P(y=1)")
        st.markdown("- μ₀, μ₁ are the class means")
        st.markdown("- Σ is the shared covariance matrix")
        
        st.subheader("Multivariate Gaussian Distribution")
        st.latex(r"""
        p(x|\mu, \Sigma) = \frac{1}{(2\pi)^{n/2}|\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu)\right)
        """)
        
        st.subheader("Maximum Likelihood Estimation")
        st.markdown("Given training data {(x⁽ⁱ⁾, y⁽ⁱ⁾)}, we maximize the log-likelihood:")
        
        st.latex(r"""
        \ell(\phi, \mu_0, \mu_1, \Sigma) = \sum_{i=1}^m \log p(x^{(i)}, y^{(i)})
        """)
        
        st.markdown("**Parameter estimates:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.latex(r"""
            \phi = \frac{1}{m}\sum_{i=1}^m \mathbf{1}\{y^{(i)} = 1\}
            """)
            
            st.latex(r"""
            \mu_0 = \frac{\sum_{i=1}^m \mathbf{1}\{y^{(i)} = 0\} x^{(i)}}{\sum_{i=1}^m \mathbf{1}\{y^{(i)} = 0\}}
            """)
        
        with col2:
            st.latex(r"""
            \mu_1 = \frac{\sum_{i=1}^m \mathbf{1}\{y^{(i)} = 1\} x^{(i)}}{\sum_{i=1}^m \mathbf{1}\{y^{(i)} = 1\}}
            """)
            
            st.latex(r"""
            \Sigma = \frac{1}{m}\sum_{i=1}^m (x^{(i)} - \mu_{y^{(i)}})(x^{(i)} - \mu_{y^{(i)}})^T
            """)
        
        st.subheader("Classification using Bayes Rule")
        st.latex(r"""
        P(y=1|x) = \frac{P(x|y=1)P(y=1)}{P(x|y=0)P(y=0) + P(x|y=1)P(y=1)}
        """)
        
        st.markdown("**Decision rule:** Predict y = 1 if P(y=1|x) > 0.5")
    
    elif section == "Implementation":
        st.header("💻 Python Implementation")
        
        st.subheader("Complete GDA Class Implementation")
        
        st.code("""
class GDA:
    def __init__(self):
        self.phi = None      # Class prior P(y=1)
        self.mu0 = None      # Mean of class 0
        self.mu1 = None      # Mean of class 1
        self.sigma = None    # Shared covariance matrix
        
    def fit(self, X, y):
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        m = len(y)  # number of training examples
        
        # Estimate φ using MLE
        self.phi = np.mean(y)
        
        # Estimate class means using MLE
        self.mu0 = np.mean(X[y == 0], axis=0)
        self.mu1 = np.mean(X[y == 1], axis=0)
        
        # Estimate shared covariance matrix using MLE
        X0_centered = X[y == 0] - self.mu0
        X1_centered = X[y == 1] - self.mu1
        
        self.sigma = (X0_centered.T @ X0_centered + 
                     X1_centered.T @ X1_centered) / m
        
        return self
    
    def predict_proba(self, X):
        X = np.array(X)
        
        # Calculate likelihoods using multivariate Gaussian
        likelihood_0 = multivariate_normal.pdf(X, self.mu0, self.sigma)
        likelihood_1 = multivariate_normal.pdf(X, self.mu1, self.sigma)
        
        # Apply Bayes rule
        prior_0 = 1 - self.phi
        prior_1 = self.phi
        
        posterior_0 = likelihood_0 * prior_0
        posterior_1 = likelihood_1 * prior_1
        
        # Normalize to get probabilities
        total = posterior_0 + posterior_1
        prob_0 = posterior_0 / total
        prob_1 = posterior_1 / total
        
        return np.column_stack([prob_0, prob_1])
    
    def predict(self, X):
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)
        """, language='python')
        
        st.subheader("Key Implementation Details")
        
        st.markdown("""
        **🔹 Parameter Estimation (MLE):**
        - **φ**: Simple average of class labels
        - **μ₀, μ₁**: Sample means for each class
        - **Σ**: Pooled covariance matrix from both classes
        
        **🔹 Prediction Process:**
        1. Calculate likelihood P(x|y=0) and P(x|y=1) using multivariate Gaussian PDF
        2. Multiply by priors P(y=0) and P(y=1)
        3. Normalize to get posterior probabilities
        4. Use argmax for final classification
        
        **🔹 Multivariate Gaussian PDF:**
        - Uses scipy.stats.multivariate_normal for numerical stability
        - Handles matrix operations efficiently
        """)
    
    elif section == "Interactive Demo":
        st.header("🎮 Interactive Demo")
        
        st.markdown("Adjust the parameters below to see how they affect the GDA model:")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Data Generation Parameters")
            n_samples = st.slider("Number of samples", 50, 500, 200)
            random_state = st.slider("Random seed", 0, 100, 42)
            
            st.subheader("Visualization Options")
            show_decision_boundary = st.checkbox("Show decision boundary", True)
            show_parameters = st.checkbox("Show estimated parameters", True)
        
        # Generate data
        X, y = generate_sample_data(n_samples, random_state)
        
        # Fit GDA model
        gda = GDA()
        gda.fit(X, y)
        
        with col2:
            # Plot data and decision boundary
            if show_decision_boundary:
                fig = plot_data_and_decision_boundary(X, y, gda, "GDA Classification with Decision Boundary")
            else:
                fig = plot_data_and_decision_boundary(X, y, None, "Training Data")
            
            st.plotly_chart(fig, use_container_width=True)
        
        if show_parameters:
            st.subheader("📊 Estimated Parameters")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Class Prior φ", f"{gda.phi:.3f}")
                st.markdown("P(y=1)")
            
            with col2:
                st.markdown("**Class 0 Mean μ₀:**")
                st.write(f"[{gda.mu0[0]:.2f}, {gda.mu0[1]:.2f}]")
            
            with col3:
                st.markdown("**Class 1 Mean μ₁:**")
                st.write(f"[{gda.mu1[0]:.2f}, {gda.mu1[1]:.2f}]")
            
            st.markdown("**Shared Covariance Matrix Σ:**")
            sigma_df = pd.DataFrame(gda.sigma, 
                                   columns=[f"Feature {i+1}" for i in range(2)],
                                   index=[f"Feature {i+1}" for i in range(2)])
            st.dataframe(sigma_df.round(3))
        
        # Model performance
        st.subheader("🎯 Model Performance")
        predictions = gda.predict(X)
        accuracy = np.mean(predictions == y)
        st.metric("Training Accuracy", f"{accuracy:.3f}")
        
        # Confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y, predictions)
        
        fig_cm = px.imshow(cm, 
                          labels=dict(x="Predicted", y="Actual", color="Count"),
                          x=['Class 0', 'Class 1'],
                          y=['Class 0', 'Class 1'],
                          title="Confusion Matrix")
        fig_cm.update_layout(width=400, height=400)
        st.plotly_chart(fig_cm, use_container_width=True)
    
    elif section == "Gaussian Density Visualization":
        st.header("🌊 Gaussian Density Visualization")
        
        st.markdown("""
        Understanding how the multivariate Gaussian distribution looks in 3D helps visualize why GDA works. 
        The shape of the distribution depends on the mean (μ) and covariance matrix (Σ).
        """)
        
        # Interactive controls
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Distribution Parameters")
            
            # Mean controls
            st.markdown("**Mean (μ):**")
            mu_x = st.slider("μₓ", -3.0, 3.0, 0.0, 0.1)
            mu_y = st.slider("μᵧ", -3.0, 3.0, 0.0, 0.1)
            
            # Covariance controls
            st.markdown("**Covariance Matrix (Σ):**")
            sigma_xx = st.slider("σₓₓ (X variance)", 0.1, 3.0, 1.0, 0.1)
            sigma_yy = st.slider("σᵧᵧ (Y variance)", 0.1, 3.0, 1.0, 0.1)
            sigma_xy = st.slider("σₓᵧ (covariance)", -1.5, 1.5, 0.0, 0.1)
            
            # Visualization options
            st.markdown("**Visualization Options:**")
            plot_type = st.selectbox("Plot Type", ["3D Surface", "Contour Plot", "Both"])
            resolution = st.slider("Grid Resolution", 20, 100, 50)
        
        with col2:
            # Create mean vector and covariance matrix
            mu = np.array([mu_x, mu_y])
            sigma = np.array([[sigma_xx, sigma_xy], 
                             [sigma_xy, sigma_yy]])
            
            # Check if covariance matrix is positive definite
            try:
                # Create grid
                x_range = np.linspace(-5, 5, resolution)
                y_range = np.linspace(-5, 5, resolution)
                X_grid, Y_grid = np.meshgrid(x_range, y_range)
                
                # Calculate density
                pos = np.dstack((X_grid, Y_grid))
                rv = multivariate_normal(mu, sigma)
                Z = rv.pdf(pos)
                
                if plot_type in ["3D Surface", "Both"]:
                    # 3D Surface plot
                    fig_3d = go.Figure(data=[go.Surface(
                        x=X_grid, y=Y_grid, z=Z,
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Density")
                    )])
                    
                    fig_3d.update_layout(
                        title="3D Gaussian Density Surface",
                        scene=dict(
                            xaxis_title="X",
                            yaxis_title="Y",
                            zaxis_title="Density",
                            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
                        ),
                        width=700,
                        height=500
                    )
                    
                    st.plotly_chart(fig_3d, use_container_width=True)
                
                if plot_type in ["Contour Plot", "Both"]:
                    # Contour plot
                    fig_contour = go.Figure(data=go.Contour(
                        x=x_range,
                        y=y_range,
                        z=Z,
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Density"),
                        contours=dict(
                            showlabels=True,
                            labelfont=dict(size=10, color='white')
                        )
                    ))
                    
                    # Add mean point
                    fig_contour.add_trace(go.Scatter(
                        x=[mu_x], y=[mu_y],
                        mode='markers',
                        marker=dict(color='red', size=12, symbol='star'),
                        name='Mean (μ)'
                    ))
                    
                    fig_contour.update_layout(
                        title="Gaussian Density Contour Plot",
                        xaxis_title="X",
                        yaxis_title="Y",
                        width=700,
                        height=500
                    )
                    
                    st.plotly_chart(fig_contour, use_container_width=True)
                
            except np.linalg.LinAlgError:
                st.error("⚠️ The covariance matrix is not positive definite. Please adjust the parameters.")
        
        # Display current parameters
        st.subheader("📊 Current Parameters")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Mean Vector:**")
            st.code(f"μ = [{mu_x:.1f}, {mu_y:.1f}]")
            
        with col2:
            st.markdown("**Covariance Matrix:**")
            st.code(f"Σ = [[{sigma_xx:.1f}, {sigma_xy:.1f}]\n     [{sigma_xy:.1f}, {sigma_yy:.1f}]]")
        
        # Educational content about covariance effects
        st.subheader("📚 Understanding Covariance Effects")
        
        st.markdown("""
        **How parameters affect the distribution shape:**
        
        - **σₓₓ, σᵧᵧ (Diagonal elements)**: Control the spread along X and Y axes
          - Larger values → More spread out distribution
          - Smaller values → More concentrated distribution
        
        - **σₓᵧ (Off-diagonal element)**: Controls the correlation between X and Y
          - σₓᵧ = 0: No correlation (ellipse aligned with axes)
          - σₓᵧ > 0: Positive correlation (ellipse tilted ↗)
          - σₓᵧ < 0: Negative correlation (ellipse tilted ↘)
        
        - **Mean (μ)**: Shifts the center of the distribution
        """)
        
        # Preset examples section
        st.subheader("🎯 Try These Examples")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Standard Normal"):
                st.session_state.update({
                    'mu_x': 0.0, 'mu_y': 0.0,
                    'sigma_xx': 1.0, 'sigma_yy': 1.0, 'sigma_xy': 0.0
                })
                st.rerun()
        
        with col2:
            if st.button("Positive Correlation"):
                st.session_state.update({
                    'mu_x': 0.0, 'mu_y': 0.0,
                    'sigma_xx': 2.0, 'sigma_yy': 1.5, 'sigma_xy': 1.2
                })
                st.rerun()
        
        with col3:
            if st.button("Negative Correlation"):
                st.session_state.update({
                    'mu_x': 1.0, 'mu_y': -1.0,
                    'sigma_xx': 1.5, 'sigma_yy': 2.0, 'sigma_xy': -1.0
                })
                st.rerun()
        
        # Multiple distributions comparison
        st.subheader("🔄 Compare Multiple Distributions")
        
        if st.checkbox("Show comparison with GDA classes"):
            # Generate sample GDA data
            X_demo, y_demo = generate_sample_data(100, 42)
            gda_demo = GDA()
            gda_demo.fit(X_demo, y_demo)
            
            # Create comparison plot
            fig_comparison = make_subplots(
                rows=1, cols=2,
                subplot_titles=("Class 0 Distribution", "Class 1 Distribution"),
                specs=[[{"type": "contour"}, {"type": "contour"}]]
            )
            
            # Class 0 distribution
            x_range = np.linspace(-6, 6, 50)
            y_range = np.linspace(-6, 6, 50)
            X_grid, Y_grid = np.meshgrid(x_range, y_range)
            pos = np.dstack((X_grid, Y_grid))
            
            rv0 = multivariate_normal(gda_demo.mu0, gda_demo.sigma)
            Z0 = rv0.pdf(pos)
            
            rv1 = multivariate_normal(gda_demo.mu1, gda_demo.sigma)
            Z1 = rv1.pdf(pos)
            
            fig_comparison.add_trace(
                go.Contour(x=x_range, y=y_range, z=Z0, 
                          colorscale='Blues', name='Class 0'),
                row=1, col=1
            )
            
            fig_comparison.add_trace(
                go.Contour(x=x_range, y=y_range, z=Z1, 
                          colorscale='Reds', name='Class 1'),
                row=1, col=2
            )
            
            # Add data points
            fig_comparison.add_trace(
                go.Scatter(x=X_demo[y_demo==0, 0], y=X_demo[y_demo==0, 1],
                          mode='markers', marker=dict(color='blue', size=4),
                          name='Class 0 Data'),
                row=1, col=1
            )
            
            fig_comparison.add_trace(
                go.Scatter(x=X_demo[y_demo==1, 0], y=X_demo[y_demo==1, 1],
                          mode='markers', marker=dict(color='red', size=4),
                          name='Class 1 Data'),
                row=1, col=2
            )
            
            fig_comparison.update_layout(
                title="GDA: Class-Conditional Distributions",
                height=400
            )
            
            st.plotly_chart(fig_comparison, use_container_width=True)
            
            # Display learned parameters
            st.markdown("**Learned Parameters from Sample Data:**")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.code(f"φ = {gda_demo.phi:.3f}")
            
            with col2:
                st.code(f"μ₀ = [{gda_demo.mu0[0]:.2f}, {gda_demo.mu0[1]:.2f}]")
            
            with col3:
                st.code(f"μ₁ = [{gda_demo.mu1[0]:.2f}, {gda_demo.mu1[1]:.2f}]")
    
    elif section == "Step-by-Step Example":
        st.header("👣 Step-by-Step Example")
        
        st.markdown("Let's walk through the complete GDA process with a concrete example:")
        
        # Generate small dataset for clarity
        np.random.seed(42)
        X_small, y_small = generate_sample_data(20, 42)
        
        st.subheader("Step 1: Generate Sample Data")
        st.markdown("We have 20 samples with 2 features each:")
        
        # Display data
        df = pd.DataFrame(X_small, columns=['Feature 1', 'Feature 2'])
        df['Class'] = y_small.astype(int)
        st.dataframe(df)
        
        # Plot initial data
        fig_initial = plot_data_and_decision_boundary(X_small, y_small, None, "Initial Training Data")
        st.plotly_chart(fig_initial, use_container_width=True)
        
        st.subheader("Step 2: Estimate Parameters using MLE")
        
        # Fit model
        gda_example = GDA()
        gda_example.fit(X_small, y_small)
        
        st.markdown("**Estimated φ (class prior):**")
        st.code(f"φ = {np.mean(y_small):.3f}")
        st.markdown(f"This means P(y=1) = {gda_example.phi:.3f}")
        
        st.markdown("**Estimated class means:**")
        col1, col2 = st.columns(2)
        
        with col1:
            st.code(f"μ₀ = [{gda_example.mu0[0]:.2f}, {gda_example.mu0[1]:.2f}]")
            st.markdown("Mean of class 0")
        
        with col2:
            st.code(f"μ₁ = [{gda_example.mu1[0]:.2f}, {gda_example.mu1[1]:.2f}]")
            st.markdown("Mean of class 1")
        
        st.markdown("**Estimated covariance matrix:**")
        st.code(f"""Σ = [[{gda_example.sigma[0,0]:.2f}, {gda_example.sigma[0,1]:.2f}],
     [{gda_example.sigma[1,0]:.2f}, {gda_example.sigma[1,1]:.2f}]]""")
        
        st.subheader("Step 3: Make Predictions")
        
        # Test point
        test_point = np.array([[0, 0]])
        probs = gda_example.predict_proba(test_point)
        prediction = gda_example.predict(test_point)
        
        st.markdown(f"**Test point:** x = [0, 0]")
        
        # Calculate likelihoods manually for explanation
        likelihood_0 = multivariate_normal.pdf(test_point, gda_example.mu0, gda_example.sigma)
        likelihood_1 = multivariate_normal.pdf(test_point, gda_example.mu1, gda_example.sigma)
        
        # Handle scalar vs array results
        if np.ndim(likelihood_0) > 0:
            likelihood_0 = likelihood_0[0]
        if np.ndim(likelihood_1) > 0:
            likelihood_1 = likelihood_1[0]
        
        st.markdown("**Step 3a: Calculate Likelihoods**")
        st.code(f"""P(x|y=0) = {likelihood_0:.6f}
P(x|y=1) = {likelihood_1:.6f}""")
        
        st.markdown("**Step 3b: Apply Bayes Rule**")
        prior_0 = 1 - gda_example.phi
        prior_1 = gda_example.phi
        
        posterior_0_unnorm = likelihood_0 * prior_0
        posterior_1_unnorm = likelihood_1 * prior_1
        
        st.code(f"""P(x|y=0) × P(y=0) = {likelihood_0:.6f} × {prior_0:.3f} = {posterior_0_unnorm:.6f}
P(x|y=1) × P(y=1) = {likelihood_1:.6f} × {prior_1:.3f} = {posterior_1_unnorm:.6f}""")
        
        st.markdown("**Step 3c: Normalize**")
        total = posterior_0_unnorm + posterior_1_unnorm
        st.code(f"""P(y=0|x) = {posterior_0_unnorm:.6f} / {total:.6f} = {probs[0,0]:.3f}
P(y=1|x) = {posterior_1_unnorm:.6f} / {total:.6f} = {probs[0,1]:.3f}""")
        
        st.markdown("**Final Prediction:**")
        st.code(f"argmax = Class {prediction[0]} (probability = {max(probs[0]):.3f})")
        
        # Visualize with test point
        fig_final = plot_data_and_decision_boundary(X_small, y_small, gda_example, 
                                                   "GDA Model with Test Point")
        
        # Add test point
        fig_final.add_trace(go.Scatter(
            x=[0],
            y=[0],
            mode='markers',
            name='Test Point',
            marker=dict(color='purple', size=15, symbol='diamond')
        ))
        
        st.plotly_chart(fig_final, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    ### 📚 Key Takeaways:
    1. **GDA is a generative model** that models the distribution of features for each class
    2. **Maximum Likelihood Estimation** is used to estimate parameters φ, μ₀, μ₁, and Σ
    3. **Bayes rule** is applied to make predictions using the learned distributions
    4. **Shared covariance assumption** makes GDA more parameter-efficient but less flexible than QDA
    5. **Works well when Gaussian assumption holds** and can outperform logistic regression with limited data
    """)

if __name__ == "__main__":
    main()
