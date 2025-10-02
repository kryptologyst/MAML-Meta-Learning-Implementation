"""
Project 151: Meta-learning Implementation
=========================================

Meta-learning, or "learning to learn," aims to train models that can quickly adapt 
to new tasks with minimal data. This project implements Model-Agnostic Meta-Learning (MAML) 
— a popular meta-learning algorithm that learns initial parameters which can be 
fine-tuned rapidly on new tasks.

Features:
- Modern PyTorch implementation with type hints
- Interactive visualization dashboard
- Mock database for task management
- Web UI for experiments
- Comprehensive logging and metrics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import logging
import json
import sqlite3
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional
from pathlib import Path
import streamlit as st
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Configuration
@dataclass
class MAMLConfig:
    """Configuration for MAML training"""
    meta_lr: float = 0.001
    inner_lr: float = 0.01
    k_shot: int = 10
    num_tasks: int = 5
    inner_steps: int = 1
    episodes: int = 200
    hidden_size: int = 40
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_interval: int = 20
    save_interval: int = 50

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('maml_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Define a modern MLP for regression with type hints
class MAMLModel(nn.Module):
    """Model-Agnostic Meta-Learning neural network"""
    
    def __init__(self, input_size: int = 1, hidden_size: int = 40, output_size: int = 1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        
        # Initialize weights properly
        self._initialize_weights()
    
    def _initialize_weights(self) -> None:
        """Initialize network weights using Xavier initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network"""
        return self.net(x)
    
    def get_parameters(self) -> List[torch.Tensor]:
        """Get model parameters as a list"""
        return list(self.parameters())

# Mock Database for Task Management
class TaskDatabase:
    """Mock database for storing and retrieving meta-learning tasks"""
    
    def __init__(self, db_path: str = "maml_tasks.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self) -> None:
        """Initialize the SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_type TEXT NOT NULL,
                parameters TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                performance REAL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                config TEXT NOT NULL,
                metrics TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info(f"Database initialized at {self.db_path}")
    
    def save_task(self, task_type: str, parameters: Dict, performance: Optional[float] = None) -> int:
        """Save a task to the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO tasks (task_type, parameters, performance)
            VALUES (?, ?, ?)
        ''', (task_type, json.dumps(parameters), performance))
        
        task_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        logger.info(f"Saved task {task_id} of type {task_type}")
        return task_id
    
    def get_tasks(self, task_type: Optional[str] = None, limit: int = 100) -> List[Dict]:
        """Retrieve tasks from the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if task_type:
            cursor.execute('''
                SELECT id, task_type, parameters, created_at, performance
                FROM tasks WHERE task_type = ? ORDER BY created_at DESC LIMIT ?
            ''', (task_type, limit))
        else:
            cursor.execute('''
                SELECT id, task_type, parameters, created_at, performance
                FROM tasks ORDER BY created_at DESC LIMIT ?
            ''', (limit,))
        
        tasks = []
        for row in cursor.fetchall():
            tasks.append({
                'id': row[0],
                'task_type': row[1],
                'parameters': json.loads(row[2]),
                'created_at': row[3],
                'performance': row[4]
            })
        
        conn.close()
        return tasks
    
    def save_experiment(self, config: Dict, metrics: Dict) -> int:
        """Save experiment results"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO experiments (config, metrics)
            VALUES (?, ?)
        ''', (json.dumps(config), json.dumps(metrics)))
        
        exp_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        logger.info(f"Saved experiment {exp_id}")
        return exp_id

# Enhanced Task Generation
class TaskGenerator:
    """Generate various types of meta-learning tasks"""
    
    @staticmethod
    def get_sinusoid_task(k: int = 10, x_range: Tuple[float, float] = (-5.0, 5.0)) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate a sinusoid regression task"""
        A = np.random.uniform(0.1, 5.0)
        phase = np.random.uniform(0, np.pi)
        x = np.random.uniform(x_range[0], x_range[1], size=(k, 1))
        y = A * np.sin(x + phase)
        
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    
    @staticmethod
    def get_linear_task(k: int = 10, x_range: Tuple[float, float] = (-5.0, 5.0)) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate a linear regression task"""
        slope = np.random.uniform(-2.0, 2.0)
        intercept = np.random.uniform(-2.0, 2.0)
        x = np.random.uniform(x_range[0], x_range[1], size=(k, 1))
        y = slope * x + intercept
        
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    
    @staticmethod
    def get_quadratic_task(k: int = 10, x_range: Tuple[float, float] = (-3.0, 3.0)) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate a quadratic regression task"""
        a = np.random.uniform(-1.0, 1.0)
        b = np.random.uniform(-2.0, 2.0)
        c = np.random.uniform(-2.0, 2.0)
        x = np.random.uniform(x_range[0], x_range[1], size=(k, 1))
        y = a * x**2 + b * x + c
        
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# Modern MAML Training Class
class MAMLTrainer:
    """Modern MAML trainer with comprehensive metrics and logging"""
    
    def __init__(self, model: MAMLModel, config: MAMLConfig, database: TaskDatabase):
        self.model = model.to(config.device)
        self.config = config
        self.database = database
        self.meta_optimizer = optim.Adam(self.model.parameters(), lr=config.meta_lr)
        
        # Metrics tracking
        self.metrics = {
            'meta_losses': [],
            'episodes': [],
            'task_performances': [],
            'adaptation_times': []
        }
        
        logger.info(f"MAML Trainer initialized on device: {config.device}")
    
    def train(self) -> Dict:
        """Train the MAML model"""
        logger.info("Starting MAML training...")
        start_time = datetime.now()
        
        for episode in range(self.config.episodes):
            episode_start = datetime.now()
            meta_loss = 0.0
            task_performances = []
            
            for task_idx in range(self.config.num_tasks):
                # Generate training and validation data
                x_train, y_train = TaskGenerator.get_sinusoid_task(self.config.k_shot)
                x_val, y_val = TaskGenerator.get_sinusoid_task(self.config.k_shot)
                
                # Move to device
                x_train, y_train = x_train.to(self.config.device), y_train.to(self.config.device)
                x_val, y_val = x_val.to(self.config.device), y_val.to(self.config.device)
                
                # Task-specific adaptation
                fast_weights = self._adapt_to_task(x_train, y_train)
                
                # Evaluate adapted model
                adapted_preds = self._forward_with_weights(x_val, fast_weights)
                task_loss = F.mse_loss(adapted_preds, y_val)
                meta_loss += task_loss
                task_performances.append(task_loss.item())
            
            # Meta-update
            meta_loss /= self.config.num_tasks
            self.meta_optimizer.zero_grad()
            meta_loss.backward()
            self.meta_optimizer.step()
            
            # Track metrics
            episode_time = (datetime.now() - episode_start).total_seconds()
            self.metrics['meta_losses'].append(meta_loss.item())
            self.metrics['episodes'].append(episode)
            self.metrics['task_performances'].append(np.mean(task_performances))
            self.metrics['adaptation_times'].append(episode_time)
            
            # Logging
            if episode % self.config.log_interval == 0:
                logger.info(f"Episode {episode}/{self.config.episodes} - "
                          f"Meta Loss: {meta_loss.item():.4f} - "
                          f"Avg Task Performance: {np.mean(task_performances):.4f} - "
                          f"Time: {episode_time:.2f}s")
            
            # Save checkpoint
            if episode % self.config.save_interval == 0 and episode > 0:
                self._save_checkpoint(episode)
        
        training_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # Save final experiment
        experiment_id = self.database.save_experiment(
            config=self.config.__dict__,
            metrics=self.metrics
        )
        
        return {
            'experiment_id': experiment_id,
            'training_time': training_time,
            'final_meta_loss': meta_loss.item(),
            'metrics': self.metrics
        }
    
    def _adapt_to_task(self, x_train: torch.Tensor, y_train: torch.Tensor) -> List[torch.Tensor]:
        """Adapt model to a specific task using gradient descent"""
        fast_weights = [p.clone() for p in self.model.parameters()]
        
        for step in range(self.config.inner_steps):
            # Forward pass with current weights
            preds = self._forward_with_weights(x_train, fast_weights)
            loss = F.mse_loss(preds, y_train)
            
            # Compute gradients
            grads = torch.autograd.grad(loss, fast_weights, create_graph=True, retain_graph=True)
            
            # Update fast weights
            fast_weights = [w - self.config.inner_lr * g for w, g in zip(fast_weights, grads)]
        
        return fast_weights
    
    def _forward_with_weights(self, x: torch.Tensor, weights: List[torch.Tensor]) -> torch.Tensor:
        """Forward pass using specific weights"""
        # Manual forward pass with custom weights
        x = F.linear(x, weights[0], weights[1])
        x = F.relu(x)
        x = F.linear(x, weights[2], weights[3])
        x = F.relu(x)
        x = F.linear(x, weights[4], weights[5])
        return x
    
    def _save_checkpoint(self, episode: int) -> None:
        """Save model checkpoint"""
        checkpoint = {
            'episode': episode,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.meta_optimizer.state_dict(),
            'config': self.config.__dict__,
            'metrics': self.metrics
        }
        
        checkpoint_path = f"checkpoints/maml_checkpoint_episode_{episode}.pth"
        Path("checkpoints").mkdir(exist_ok=True)
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def evaluate_on_new_task(self, task_type: str = "sinusoid", k_shot: int = 10) -> Dict:
        """Evaluate the meta-learned model on a new task"""
        logger.info(f"Evaluating on new {task_type} task...")
        
        # Generate new task
        if task_type == "sinusoid":
            x_test, y_test = TaskGenerator.get_sinusoid_task(k_shot)
        elif task_type == "linear":
            x_test, y_test = TaskGenerator.get_linear_task(k_shot)
        elif task_type == "quadratic":
            x_test, y_test = TaskGenerator.get_quadratic_task(k_shot)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
        
        x_test, y_test = x_test.to(self.config.device), y_test.to(self.config.device)
        
        # Quick adaptation
        fast_weights = self._adapt_to_task(x_test, y_test)
        
        # Evaluate
        with torch.no_grad():
            adapted_preds = self._forward_with_weights(x_test, fast_weights)
            mse_loss = F.mse_loss(adapted_preds, y_test).item()
            mae_loss = F.l1_loss(adapted_preds, y_test).item()
        
        # Save task to database
        task_id = self.database.save_task(
            task_type=task_type,
            parameters={'k_shot': k_shot, 'x_range': (-5.0, 5.0)},
            performance=mse_loss
        )
        
        return {
            'task_id': task_id,
            'task_type': task_type,
            'mse_loss': mse_loss,
            'mae_loss': mae_loss,
            'predictions': adapted_preds.cpu().numpy(),
            'targets': y_test.cpu().numpy(),
            'inputs': x_test.cpu().numpy()
        }

# Visualization and UI Components
class MAMLVisualizer:
    """Interactive visualization for MAML experiments"""
    
    @staticmethod
    def plot_training_metrics(metrics: Dict) -> go.Figure:
        """Create interactive training metrics plot"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Meta Loss', 'Task Performance', 'Adaptation Time', 'Loss Comparison'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Meta loss
        fig.add_trace(
            go.Scatter(x=metrics['episodes'], y=metrics['meta_losses'], 
                      mode='lines', name='Meta Loss', line=dict(color='blue')),
            row=1, col=1
        )
        
        # Task performance
        fig.add_trace(
            go.Scatter(x=metrics['episodes'], y=metrics['task_performances'], 
                      mode='lines', name='Task Performance', line=dict(color='red')),
            row=1, col=2
        )
        
        # Adaptation time
        fig.add_trace(
            go.Scatter(x=metrics['episodes'], y=metrics['adaptation_times'], 
                      mode='lines', name='Adaptation Time', line=dict(color='green')),
            row=2, col=1
        )
        
        # Loss comparison
        fig.add_trace(
            go.Scatter(x=metrics['episodes'], y=metrics['meta_losses'], 
                      mode='lines', name='Meta Loss', line=dict(color='blue')),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(x=metrics['episodes'], y=metrics['task_performances'], 
                      mode='lines', name='Task Performance', line=dict(color='red')),
            row=2, col=2
        )
        
        fig.update_layout(height=800, title_text="MAML Training Metrics", showlegend=True)
        return fig
    
    @staticmethod
    def plot_task_prediction(evaluation_result: Dict) -> go.Figure:
        """Plot task prediction results"""
        fig = go.Figure()
        
        # Sort by x values for better visualization
        sorted_indices = np.argsort(evaluation_result['inputs'].flatten())
        x_sorted = evaluation_result['inputs'].flatten()[sorted_indices]
        y_true_sorted = evaluation_result['targets'].flatten()[sorted_indices]
        y_pred_sorted = evaluation_result['predictions'].flatten()[sorted_indices]
        
        # True values
        fig.add_trace(go.Scatter(
            x=x_sorted, y=y_true_sorted,
            mode='markers', name='True Values',
            marker=dict(color='blue', size=8)
        ))
        
        # Predictions
        fig.add_trace(go.Scatter(
            x=x_sorted, y=y_pred_sorted,
            mode='markers', name='MAML Predictions',
            marker=dict(color='red', size=6)
        ))
        
        fig.update_layout(
            title=f"MAML Prediction on {evaluation_result['task_type']} Task<br>"
                  f"MSE: {evaluation_result['mse_loss']:.4f}, MAE: {evaluation_result['mae_loss']:.4f}",
            xaxis_title="Input",
            yaxis_title="Output",
            height=500
        )
        
        return fig

# Streamlit Web UI
def create_streamlit_app():
    """Create interactive Streamlit web application"""
    st.set_page_config(
        page_title="MAML Meta-Learning Dashboard",
        page_icon="🧠",
        layout="wide"
    )
    
    st.title("🧠 MAML Meta-Learning Dashboard")
    st.markdown("Interactive dashboard for Model-Agnostic Meta-Learning experiments")
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    meta_lr = st.sidebar.slider("Meta Learning Rate", 0.0001, 0.01, 0.001, 0.0001)
    inner_lr = st.sidebar.slider("Inner Learning Rate", 0.001, 0.1, 0.01, 0.001)
    k_shot = st.sidebar.slider("K-Shot", 5, 50, 10, 1)
    num_tasks = st.sidebar.slider("Number of Tasks per Episode", 3, 20, 5, 1)
    episodes = st.sidebar.slider("Number of Episodes", 50, 500, 200, 10)
    inner_steps = st.sidebar.slider("Inner Steps", 1, 10, 1, 1)
    
    # Initialize components
    if 'database' not in st.session_state:
        st.session_state.database = TaskDatabase()
    
    if 'trainer' not in st.session_state:
        config = MAMLConfig(
            meta_lr=meta_lr,
            inner_lr=inner_lr,
            k_shot=k_shot,
            num_tasks=num_tasks,
            episodes=episodes,
            inner_steps=inner_steps
        )
        model = MAMLModel(hidden_size=40)
        st.session_state.trainer = MAMLTrainer(model, config, st.session_state.database)
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["Training", "Evaluation", "Database", "Visualization"])
    
    with tab1:
        st.header("Train MAML Model")
        
        if st.button("Start Training", type="primary"):
            with st.spinner("Training MAML model..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Update trainer config
                st.session_state.trainer.config.meta_lr = meta_lr
                st.session_state.trainer.config.inner_lr = inner_lr
                st.session_state.trainer.config.k_shot = k_shot
                st.session_state.trainer.config.num_tasks = num_tasks
                st.session_state.trainer.config.episodes = episodes
                st.session_state.trainer.config.inner_steps = inner_steps
                
                # Train model
                results = st.session_state.trainer.train()
                
                st.success("Training completed!")
                st.json(results)
                
                # Plot training metrics
                if results['metrics']:
                    fig = MAMLVisualizer.plot_training_metrics(results['metrics'])
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("Evaluate on New Tasks")
        
        task_type = st.selectbox("Task Type", ["sinusoid", "linear", "quadratic"])
        eval_k_shot = st.slider("K-Shot for Evaluation", 5, 50, 10, 1)
        
        if st.button("Evaluate Model"):
            with st.spinner("Evaluating model..."):
                eval_result = st.session_state.trainer.evaluate_on_new_task(task_type, eval_k_shot)
                
                st.success(f"Evaluation completed! MSE: {eval_result['mse_loss']:.4f}")
                
                # Plot results
                fig = MAMLVisualizer.plot_task_prediction(eval_result)
                st.plotly_chart(fig, use_container_width=True)
                
                st.json(eval_result)
    
    with tab3:
        st.header("Task Database")
        
        # Show saved tasks
        tasks = st.session_state.database.get_tasks(limit=50)
        
        if tasks:
            st.subheader("Recent Tasks")
            for task in tasks[:10]:
                st.write(f"**Task {task['id']}**: {task['task_type']} - "
                        f"Performance: {task['performance']:.4f} - "
                        f"Created: {task['created_at']}")
        else:
            st.info("No tasks in database yet. Run some evaluations to populate it.")
    
    with tab4:
        st.header("Interactive Visualization")
        
        st.subheader("Generate Custom Task")
        custom_task_type = st.selectbox("Task Type", ["sinusoid", "linear", "quadratic"])
        custom_k = st.slider("Number of Points", 10, 100, 50, 1)
        
        if st.button("Generate and Visualize"):
            if custom_task_type == "sinusoid":
                x, y = TaskGenerator.get_sinusoid_task(custom_k)
            elif custom_task_type == "linear":
                x, y = TaskGenerator.get_linear_task(custom_k)
            else:
                x, y = TaskGenerator.get_quadratic_task(custom_k)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=x.numpy().flatten(), y=y.numpy().flatten(),
                mode='markers', name=f'{custom_task_type} Task',
                marker=dict(size=6)
            ))
            
            fig.update_layout(
                title=f"Generated {custom_task_type} Task",
                xaxis_title="Input",
                yaxis_title="Output",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)

# Main execution
def main():
    """Main execution function"""
    logger.info("Starting MAML Meta-Learning Implementation")
    
    # Initialize components
    config = MAMLConfig()
    model = MAMLModel(hidden_size=config.hidden_size)
    database = TaskDatabase()
    trainer = MAMLTrainer(model, config, database)
    
    # Train the model
    logger.info("Training MAML model...")
    results = trainer.train()
    
    # Evaluate on different task types
    logger.info("Evaluating on different task types...")
    eval_results = {}
    for task_type in ["sinusoid", "linear", "quadratic"]:
        eval_results[task_type] = trainer.evaluate_on_new_task(task_type, k_shot=20)
        logger.info(f"{task_type} task - MSE: {eval_results[task_type]['mse_loss']:.4f}")
    
    # Create visualizations
    logger.info("Creating visualizations...")
    training_fig = MAMLVisualizer.plot_training_metrics(results['metrics'])
    training_fig.write_html("training_metrics.html")
    
    # Save evaluation plots
    for task_type, eval_result in eval_results.items():
        eval_fig = MAMLVisualizer.plot_task_prediction(eval_result)
        eval_fig.write_html(f"evaluation_{task_type}.html")
    
    logger.info("MAML implementation completed successfully!")
    return results, eval_results

if __name__ == "__main__":
    # Run the Streamlit app
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "streamlit":
        create_streamlit_app()
    else:
        # Run main training and evaluation
        main()