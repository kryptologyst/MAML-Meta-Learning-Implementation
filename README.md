# MAML Meta-Learning Implementation

A modern, comprehensive implementation of Model-Agnostic Meta-Learning (MAML) with interactive visualization, web UI, and database integration.

## Overview

Meta-learning, or "learning to learn," aims to train models that can quickly adapt to new tasks with minimal data. This project implements MAML — a popular meta-learning algorithm that learns initial parameters which can be fine-tuned rapidly on new tasks.

### Features

- **Modern PyTorch Implementation**: Latest PyTorch practices with type hints and proper error handling
- **Interactive Web UI**: Streamlit-based dashboard for experiments and visualization
- **Mock Database**: SQLite database for task storage and experiment tracking
- **Comprehensive Logging**: Detailed metrics tracking and logging system
- **Multiple Task Types**: Support for sinusoid, linear, and quadratic regression tasks
- **Interactive Visualizations**: Plotly-based charts for training metrics and predictions
- **Configuration Management**: Dataclass-based configuration system
- **Checkpoint System**: Model checkpointing and experiment persistence

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/kryptologyst/MAML-Meta-Learning-Implementation.git
cd MAML-Meta-Learning-Implementation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Application

#### Option 1: Interactive Web UI (Recommended)
```bash
streamlit run 0151.py streamlit
```

#### Option 2: Command Line Training
```bash
python 0151.py
```

## Usage

### Web UI Dashboard

The Streamlit dashboard provides four main tabs:

1. **Training**: Configure hyperparameters and train the MAML model
2. **Evaluation**: Test the trained model on new tasks
3. **Database**: View saved tasks and experiment history
4. **Visualization**: Generate and visualize custom tasks

### Configuration Options

- **Meta Learning Rate**: Controls the outer loop optimization
- **Inner Learning Rate**: Controls task-specific adaptation speed
- **K-Shot**: Number of examples per task
- **Number of Tasks**: Tasks per training episode
- **Episodes**: Total training episodes
- **Inner Steps**: Gradient steps for task adaptation

### Task Types

The implementation supports three regression task types:

1. **Sinusoid**: `y = A * sin(x + phase)`
2. **Linear**: `y = slope * x + intercept`
3. **Quadratic**: `y = a * x² + b * x + c`

## Architecture

### Core Components

- **`MAMLModel`**: Neural network architecture with proper weight initialization
- **`MAMLTrainer`**: Training loop with metrics tracking and checkpointing
- **`TaskGenerator`**: Generates various types of regression tasks
- **`TaskDatabase`**: SQLite database for experiment persistence
- **`MAMLVisualizer`**: Interactive plotting utilities
- **`MAMLConfig`**: Configuration management with dataclasses

### Database Schema

The SQLite database includes two main tables:

- **`tasks`**: Stores individual task data and performance metrics
- **`experiments`**: Stores complete experiment configurations and results

## Metrics and Visualization

### Training Metrics

- Meta-loss progression
- Task performance over time
- Adaptation time per episode
- Loss comparison plots

### Evaluation Metrics

- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Prediction vs. target scatter plots

## 🔧 Development

### Code Structure

```
0151_Meta-learning_implementation/
├── 0151.py                 # Main implementation
├── requirements.txt        # Dependencies
├── README.md              # This file
├── maml_tasks.db          # SQLite database (created on first run)
├── checkpoints/           # Model checkpoints (created during training)
├── *.html                 # Generated visualization files
└── maml_training.log      # Training logs
```

### Adding New Task Types

To add a new task type:

1. Add a new method to `TaskGenerator` class
2. Update the task type selection in the UI
3. Add the task type to the evaluation methods

### Customizing the Model

The `MAMLModel` class can be easily extended:

```python
class CustomMAMLModel(MAMLModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Add custom layers or modifications
```

## Performance

The implementation includes several optimizations:

- GPU support with automatic device detection
- Efficient gradient computation with `create_graph=True`
- Proper weight initialization using Xavier uniform
- Memory-efficient checkpointing

## Testing

Run the test suite:

```bash
pytest tests/
```

## Logging

The application provides comprehensive logging:

- Training progress and metrics
- Database operations
- Error handling and debugging information
- Experiment tracking

Logs are written to both console and `maml_training.log` file.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is part of the "1000 AI Projects" series and is available under the MIT License.

## Acknowledgments

- Original MAML paper: Finn et al. (2017)
- PyTorch team for the excellent framework
- Streamlit team for the web UI framework
- Plotly team for interactive visualizations

## References

1. Finn, C., Abbeel, P., & Levine, S. (2017). Model-agnostic meta-learning for fast adaptation of deep networks. ICML.
2. PyTorch Documentation: https://pytorch.org/docs/
3. Streamlit Documentation: https://docs.streamlit.io/
4. Plotly Documentation: https://plotly.com/python/


# MAML-Meta-Learning-Implementation
