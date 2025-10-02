"""
Test suite for MAML Meta-Learning Implementation
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import os

# Import the main components
from 0151 import MAMLModel, MAMLConfig, TaskGenerator, TaskDatabase, MAMLTrainer

class TestMAMLModel:
    """Test cases for MAMLModel"""
    
    def test_model_initialization(self):
        """Test model initialization"""
        model = MAMLModel(input_size=1, hidden_size=20, output_size=1)
        assert model.input_size == 1
        assert model.hidden_size == 20
        assert model.output_size == 1
    
    def test_model_forward(self):
        """Test forward pass"""
        model = MAMLModel()
        x = torch.randn(10, 1)
        y = model(x)
        assert y.shape == (10, 1)
    
    def test_model_parameters(self):
        """Test parameter retrieval"""
        model = MAMLModel()
        params = model.get_parameters()
        assert len(params) == 6  # 3 layers * 2 parameters each (weight + bias)

class TestMAMLConfig:
    """Test cases for MAMLConfig"""
    
    def test_config_defaults(self):
        """Test default configuration values"""
        config = MAMLConfig()
        assert config.meta_lr == 0.001
        assert config.inner_lr == 0.01
        assert config.k_shot == 10
        assert config.num_tasks == 5
        assert config.episodes == 200
    
    def test_config_custom(self):
        """Test custom configuration"""
        config = MAMLConfig(meta_lr=0.01, episodes=100)
        assert config.meta_lr == 0.01
        assert config.episodes == 100

class TestTaskGenerator:
    """Test cases for TaskGenerator"""
    
    def test_sinusoid_task(self):
        """Test sinusoid task generation"""
        x, y = TaskGenerator.get_sinusoid_task(k=5)
        assert x.shape == (5, 1)
        assert y.shape == (5, 1)
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)
    
    def test_linear_task(self):
        """Test linear task generation"""
        x, y = TaskGenerator.get_linear_task(k=5)
        assert x.shape == (5, 1)
        assert y.shape == (5, 1)
    
    def test_quadratic_task(self):
        """Test quadratic task generation"""
        x, y = TaskGenerator.get_quadratic_task(k=5)
        assert x.shape == (5, 1)
        assert y.shape == (5, 1)

class TestTaskDatabase:
    """Test cases for TaskDatabase"""
    
    def test_database_initialization(self):
        """Test database initialization"""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db_path = tmp.name
        
        try:
            db = TaskDatabase(db_path)
            assert os.path.exists(db_path)
        finally:
            os.unlink(db_path)
    
    def test_save_and_retrieve_task(self):
        """Test saving and retrieving tasks"""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db_path = tmp.name
        
        try:
            db = TaskDatabase(db_path)
            
            # Save a task
            task_id = db.save_task("sinusoid", {"k": 10}, 0.5)
            assert task_id > 0
            
            # Retrieve tasks
            tasks = db.get_tasks()
            assert len(tasks) == 1
            assert tasks[0]['task_type'] == "sinusoid"
            assert tasks[0]['performance'] == 0.5
        finally:
            os.unlink(db_path)

class TestMAMLTrainer:
    """Test cases for MAMLTrainer"""
    
    def test_trainer_initialization(self):
        """Test trainer initialization"""
        config = MAMLConfig(episodes=2, num_tasks=2)
        model = MAMLModel()
        
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db_path = tmp.name
        
        try:
            database = TaskDatabase(db_path)
            trainer = MAMLTrainer(model, config, database)
            assert trainer.config == config
            assert trainer.model == model
        finally:
            os.unlink(db_path)
    
    def test_adapt_to_task(self):
        """Test task adaptation"""
        config = MAMLConfig()
        model = MAMLModel()
        
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db_path = tmp.name
        
        try:
            database = TaskDatabase(db_path)
            trainer = MAMLTrainer(model, config, database)
            
            # Generate test data
            x_train = torch.randn(5, 1)
            y_train = torch.randn(5, 1)
            
            # Test adaptation
            fast_weights = trainer._adapt_to_task(x_train, y_train)
            assert len(fast_weights) == 6  # 3 layers * 2 parameters each
        finally:
            os.unlink(db_path)

class TestIntegration:
    """Integration tests"""
    
    def test_end_to_end_training(self):
        """Test end-to-end training process"""
        config = MAMLConfig(episodes=2, num_tasks=2, log_interval=1)
        model = MAMLModel()
        
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db_path = tmp.name
        
        try:
            database = TaskDatabase(db_path)
            trainer = MAMLTrainer(model, config, database)
            
            # Run training
            results = trainer.train()
            
            assert 'experiment_id' in results
            assert 'training_time' in results
            assert 'final_meta_loss' in results
            assert 'metrics' in results
            assert len(results['metrics']['meta_losses']) == 2
        finally:
            os.unlink(db_path)
    
    def test_evaluation(self):
        """Test model evaluation"""
        config = MAMLConfig(episodes=1, num_tasks=1)
        model = MAMLModel()
        
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db_path = tmp.name
        
        try:
            database = TaskDatabase(db_path)
            trainer = MAMLTrainer(model, config, database)
            
            # Train briefly
            trainer.train()
            
            # Evaluate
            eval_result = trainer.evaluate_on_new_task("sinusoid", k_shot=5)
            
            assert 'task_id' in eval_result
            assert 'mse_loss' in eval_result
            assert 'mae_loss' in eval_result
            assert 'predictions' in eval_result
            assert 'targets' in eval_result
        finally:
            os.unlink(db_path)

if __name__ == "__main__":
    pytest.main([__file__])
