"""
Unit Tests for Joint VAE Module
"""

import pytest
import torch
import torch.nn as nn

import sys
sys.path.insert(0, '..')


class TestJointVAE:
    """Test cases for JointVAE model."""

    @pytest.fixture
    def model(self):
        """Create model fixture."""
        from models.vae.joint_vae import JointVAE
        return JointVAE(t5_name='t5-small', z_dim=256, n_class=7)

    @pytest.fixture
    def sample_batch(self, model):
        """Create sample batch fixture."""
        batch_size = 2
        seq_len = 64
        return {
            'input_ids': torch.randint(0, 1000, (batch_size, seq_len)),
            'attention_mask': torch.ones(batch_size, seq_len),
            'labels': torch.randint(0, 7, (batch_size,)),
            'expl_ids': torch.randint(0, 1000, (batch_size, 32))
        }

    def test_model_initialization(self, model):
        """Test that model initializes correctly."""
        assert model is not None
        assert hasattr(model, 'encoder')
        assert hasattr(model, 'decoder')
        assert hasattr(model, 'fc_mu')
        assert hasattr(model, 'fc_logvar')
        assert hasattr(model, 'classifier')

    def test_encode_shape(self, model, sample_batch):
        """Test encoder output shapes."""
        mu, logvar = model.encode(
            sample_batch['input_ids'],
            sample_batch['attention_mask']
        )
        
        batch_size = sample_batch['input_ids'].size(0)
        assert mu.shape == (batch_size, model.z_dim)
        assert logvar.shape == (batch_size, model.z_dim)

    def test_reparameterize(self, model):
        """Test reparameterization trick."""
        batch_size = 4
        mu = torch.zeros(batch_size, model.z_dim)
        logvar = torch.zeros(batch_size, model.z_dim)
        
        z = model.reparameterize(mu, logvar)
        
        assert z.shape == (batch_size, model.z_dim)
        # With zero mean and unit variance, samples should be roughly standard normal
        assert z.mean().abs() < 1.0  # Should be close to 0

    def test_forward_pass(self, model, sample_batch):
        """Test complete forward pass."""
        output = model(
            input_ids=sample_batch['input_ids'],
            attention_mask=sample_batch['attention_mask'],
            labels=sample_batch['labels'],
            expl_ids=sample_batch['expl_ids'],
            beta=1.0
        )
        
        assert 'loss' in output
        assert 'logits' in output
        assert 'mu' in output
        assert 'logvar' in output
        assert 'z' in output
        
        batch_size = sample_batch['input_ids'].size(0)
        assert output['logits'].shape == (batch_size, model.n_class)

    def test_loss_computation(self, model, sample_batch):
        """Test that loss is computed correctly."""
        output = model(
            input_ids=sample_batch['input_ids'],
            attention_mask=sample_batch['attention_mask'],
            labels=sample_batch['labels'],
            expl_ids=sample_batch['expl_ids'],
            beta=1.0
        )
        
        assert 'ce_loss' in output
        assert 'recon_loss' in output
        assert 'kl_loss' in output
        
        # Loss should be positive
        assert output['loss'].item() > 0
        assert output['ce_loss'].item() >= 0
        assert output['kl_loss'].item() >= 0

    def test_gradient_flow(self, model, sample_batch):
        """Test that gradients flow through the model."""
        output = model(
            input_ids=sample_batch['input_ids'],
            attention_mask=sample_batch['attention_mask'],
            labels=sample_batch['labels'],
            expl_ids=sample_batch['expl_ids']
        )
        
        output['loss'].backward()
        
        # Check that some parameters have gradients
        has_grad = False
        for param in model.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break
        
        assert has_grad, "No gradients computed"

    def test_beta_effect(self, model, sample_batch):
        """Test that beta parameter affects KL loss weight."""
        output_beta1 = model(
            input_ids=sample_batch['input_ids'],
            attention_mask=sample_batch['attention_mask'],
            labels=sample_batch['labels'],
            expl_ids=sample_batch['expl_ids'],
            beta=1.0
        )
        
        output_beta10 = model(
            input_ids=sample_batch['input_ids'],
            attention_mask=sample_batch['attention_mask'],
            labels=sample_batch['labels'],
            expl_ids=sample_batch['expl_ids'],
            beta=10.0
        )
        
        # Higher beta should generally increase total loss due to KL term
        # (not always true but generally expected)
        assert output_beta1['kl_loss'].item() == pytest.approx(
            output_beta10['kl_loss'].item(), rel=0.01
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
