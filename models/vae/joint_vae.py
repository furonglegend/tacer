"""
Joint VAE Model for Graph Explanation Generation
Treats explanation as a differentiable latent variable for end-to-end training.
"""

import torch
import torch.nn as nn
from torch.distributions import Normal, kl_divergence
from transformers import T5EncoderModel, T5ForConditionalGeneration, T5Tokenizer


class JointVAE(nn.Module):
    """
    Joint Variational Autoencoder that generates explanations as latent variables.
    Combines classification and explanation generation in a unified probabilistic framework.
    """

    def __init__(self, t5_name='t5-small', z_dim=256, n_class=7, hidden_dim=768):
        super().__init__()
        self.z_dim = z_dim
        self.n_class = n_class
        self.hidden_dim = hidden_dim

        # Text encoder (T5 encoder)
        self.encoder = T5EncoderModel.from_pretrained(t5_name)
        encoder_hidden = self.encoder.config.d_model

        # Latent space projection layers
        self.fc_mu = nn.Linear(encoder_hidden, z_dim)
        self.fc_logvar = nn.Linear(encoder_hidden, z_dim)

        # Explanation decoder (T5 decoder)
        self.decoder = T5ForConditionalGeneration.from_pretrained(t5_name)
        self.tokenizer = T5Tokenizer.from_pretrained(t5_name)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, n_class)
        )

        # Projection from z to decoder input
        self.z_to_decoder = nn.Linear(z_dim, encoder_hidden)

    def encode(self, input_ids, attention_mask):
        """Encode input text to latent distribution parameters."""
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # Use mean pooling over sequence
        hidden = encoder_outputs.last_hidden_state
        pooled = (hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1, keepdim=True)

        mu = self.fc_mu(pooled)
        logvar = self.fc_logvar(pooled)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Reparameterization trick for VAE sampling."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z, labels=None):
        """Decode latent z to explanation text."""
        # Project z to decoder hidden dimension
        decoder_input = self.z_to_decoder(z).unsqueeze(1)

        if labels is not None:
            # Training mode: compute loss
            outputs = self.decoder(
                inputs_embeds=decoder_input,
                labels=labels
            )
            return outputs
        else:
            # Inference mode: generate text
            return decoder_input

    def forward(self, input_ids, attention_mask, labels=None, expl_ids=None, beta=1.0):
        """
        Forward pass combining encoding, sampling, decoding and classification.

        Args:
            input_ids: Tokenized input text [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            labels: Classification labels [batch]
            expl_ids: Target explanation token ids [batch, expl_len]
            beta: KL divergence weight for beta-VAE

        Returns:
            dict with loss, logits, and explanation logits
        """
        # Encode to latent space
        mu, logvar = self.encode(input_ids, attention_mask)

        # Sample z using reparameterization trick
        z = self.reparameterize(mu, logvar)

        # Classification from z
        logits = self.classifier(z)

        # Compute losses
        result = {'logits': logits, 'mu': mu, 'logvar': logvar, 'z': z}

        if labels is not None and expl_ids is not None:
            # Classification loss
            ce_loss = nn.CrossEntropyLoss()(logits, labels)

            # Reconstruction loss (explanation generation)
            decoder_outputs = self.decode(z, labels=expl_ids)
            recon_loss = decoder_outputs.loss

            # KL divergence loss
            q = Normal(mu, torch.exp(0.5 * logvar))
            p = Normal(torch.zeros_like(mu), torch.ones_like(logvar))
            kl_loss = kl_divergence(q, p).sum(-1).mean()

            # Combined loss
            total_loss = ce_loss + recon_loss + beta * kl_loss

            result.update({
                'loss': total_loss,
                'ce_loss': ce_loss,
                'recon_loss': recon_loss,
                'kl_loss': kl_loss,
                'expl_logits': decoder_outputs.logits
            })

        return result

    def generate_explanation(self, input_ids, attention_mask, max_length=128, num_beams=4):
        """
        Generate explanation text for given input.

        Args:
            input_ids: Tokenized input text
            attention_mask: Attention mask
            max_length: Maximum explanation length
            num_beams: Beam search width

        Returns:
            List of generated explanation strings
        """
        self.eval()
        with torch.no_grad():
            mu, logvar = self.encode(input_ids, attention_mask)
            z = mu  # Use mean for deterministic generation

            # Project z to decoder input
            encoder_outputs = self.z_to_decoder(z).unsqueeze(1)

            # Generate using T5 decoder
            generated_ids = self.decoder.generate(
                inputs_embeds=encoder_outputs,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

            # Decode to text
            explanations = self.tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )

        return explanations

    def sample_prior(self, n_samples, device):
        """Sample explanations from the prior distribution."""
        z = torch.randn(n_samples, self.z_dim).to(device)
        encoder_outputs = self.z_to_decoder(z).unsqueeze(1)

        generated_ids = self.decoder.generate(
            inputs_embeds=encoder_outputs,
            max_length=128,
            num_beams=4,
            early_stopping=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )

        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
