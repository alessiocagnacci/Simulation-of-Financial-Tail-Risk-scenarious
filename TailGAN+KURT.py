import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import t as student_t, norm, kurtosis, skew, jarque_bera
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# =============================================================================
# Differentiable Neural Sorting (Grover et al., 2019)
# =============================================================================
def create_overlapping_sequences(data, seq_len=256, stride=10):
    n_seq = (len(data) - seq_len) // stride + 1
    sequences = np.zeros((n_seq, seq_len, data.shape[1]))
    for i in range(n_seq):
        sequences[i] = data[i*stride:i*stride+seq_len]
    return sequences


class DifferentiableSorting(nn.Module):
    """
    Differentiable sorting operator for ranking PnL samples.
    """

    def __init__(self, tau=1.0):
        super(DifferentiableSorting, self).__init__()
        self.tau = tau

    def forward(self, x):
        batch_size, n = x.shape
        device = x.device

        if n <= 1:
            return x

        x_expanded = x.unsqueeze(2)
        x_expanded_t = x.unsqueeze(1)
        B = torch.abs(x_expanded - x_expanded_t)

        indices = torch.arange(n, device=device).float()
        weights = (n + 1 - 2 * indices)

        Gamma = torch.zeros(batch_size, n, n, device=device)

        for i in range(n):
            logits = (weights[i] - B[:, i, :]) / self.tau
            Gamma[:, i, :] = torch.softmax(logits, dim=1)

        sorted_x = torch.bmm(Gamma, x.unsqueeze(2)).squeeze(2)

        return sorted_x


# =============================================================================
# TAIL-GAN Generator
# =============================================================================

class TailGANGenerator(nn.Module):
    """Generator network for TAIL-GAN."""

    def __init__(self, noise_dim, output_dim, hidden_dims=[500, 256, 512],
                 output_init_scale=0.02, max_return_scale=0.2):
        super(TailGANGenerator, self).__init__()

        self.max_return_scale = max_return_scale

        layers = []
        prev_dim = noise_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LeakyReLU(0.2))
            prev_dim = hidden_dim

        final_layer = nn.Linear(prev_dim, output_dim)
        layers.append(final_layer)
        self.network = nn.Sequential(*layers)

        with torch.no_grad():
            final_layer.weight.mul_(output_init_scale)
            final_layer.bias.mul_(output_init_scale)

    def forward(self, z):
        raw_output = self.network(z)
        return torch.tanh(raw_output) * self.max_return_scale


# =============================================================================
# TAIL-GAN Discriminator
# =============================================================================

class TailGANDiscriminator(nn.Module):
    """Discriminator network for TAIL-GAN."""

    def __init__(self, n_samples, hidden_dims=[500, 256, 128, 2], tau=1.0):
        super(TailGANDiscriminator, self).__init__()

        self.n_samples = max(n_samples, 2)
        self.sorting = DifferentiableSorting(tau=tau)

        layers = []
        prev_dim = self.n_samples

        for hidden_dim in hidden_dims[:-1]:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LeakyReLU(0.2))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, hidden_dims[-1]))
        self.network = nn.Sequential(*layers)

    def forward(self, pnl_samples):
        if pnl_samples.dim() == 1:
            pnl_samples = pnl_samples.unsqueeze(0)

        if pnl_samples.size(1) < 2:
            pad = torch.zeros(pnl_samples.size(0), 2 - pnl_samples.size(1), device=pnl_samples.device)
            pnl_samples = torch.cat([pnl_samples, pad], dim=1)

        sorted_pnl = self.sorting(pnl_samples)
        output = self.network(sorted_pnl)
        return output


# =============================================================================
# Early Stopping Class
# =============================================================================

class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(self, patience=20, min_delta=1e-4, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, val_loss, epoch):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            if self.verbose:
                print(f"  EarlyStopping: Initial best score: {score:.6f} at epoch {epoch}")
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose and self.counter % 5 == 0:
                print(f"  EarlyStopping: No improvement ({self.counter}/{self.patience})")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"  EarlyStopping: STOPPING at epoch {epoch} (best was {self.best_epoch})")
        else:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            if self.verbose:
                print(f"  EarlyStopping: Improvement to {score:.6f} at epoch {epoch}")

        return self.early_stop


# =============================================================================
# TAIL-GAN Model
# =============================================================================
class TailGAN:
    """TAIL-GAN: Generative Adversarial Network for Tail Risk Scenarios."""

    def __init__(self, n_assets, seq_len, n_strategies=10, alpha=0.05, lambda_gp=10.0,
                 noise_dim=200, hidden_dims_g=[200, 128, 256],
                 hidden_dims_d=[200, 128, 64, 1],
                 lr_g=5e-5, lr_d=1e-4, patience=20, min_delta=1e-4, tau=1.0,
                 strategy_seed=42, n_critic=5, output_init_scale=0.02,
                 max_return_scale=0.2, kurtosis_weight=0.0):

        self.M = n_assets
        self.T = seq_len
        self.data_dim = self.M * self.T

        self.n_strategies = max(n_strategies, 5)
        self.alpha = alpha
        self.lambda_gp = lambda_gp
        self.noise_dim = noise_dim
        self.tau = tau
        
        self.n_critic = n_critic
        self.kurtosis_weight = kurtosis_weight

        print(f"Inizializzazione TailGAN: Assets(M)={self.M}, TimeSteps(T)={self.T} -> DataDim={self.data_dim}")

        self.generator = TailGANGenerator(noise_dim, self.data_dim, hidden_dims_g,
                                           output_init_scale=output_init_scale,
                                           max_return_scale=max_return_scale)
        self.discriminator = TailGANDiscriminator(self.T, hidden_dims_d, tau=tau)

        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=lr_g, betas=(0.5, 0.9))
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=lr_d, betas=(0.5, 0.9))

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.generator.to(self.device)
        self.discriminator.to(self.device)

        gen_state = torch.get_rng_state()
        torch.manual_seed(strategy_seed)
        self.strategy_weights = torch.randn(self.n_strategies, self.M, device=self.device)
        self.strategy_weights = self.strategy_weights / (
            self.strategy_weights.abs().sum(dim=1, keepdim=True) + 1e-8
        )
        torch.set_rng_state(gen_state)

        self.history = {
            'g_loss': [], 'd_loss': [],
            'vaR_error': [], 'es_error': [],
            'total_error': [], 'kurtosis_penalty': []
        }

        self.early_stopping = EarlyStopping(patience=patience, min_delta=min_delta, verbose=True)
        self.best_generator_state = None
        self.best_discriminator_state = None

    def _gradient_penalty(self, real_pnl, fake_pnl):
        batch_size = real_pnl.size(0)

        alpha = torch.rand(batch_size, 1, device=self.device)
        alpha = alpha.expand_as(real_pnl)

        interpolated = alpha * real_pnl + (1 - alpha) * fake_pnl
        interpolated.requires_grad_(True)

        d_interpolated = self.discriminator(interpolated)

        grad = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True,
            retain_graph=True
        )[0]

        grad_norm = grad.view(batch_size, -1).norm(2, dim=1)
        gp = ((grad_norm - 1) ** 2).mean()

        return gp

    def _compute_strategy_pnls(self, price_scenarios):
        
        batch_size = price_scenarios.size(0)
        returns = price_scenarios.view(batch_size, self.T, self.M)

        pnls = torch.zeros(batch_size, self.n_strategies, self.T, device=self.device)

        for k in range(self.n_strategies):
            weights = self.strategy_weights[k]
            portfolio_returns = torch.einsum('btm,m->bt', returns, weights)
            pnls[:, k, :] = portfolio_returns

        return pnls

    def _compute_validation_error(self, val_data):
        if len(val_data) == 0:
            return float('inf')

        n_samples = min(100, len(val_data))
        generated = self.generate(n_samples)

        real_returns = val_data.flatten()
        gen_returns = generated.flatten()

        sorted_real = np.sort(real_returns)
        idx = max(1, int(self.alpha * len(sorted_real)))
        vaR_real = sorted_real[idx]
        es_real = np.mean(sorted_real[:idx])

        sorted_gen = np.sort(gen_returns)
        idx = max(1, int(self.alpha * len(sorted_gen)))
        vaR_gen = sorted_gen[idx]
        es_gen = np.mean(sorted_gen[:idx])

        vaR_error = abs(vaR_gen - vaR_real) / (abs(vaR_real) + 1e-8)
        es_error = abs(es_gen - es_real) / (abs(es_real) + 1e-8)

        return (vaR_error + es_error) / 2

    def _compute_kurtosis_penalty(self, fake_pnl, real_pnl):
        
        fake_flat = fake_pnl.reshape(-1)
        real_flat = real_pnl.reshape(-1).detach()

        fake_mean = fake_flat.mean()
        fake_std = fake_flat.std() + 1e-8
        fake_kurt = ((fake_flat - fake_mean) / fake_std).pow(4).mean() - 3.0

        real_mean = real_flat.mean()
        real_std = real_flat.std() + 1e-8
        real_kurt = ((real_flat - real_mean) / real_std).pow(4).mean() - 3.0

        return (fake_kurt - real_kurt) ** 2

    def train_step(self, real_prices):
        batch_size = real_prices.size(0)
        real_prices = real_prices.to(self.device)

       
        strategy_idx = np.random.randint(self.n_strategies)

        # ---------------------
        # 1. Train Discriminator (n_critic volte, come nello script WA-GAN)
        # ---------------------
        d_loss_val = None
        fake_pnl = None
        for _ in range(self.n_critic):
            self.optimizer_D.zero_grad()

            z = student_t.rvs(5, size=(batch_size, self.noise_dim))
            z = torch.tensor(z, dtype=torch.float32).to(self.device)
            fake_prices = self.generator(z)

            real_pnls = self._compute_strategy_pnls(real_prices)
            fake_pnls = self._compute_strategy_pnls(fake_prices)

            real_pnl = real_pnls[:, strategy_idx, :]
            fake_pnl = fake_pnls[:, strategy_idx, :]

            d_real = self.discriminator(real_pnl)
            d_fake = self.discriminator(fake_pnl.detach())

            gp = self._gradient_penalty(real_pnl, fake_pnl.detach())
            d_loss = -d_real.mean() + d_fake.mean() + self.lambda_gp * gp

            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1.0)
            self.optimizer_D.step()
            d_loss_val = d_loss

        # ---------------------
        # 2. Train Generator (una sola volta)
        # ---------------------
        self.optimizer_G.zero_grad()

        z = student_t.rvs(5, size=(batch_size, self.noise_dim))
        z = torch.tensor(z, dtype=torch.float32).to(self.device)
        fake_prices = self.generator(z)
        fake_pnls = self._compute_strategy_pnls(fake_prices)
        fake_pnl = fake_pnls[:, strategy_idx, :]

        d_fake_for_G = self.discriminator(fake_pnl)
        adv_loss = -d_fake_for_G.mean()

        
        kurt_penalty = self._compute_kurtosis_penalty(fake_pnl, real_pnl)
        g_loss = adv_loss + self.kurtosis_weight * kurt_penalty

        g_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 1.0)
        self.optimizer_G.step()

        # ---------------------
        # 3. Metriche di Errore
        # ---------------------
        with torch.no_grad():
            real_pnls = self._compute_strategy_pnls(real_prices)
            real_pnl = real_pnls[:, strategy_idx, :]

            real_sorted, _ = torch.sort(real_pnl, dim=1)
            idx = max(1, min(int(self.alpha * self.T), self.T - 1))

            vaR_true = real_sorted[:, idx].mean()
            es_true = real_sorted[:, :idx].mean()

            fake_sorted, _ = torch.sort(fake_pnl, dim=1)
            vaR_est = fake_sorted[:, idx].mean()
            es_est = fake_sorted[:, :idx].mean()

            vaR_error = torch.abs(vaR_est - vaR_true) / (torch.abs(vaR_true) + 1e-8)
            es_error = torch.abs(es_est - es_true) / (torch.abs(es_true) + 1e-8)
            total_error = (vaR_error + es_error) / 2

        return {
            'd_loss': d_loss_val.item(),
            'g_loss': g_loss.item(),
            'vaR_error': vaR_error.item(),
            'es_error': es_error.item(),
            'total_error': total_error.item(),
            'kurtosis_penalty': kurt_penalty.item()
        }

    def fit(self, train_data, val_data=None, epochs=1000, batch_size=32, verbose=100):
        train_data = torch.tensor(train_data, dtype=torch.float32)
        dataset = TensorDataset(train_data)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        if val_data is None:
            val_data = train_data[:min(300, len(train_data))]
        else:
            val_data = torch.tensor(val_data, dtype=torch.float32)

        print(f"\n{'='*60}")
        print(f"TRAINING TAIL-GAN")
        print(f"{'='*60}")
        print(f"Training samples: {len(train_data)}")
        print(f"Validation samples: {len(val_data)}")
        print(f"Data dimension: {self.data_dim}")
        print(f"Assets: {self.M}, Time steps: {self.T}")
        print(f"Confidence level: {self.alpha}")
        print(f"Patience: {self.early_stopping.patience} epochs")
        print(f"{'='*60}\n")

        best_val_error = float('inf')
        best_epoch = 0

        for epoch in range(epochs):
            epoch_losses = {'d_loss': [], 'g_loss': [],
                           'vaR_error': [], 'es_error': [], 'total_error': [],
                           'kurtosis_penalty': []}

            for batch in dataloader:
                real_prices = batch[0]
                losses = self.train_step(real_prices)

                for key in epoch_losses:
                    epoch_losses[key].append(losses.get(key, 0))

            avg_losses = {key: np.mean(values) for key, values in epoch_losses.items()}

            for key in self.history:
                self.history[key].append(avg_losses.get(key, 0))

            if (epoch + 1) % 10 == 0:
                val_error = self._compute_validation_error(val_data)

                if val_error < best_val_error - self.early_stopping.min_delta:
                    best_val_error = val_error
                    best_epoch = epoch + 1
                    self.best_generator_state = {k: v.cpu().clone() for k, v in self.generator.state_dict().items()}
                    self.best_discriminator_state = {k: v.cpu().clone() for k, v in self.discriminator.state_dict().items()}

                if (epoch + 1) - best_epoch >= self.early_stopping.patience:
                    print(f"\nEarly stopping triggered at epoch {epoch+1}")
                    print(f"Best validation error: {best_val_error:.6f} at epoch {best_epoch}")
                    break

            if (epoch + 1) % verbose == 0:
                print(f"Epoch {epoch+1}/{epochs}: "
                      f"D_loss: {avg_losses['d_loss']:.4f}, "
                      f"G_loss: {avg_losses['g_loss']:.4f}, "
                      f"VaR Error: {avg_losses['vaR_error']:.4f}, "
                      f"ES Error: {avg_losses['es_error']:.4f}, "
                      f"Kurt Penalty: {avg_losses.get('kurtosis_penalty', 0):.4f}")

        if self.best_generator_state is not None:
            print(f"\nRestoring best model from epoch {best_epoch}")
            self.generator.load_state_dict({k: v.to(self.device) for k, v in self.best_generator_state.items()})
            self.discriminator.load_state_dict({k: v.to(self.device) for k, v in self.best_discriminator_state.items()})

        print(f"\nTraining completed. Best validation error: {best_val_error:.6f}")
        print(f"Best epoch: {best_epoch}")

    def generate(self, n_samples):
        self.generator.eval()
        with torch.no_grad():
            z = student_t.rvs(5, size=(n_samples, self.noise_dim))
            z = torch.tensor(z, dtype=torch.float32).to(self.device)
            generated = self.generator(z).cpu().numpy()
        return generated


# =============================================================================
# Kurtosis Tests and Statistical Analysis
# =============================================================================

class KurtosisAnalyzer:
    """
    Analisi della kurtosi per valutare la pesantezza delle code
    delle distribuzioni generate vs reali.
    """

    @staticmethod
    def compute_kurtosis(data):
        return kurtosis(data, fisher=True)

    @staticmethod
    def compute_skewness(data):
        return skew(data)

    @staticmethod
    def compute_jarque_bera(data):
        return jarque_bera(data)

    @staticmethod
    def analyze_kurtosis(real_data, generated_data, alpha_levels=[0.01, 0.05, 0.10]):
        real_flat = real_data.flatten()
        gen_flat = generated_data.flatten()

        results = {
            'real': {},
            'generated': {},
            'comparison': {},
            'tail_tests': {}
        }

        results['real']['kurtosis'] = KurtosisAnalyzer.compute_kurtosis(real_flat)
        results['real']['skewness'] = KurtosisAnalyzer.compute_skewness(real_flat)
        results['real']['mean'] = np.mean(real_flat)
        results['real']['std'] = np.std(real_flat)
        results['real']['min'] = np.min(real_flat)
        results['real']['max'] = np.max(real_flat)

        results['generated']['kurtosis'] = KurtosisAnalyzer.compute_kurtosis(gen_flat)
        results['generated']['skewness'] = KurtosisAnalyzer.compute_skewness(gen_flat)
        results['generated']['mean'] = np.mean(gen_flat)
        results['generated']['std'] = np.std(gen_flat)
        results['generated']['min'] = np.min(gen_flat)
        results['generated']['max'] = np.max(gen_flat)

        results['comparison']['kurtosis_ratio'] = results['generated']['kurtosis'] / (results['real']['kurtosis'] + 1e-10)
        results['comparison']['kurtosis_diff'] = abs(results['generated']['kurtosis'] - results['real']['kurtosis'])
        results['comparison']['kurtosis_relative_diff'] = results['comparison']['kurtosis_diff'] / (abs(results['real']['kurtosis']) + 1e-10)

        jb_real = KurtosisAnalyzer.compute_jarque_bera(real_flat)
        jb_gen = KurtosisAnalyzer.compute_jarque_bera(gen_flat)

        results['tail_tests']['jarque_bera'] = {
            'real': {'statistic': jb_real[0], 'p_value': jb_real[1]},
            'generated': {'statistic': jb_gen[0], 'p_value': jb_gen[1]}
        }

        if results['real']['kurtosis'] > 0:
            results['real']['kurtosis_interpretation'] = "Leptocurtica (code pesanti)"
        elif results['real']['kurtosis'] < 0:
            results['real']['kurtosis_interpretation'] = "Platicurtica (code leggere)"
        else:
            results['real']['kurtosis_interpretation'] = "Mesocurtica (normale)"

        if results['generated']['kurtosis'] > 0:
            results['generated']['kurtosis_interpretation'] = "Leptocurtica (code pesanti)"
        elif results['generated']['kurtosis'] < 0:
            results['generated']['kurtosis_interpretation'] = "Platicurtica (code leggere)"
        else:
            results['generated']['kurtosis_interpretation'] = "Mesocurtica (normale)"

        for alpha in alpha_levels:
            vaR_real = np.percentile(real_flat, alpha * 100)
            es_real = np.mean(real_flat[real_flat <= vaR_real])

            vaR_gen = np.percentile(gen_flat, alpha * 100)
            es_gen = np.mean(gen_flat[gen_flat <= vaR_gen])

            results['tail_tests'][f'VaR_{alpha:.2f}'] = {
                'real': vaR_real,
                'generated': vaR_gen,
                'relative_error': abs(vaR_gen - vaR_real) / (abs(vaR_real) + 1e-10)
            }

            results['tail_tests'][f'ES_{alpha:.2f}'] = {
                'real': es_real,
                'generated': es_gen,
                'relative_error': abs(es_gen - es_real) / (abs(es_real) + 1e-10)
            }

        return results

    @staticmethod
    def print_kurtosis_report(results):
        print("\n" + "=" * 80)
        print("KURTOSIS AND TAIL ANALYSIS REPORT")
        print("=" * 80)

        print("\n" + "-" * 40)
        print("REAL DATA STATISTICS")
        print("-" * 40)
        print(f"Mean: {results['real']['mean']:.6f}")
        print(f"Std Dev: {results['real']['std']:.6f}")
        print(f"Min: {results['real']['min']:.6f}")
        print(f"Max: {results['real']['max']:.6f}")
        print(f"Excess Kurtosis: {results['real']['kurtosis']:.4f}")
        print(f"Interpretation: {results['real']['kurtosis_interpretation']}")
        print(f"Skewness: {results['real']['skewness']:.4f}")

        print("\n" + "-" * 40)
        print("GENERATED DATA STATISTICS")
        print("-" * 40)
        print(f"Mean: {results['generated']['mean']:.6f}")
        print(f"Std Dev: {results['generated']['std']:.6f}")
        print(f"Min: {results['generated']['min']:.6f}")
        print(f"Max: {results['generated']['max']:.6f}")
        print(f"Excess Kurtosis: {results['generated']['kurtosis']:.4f}")
        print(f"Interpretation: {results['generated']['kurtosis_interpretation']}")
        print(f"Skewness: {results['generated']['skewness']:.4f}")

        print("\n" + "-" * 40)
        print("KURTOSIS COMPARISON")
        print("-" * 40)
        print(f"Real Kurtosis: {results['real']['kurtosis']:.4f}")
        print(f"Generated Kurtosis: {results['generated']['kurtosis']:.4f}")
        print(f"Absolute Difference: {results['comparison']['kurtosis_diff']:.4f}")
        print(f"Relative Difference: {results['comparison']['kurtosis_relative_diff']:.2%}")
        print(f"Kurtosis Ratio (Gen/Real): {results['comparison']['kurtosis_ratio']:.4f}")

        print("\n" + "-" * 40)
        print("JARQUE-BERA TEST (Normalità)")
        print("-" * 40)
        print(f"Real Data - Statistic: {results['tail_tests']['jarque_bera']['real']['statistic']:.4f}, "
              f"p-value: {results['tail_tests']['jarque_bera']['real']['p_value']:.6f}")
        print(f"Generated Data - Statistic: {results['tail_tests']['jarque_bera']['generated']['statistic']:.4f}, "
              f"p-value: {results['tail_tests']['jarque_bera']['generated']['p_value']:.6f}")

        print("\n" + "-" * 40)
        print("TAIL RISK COMPARISON (VaR & ES)")
        print("-" * 40)
        for key in results['tail_tests']:
            if key.startswith('VaR_') or key.startswith('ES_'):
                print(f"\n{key}:")
                print(f"  Real: {results['tail_tests'][key]['real']:.6f}")
                print(f"  Generated: {results['tail_tests'][key]['generated']:.6f}")
                print(f"  Relative Error: {results['tail_tests'][key]['relative_error']:.2%}")

        print("\n" + "=" * 80)

        print("\nFINAL INTERPRETATION:")
        kurt_diff = results['comparison']['kurtosis_relative_diff']
        if kurt_diff < 0.1:
            print("✓ Ottima corrispondenza della kurtosi (< 10% differenza)")
        elif kurt_diff < 0.25:
            print("✓ Buona corrispondenza della kurtosi (< 25% differenza)")
        elif kurt_diff < 0.5:
            print("⚠ Moderata discrepanza nella kurtosi (25-50% differenza)")
        else:
            print("✗ Significativa discrepanza nella kurtosi (> 50% differenza)")

        if results['real']['kurtosis'] > 0 and results['generated']['kurtosis'] > 0:
            print("✓ Entrambe le distribuzioni hanno code pesanti (leptocurtiche)")
        elif results['real']['kurtosis'] < 0 and results['generated']['kurtosis'] < 0:
            print("✓ Entrambe le distribuzioni hanno code leggere (platicurtiche)")
        elif abs(results['real']['kurtosis'] - results['generated']['kurtosis']) < 0.5:
            print("✓ Kurtosi simile (differenza < 0.5)")
        else:
            print("✗ Differenza significativa nella kurtosi")

        print("=" * 80)


# =============================================================================
# Visualization Functions for Kurtosis
# =============================================================================

def plot_kurtosis_comparison(real_data, generated_data, results, save_path=None):
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    real_flat = real_data.flatten()
    gen_flat = generated_data.flatten()

    ax = axes[0, 0]
    ax.hist(real_flat, bins=50, alpha=0.5, label='Real', density=True, color='blue')
    ax.hist(gen_flat, bins=50, alpha=0.5, label='Generated', density=True, color='orange')
    ax.set_title('Distribution Comparison', fontsize=14)
    ax.set_xlabel('Returns')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    sorted_real = np.sort(real_flat)
    sorted_gen = np.sort(gen_flat)
    min_len = min(len(sorted_real), len(sorted_gen))
    ax.scatter(sorted_real[:min_len], sorted_gen[:min_len], alpha=0.3, s=10)
    ax.plot([min(sorted_real), max(sorted_real)],
            [min(sorted_real), max(sorted_real)], 'r--', linewidth=2, label='Identity')
    ax.set_title('QQ Plot - Generated vs Real', fontsize=14)
    ax.set_xlabel('Real Quantiles')
    ax.set_ylabel('Generated Quantiles')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    kurtosis_values = [results['real']['kurtosis'], results['generated']['kurtosis']]
    bars = ax.bar(['Real', 'Generated'], kurtosis_values, color=['blue', 'orange'], alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.set_title('Excess Kurtosis Comparison', fontsize=14)
    ax.set_ylabel('Excess Kurtosis')
    ax.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars, kurtosis_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1 * abs(val) + 0.1,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)

    ax = axes[1, 1]
    real_sorted_desc = np.sort(real_flat)[::-1]
    gen_sorted_desc = np.sort(gen_flat)[::-1]

    tail_size = int(0.2 * len(real_sorted_desc))
    ax.plot(real_sorted_desc[:tail_size], 'b-', label='Real', linewidth=2)
    ax.plot(gen_sorted_desc[:tail_size], 'r-', label='Generated', linewidth=2)
    ax.set_title('Tail Comparison (Top 20%)', fontsize=14)
    ax.set_xlabel('Rank in Tail')
    ax.set_ylabel('Return')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# =============================================================================
# Data Loader
# =============================================================================
def load_multi_asset_data(file_paths, price_col='Close', date_col='Date'):
    
    print(f"Loading and merging {len(file_paths)} assets by date...")

    price_dfs = []
    for name, path in file_paths.items():
        print(f"Loading: {name}")
        df = pd.read_csv(path)

        if date_col not in df.columns:
            raise ValueError(f"Column '{date_col}' not found in {path}")
        if price_col not in df.columns:
            raise ValueError(f"Column '{price_col}' not found in {path}")

        sub = df[[date_col, price_col]].rename(columns={price_col: name})
        sub[date_col] = pd.to_datetime(sub[date_col])
        print(f"  -> {name}: {len(sub)} righe di prezzo, range date "
              f"{sub[date_col].min().date()} - {sub[date_col].max().date()}")
        price_dfs.append(sub)

    merged = price_dfs[0]
    for sub in price_dfs[1:]:
        merged = pd.merge(merged, sub, on=date_col, how='inner')

    merged = merged.sort_values(date_col).reset_index(drop=True)

    print(f"\nRighe dopo inner join sulle date comuni: {len(merged)}")
    print(f"Date comuni: {merged[date_col].min().date()} - {merged[date_col].max().date()}")
    for name, path in file_paths.items():
        pass  # nota gia' stampata sopra per singolo asset

    asset_names = list(file_paths.keys())
    prices = merged[asset_names].values

    log_returns = np.diff(np.log(prices + 1e-10), axis=0)
    mask = ~np.any(np.isnan(log_returns) | np.isinf(log_returns), axis=1)
    log_returns = log_returns[mask]

    print(f"\nLog-return finali (allineati per data): {log_returns.shape}")

    return log_returns, asset_names


# =============================================================================
# Performance Evaluation
# =============================================================================

class TailGANEvaluator:
    """Evaluator for TAIL-GAN performance."""

    @staticmethod
    def evaluate_tail_risk(generated_data, real_data, alpha=0.05):
        if len(generated_data) == 0 or len(real_data) == 0:
            return {'vaR_error': 0, 'es_error': 0, 'total_error': 0}

        real_returns = real_data.flatten()
        gen_returns = generated_data.flatten()

        sorted_real = np.sort(real_returns)
        idx = max(1, int(alpha * len(sorted_real)))
        vaR_real = sorted_real[idx]
        es_real = np.mean(sorted_real[:idx])

        sorted_gen = np.sort(gen_returns)
        idx = max(1, int(alpha * len(sorted_gen)))
        vaR_gen = sorted_gen[idx]
        es_gen = np.mean(sorted_gen[:idx])

        vaR_error = abs(vaR_gen - vaR_real) / (abs(vaR_real) + 1e-8)
        es_error = abs(es_gen - es_real) / (abs(es_real) + 1e-8)

        return {
            'vaR_error': vaR_error,
            'es_error': es_error,
            'total_error': (vaR_error + es_error) / 2
        }


# =============================================================================
# Visualization Functions
# =============================================================================

def plot_training_history(history, save_path=None):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(history['d_loss'], linewidth=2)
    axes[0, 0].set_title('Discriminator Loss', fontsize=14)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(history['g_loss'], linewidth=2, color='orange')
    axes[0, 1].set_title('Generator Loss', fontsize=14)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(history['vaR_error'], linewidth=2, color='green', label='VaR')
    axes[1, 0].plot(history['es_error'], linewidth=2, color='red', label='ES')
    axes[1, 0].set_title('VaR and ES Errors', fontsize=14)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Error')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(history['total_error'], linewidth=2, color='purple', label='Total Error')
    if 'kurtosis_penalty' in history and len(history['kurtosis_penalty']) > 0:
        ax2 = axes[1, 1].twinx()
        ax2.plot(history['kurtosis_penalty'], linewidth=2, color='brown', label='Kurtosis Penalty')
        ax2.set_ylabel('Kurtosis Penalty', color='brown')
        ax2.tick_params(axis='y', labelcolor='brown')
    axes[1, 1].set_title('Total Error & Kurtosis Penalty', fontsize=14)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Error', color='purple')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_rank_frequency(real_data, generated_data, save_path=None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    real_returns = real_data.flatten()
    gen_returns = generated_data.flatten()

    real_sorted = np.sort(real_returns)[::-1]
    gen_sorted = np.sort(gen_returns)[::-1]

    for idx, strategy_type in enumerate(['Static', 'Mean-Reversion', 'Trend-Following']):
        ax = axes[idx]
        ax.plot(real_sorted, 'b-', label='Market', linewidth=2)
        ax.plot(gen_sorted, 'r-', label='TAIL-GAN', linewidth=2)
        ax.set_title(f'{strategy_type} Strategy', fontsize=12)
        ax.set_xlabel('Rank')
        ax.set_ylabel('PnL')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        ax.set_xscale('log')

    plt.suptitle('Rank-Frequency Distribution', fontsize=16)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# =============================================================================
# Main Execution
# =============================================================================

def main():
    """Main function for TAIL-GAN analysis with kurtosis test."""
    print("=" * 70)
    print("TAIL-GAN: Tail Risk Scenario Generation WITH KURTOSIS TEST")
    print("=" * 70)

    print("\n" + "=" * 60)
    print("LOADING DATA")
    print("=" * 60)

    file_dict = {
        "NKX": "/content/^nkx_d-2.csv",
        "NDX": "/content/^ndx_d copia.csv",
        "HSI": "/content/^hsi_d.csv",
        "FTSEmib": "/content/^fmib_d.csv",
    }
    try:
        returns, asset_names = load_multi_asset_data(
            file_dict,
            price_col='Close',
            date_col='Date'
        )
    except FileNotFoundError:
        print("\nGenerating synthetic data for demonstration...")
        np.random.seed(42)
        n_samples = 2000
        data_dim = 4
        returns = student_t.rvs(3, size=(n_samples, data_dim)) * 0.02
        asset_names = ['Asset_1', 'Asset_2', 'Asset_3', 'Asset_4']

    print(f"\nData shape: {returns.shape}")
    print(f"Assets: {asset_names}")

    seq_len = 256
    stride = 10
    n_assets = returns.shape[1]

    if len(returns) < seq_len:
        print(f"\n ERRORE: Hai solo {len(returns)} giorni di dati, ma hai richiesto finestre da {seq_len} giorni!")
        print("Usa dataset più storici, oppure abbassa seq_len (es. seq_len=64).")
        return None, None, None

    sequences = create_overlapping_sequences(returns, seq_len=seq_len, stride=stride)
    flat_sequences = sequences.reshape(len(sequences), -1)

    print(f"Dataset finale (finestre temporali): {flat_sequences.shape}")

    n_train = int(0.6 * len(flat_sequences))
    n_val = int(0.2 * len(flat_sequences))

    train_data = flat_sequences[:n_train]
    val_data = flat_sequences[n_train:n_train+n_val]
    test_data = flat_sequences[n_train+n_val:]

    print(f"Training set: {len(train_data)} campioni, Validation: {len(val_data)}, Test: {len(test_data)}")

    if len(train_data) == 0:
        print("\n ERRORE: Il set di training è vuoto! Abbassa lo 'stride' o aumenta i dati.")
        return None, None, None

    print(f"Training: {len(train_data)}, Validation: {len(val_data)}, Test: {len(test_data)}")

    tailgan = TailGAN(
        n_assets=n_assets,
        seq_len=seq_len,
        n_strategies=10,
        alpha=0.05,
        noise_dim=200,
        hidden_dims_g=[200, 128, 256],
        hidden_dims_d=[200, 128, 64, 1],
        lr_g=5e-5,
        lr_d=1e-4,
        n_critic=5,
        max_return_scale=0.2,
        kurtosis_weight=0.3,  
        patience=300,
        min_delta=1e-4
    )

    tailgan.fit(
        train_data=train_data,
        val_data=val_data,
        epochs=5000,
        batch_size=128,
        verbose=1
    )

    print("\n" + "=" * 60)
    print("GENERATING SCENARIOS")
    print("=" * 60)

    n_generate = 500
    generated_scenarios = tailgan.generate(n_generate)

    print(f"Generated {len(generated_scenarios)} scenarios")

    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)

    evaluator = TailGANEvaluator()
    results = evaluator.evaluate_tail_risk(generated_scenarios, test_data, alpha=0.05)

    print(f"\nVaR Relative Error: {results['vaR_error']:.4f}")
    print(f"ES Relative Error: {results['es_error']:.4f}")
    print(f"Total Relative Error: {results['total_error']:.4f}")

    print("\n" + "=" * 60)
    print("KURTOSIS ANALYSIS")
    print("=" * 60)

    kurtosis_results = KurtosisAnalyzer.analyze_kurtosis(
        test_data,
        generated_scenarios,
        alpha_levels=[0.01, 0.05, 0.10]
    )

    KurtosisAnalyzer.print_kurtosis_report(kurtosis_results)

    print("\n" + "=" * 60)
    print("VISUALIZATIONS")
    print("=" * 60)

    plot_training_history(tailgan.history)
    plot_rank_frequency(test_data, generated_scenarios)
    plot_kurtosis_comparison(test_data, generated_scenarios, kurtosis_results)

    print("\n" + "=" * 70)
    print("TAIL-GAN ANALYSIS COMPLETE!")
    print("=" * 70)

    print("\nFINAL SUMMARY:")
    print(f"  - Assets: {len(asset_names)}")
    print(f"  - Data dimension: {returns.shape[1]}")
    print(f"  - Training samples: {len(train_data)}")
    print(f"  - Test samples: {len(test_data)}")
    print(f"  - Generated scenarios: {n_generate}")
    print(f"  - VaR Error: {results['vaR_error']:.2%}")
    print(f"  - ES Error: {results['es_error']:.2%}")
    print(f"  - Real Kurtosis: {kurtosis_results['real']['kurtosis']:.4f}")
    print(f"  - Generated Kurtosis: {kurtosis_results['generated']['kurtosis']:.4f}")
    print(f"  - Kurtosis Difference: {kurtosis_results['comparison']['kurtosis_diff']:.4f}")

    return tailgan, generated_scenarios, kurtosis_results


if __name__ == "__main__":
    tailgan, scenarios, kurtosis_results = main()