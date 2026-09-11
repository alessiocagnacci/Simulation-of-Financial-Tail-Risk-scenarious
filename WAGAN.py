import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import norm, pareto, kurtosis, skew, t as student_t, genpareto
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# =============================================================================
# Aitchison Geometry Utilities - CON GESTIONE d=1
# =============================================================================

class AitchisonTransform:
    """Implements Aitchison geometry operations for compositional data analysis."""

    def __init__(self, d):
        self.d = d
        if d == 1:
            self.e_basis = np.array([]).reshape(0, 1)
            self.M = np.array([]).reshape(0, 1)
        else:
            self.H = np.eye(d) - np.ones((d, d)) / d
            self.e_basis = self._build_orthonormal_basis(d)
            self.M = self.e_basis.T

    def _build_orthonormal_basis(self, d):
        basis = []
        for i in range(1, d):
            e_i = np.zeros(d)
            e_i[:i-1] = np.sqrt(i / (i + 1)) * (i - 1)
            if i - 1 < d:
                e_i[i-1] = -np.sqrt(i / (i + 1))
            basis.append(e_i)
        return np.array(basis).T

    def clr(self, x):
        x = np.array(x)
        if self.d == 1:
            return np.array([0.0])
        log_x = np.log(x)
        g = np.exp(np.mean(log_x))
        return log_x - np.log(g)

    def clr_inv(self, x):
        if self.d == 1:
            return np.array([1.0])
        x = np.array(x)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x)

    def to_coordinates(self, x):
        if self.d == 1:
            return np.array([0.0])
        clr_x = self.clr(x)
        return np.dot(self.M, clr_x)

    def from_coordinates(self, coords):
        if self.d == 1:
            return np.array([1.0])
        h_point = np.dot(self.e_basis, coords)
        return self.clr_inv(h_point)


# =============================================================================
# Neural Network Architectures
# =============================================================================

class HeavyTailGenerator(nn.Module):
    """Generator network with support for heavy-tailed latent distributions."""

    def __init__(self, latent_dim, output_dim, hidden_layers=4, hidden_size=256,
                 latent_dist='student', df=2.5):
        super(HeavyTailGenerator, self).__init__()

        self.latent_dim = latent_dim
        self.latent_dist = latent_dist
        self.df = df

        layers = []
        prev_size = latent_dim

        for i in range(hidden_layers):
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.LeakyReLU(0.01))
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, z):
        return self.network(z)

    def sample_latent(self, batch_size):
        """Sample from heavy-tailed latent distribution."""
        if self.latent_dist == 'student':
            z = student_t.rvs(self.df, size=(batch_size, self.latent_dim))
            return torch.tensor(z, dtype=torch.float32)
        elif self.latent_dist == 'cauchy':
            z = student_t.rvs(1, size=(batch_size, self.latent_dim))
            return torch.tensor(z, dtype=torch.float32)
        elif self.latent_dist == 'laplace':
            u = np.random.uniform(-0.5, 0.5, (batch_size, self.latent_dim))
            z = -np.sign(u) * np.log(1 - 2 * np.abs(u))
            return torch.tensor(z, dtype=torch.float32)
        else:
            z = np.random.randn(batch_size, self.latent_dim)
            return torch.tensor(z, dtype=torch.float32)


class HeavyTailCritic(nn.Module):
    """Critic network with improved gradient handling for heavy tails."""

    def __init__(self, input_dim, hidden_layers=4, hidden_size=256):
        super(HeavyTailCritic, self).__init__()

        layers = []
        prev_size = input_dim

        for i in range(hidden_layers):
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.LeakyReLU(0.2))
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# =============================================================================
# Enhanced WA-GAN with Heavy-Tail Corrections
# =============================================================================

class WAGAN_HeavyTail:
    def __init__(self, d, latent_dim=None, hidden_layers=4, hidden_size=256,
                 lambda_gp=10.0, rho=1.0, n_critic=5, lr=0.0001, betas=(0.5, 0.9),
                 latent_dist='student', df=2.5, kurtosis_weight=0.1):
        self.d = d
        self.latent_dim = latent_dim if latent_dim is not None else max(1, d - 1)
        self.hidden_layers = hidden_layers
        self.hidden_size = hidden_size
        self.lambda_gp = lambda_gp
        self.rho = rho
        self.n_critic = n_critic
        self.lr = lr
        self.betas = betas
        self.latent_dist = latent_dist
        self.df = df
        self.kurtosis_weight = kurtosis_weight

        self.aitchison = AitchisonTransform(self.d)

        self.generator = HeavyTailGenerator(
            self.latent_dim, max(1, self.d - 1), hidden_layers, hidden_size,
            latent_dist=latent_dist, df=df
        )
        self.critic = HeavyTailCritic(max(1, self.d - 1), hidden_layers, hidden_size)

        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=lr, betas=betas)
        self.optimizer_D = optim.Adam(self.critic.parameters(), lr=lr, betas=betas)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.generator.to(self.device)
        self.critic.to(self.device)

        self.history = {'critic_loss': [], 'generator_loss': [], 'margin_penalty': [],
                       'kurtosis_loss': []}

    def _compute_gradient_penalty(self, real_data, fake_data):
        batch_size = real_data.size(0)

        alpha = torch.rand(batch_size, 1).to(self.device)
        alpha = alpha.expand_as(real_data)
        interpolated = alpha * real_data + (1 - alpha) * fake_data
        interpolated.requires_grad_(True)

        critic_interpolated = self.critic(interpolated)

        grad = torch.autograd.grad(
            outputs=critic_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(critic_interpolated),
            create_graph=True,
            retain_graph=True
        )[0]

        grad_norm = grad.view(batch_size, -1).norm(2, dim=1)
        gradient_penalty = ((grad_norm - 1) ** 2).mean()

        return gradient_penalty

    def _compute_marginal_penalty(self, generated_coords):
        batch_size = generated_coords.size(0)

        simplex_points = []
        for i in range(batch_size):
            point = self.aitchison.from_coordinates(
                generated_coords[i].detach().cpu().numpy()
            )
            simplex_points.append(point)

        simplex_points = torch.tensor(np.array(simplex_points)).to(self.device)
        empirical_mean = simplex_points.mean(dim=0)
        target_mean = torch.ones(self.d).to(self.device) / self.d
        margin_penalty = torch.norm(empirical_mean - target_mean) ** 2

        return margin_penalty

    def _compute_kurtosis_penalty(self, generated_coords):
        """Penalize deviations from target kurtosis."""
        batch_size = generated_coords.size(0)

        coords_np = generated_coords.detach().cpu().numpy()

        if self.latent_dist == 'student' and self.df > 4:
            target_kurtosis = 6 / (self.df - 4)
        elif self.latent_dist == 'student' and self.df <= 4:
            target_kurtosis = 15
        elif self.latent_dist == 'cauchy':
            target_kurtosis = 20
        elif self.latent_dist == 'laplace':
            target_kurtosis = 3
        else:
            target_kurtosis = 0

        kurtosis_vals = []
        for j in range(generated_coords.shape[1]):
            coord_j = coords_np[:, j]
            mean_j = np.mean(coord_j)
            std_j = np.std(coord_j)
            if std_j > 0:
                kurt = np.mean(((coord_j - mean_j) / std_j) ** 4) - 3
                kurtosis_vals.append(kurt)
            else:
                kurtosis_vals.append(0)

        avg_kurtosis = np.mean(kurtosis_vals)
        kurtosis_penalty = (avg_kurtosis - target_kurtosis) ** 2

        return torch.tensor(kurtosis_penalty, dtype=torch.float32).to(self.device)

    def train_step(self, real_coords):
        batch_size = real_coords.size(0)
        real_coords = real_coords.to(self.device)

        for _ in range(self.n_critic):
            z = self.generator.sample_latent(batch_size).to(self.device)
            fake_coords = self.generator(z)

            critic_real = self.critic(real_coords).mean()
            critic_fake = self.critic(fake_coords).mean()

            gp = self._compute_gradient_penalty(real_coords, fake_coords)
            critic_loss = -critic_real + critic_fake + self.lambda_gp * gp

            self.optimizer_D.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
            self.optimizer_D.step()

        z = self.generator.sample_latent(batch_size).to(self.device)
        fake_coords = self.generator(z)

        generator_loss = -self.critic(fake_coords).mean()
        margin_penalty = self._compute_marginal_penalty(fake_coords)
        kurtosis_penalty = self._compute_kurtosis_penalty(fake_coords)

        total_generator_loss = (generator_loss +
                               self.rho * margin_penalty +
                               self.kurtosis_weight * kurtosis_penalty)

        self.optimizer_G.zero_grad()
        total_generator_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 1.0)
        self.optimizer_G.step()

        return {
            'critic_loss': critic_loss.item(),
            'generator_loss': generator_loss.item(),
            'margin_penalty': margin_penalty.item(),
            'kurtosis_loss': kurtosis_penalty.item()
        }

    def fit(self, coords, epochs=5000, batch_size=64, verbose=True):
        coords = torch.tensor(np.array(coords), dtype=torch.float32)
        dataset = TensorDataset(coords)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            epoch_losses = {'critic_loss': [], 'generator_loss': [],
                           'margin_penalty': [], 'kurtosis_loss': []}

            for batch in dataloader:
                real_coords = batch[0]
                losses = self.train_step(real_coords)
                for key in epoch_losses:
                    epoch_losses[key].append(losses.get(key, 0))

            avg_losses = {key: np.mean(values) for key, values in epoch_losses.items()}
            for key in self.history:
                self.history[key].append(avg_losses.get(key, 0))

            if verbose and (epoch + 1) % 500 == 0:
                print(f"Epoch {epoch+1}/{epochs}: "
                      f"Critic: {avg_losses['critic_loss']:.4f}, "
                      f"Generator: {avg_losses['generator_loss']:.4f}, "
                      f"Margin: {avg_losses['margin_penalty']:.4f}, "
                      f"Kurtosis: {avg_losses.get('kurtosis_loss', 0):.4f}")

    def generate(self, n_samples):
        self.generator.eval()
        with torch.no_grad():
            z = self.generator.sample_latent(n_samples).to(self.device)
            generated_coords = self.generator(z).cpu().numpy()
        return generated_coords

    def generate_angles(self, n_samples):
        coords = self.generate(n_samples)
        angles = []
        for i in range(n_samples):
            angle = self.aitchison.from_coordinates(coords[i])
            angles.append(angle)
        return np.array(angles)


# =============================================================================
# Enhanced Simulator for CSV Data with Log Returns
# =============================================================================

class FinancialExtremeSimulator_HeavyTail:
    """Enhanced simulator for real financial data from CSV with log returns."""

    def __init__(self, data, k1=None, k2=None, latent_dim=None,
                 hidden_layers=4, hidden_size=256,
                 lambda_gp=10.0, rho=1.0, n_critic=5,
                 latent_dist='student', df=2.5, kurtosis_weight=0.1,
                 gpd_threshold_q=0.95):
        
        self.data = np.array(data)
        self.n, self.d = self.data.shape
        self.loss_data = -self.data

        self.k1 = k1 if k1 is not None else int(np.sqrt(self.n))
        self.k2 = k2 if k2 is not None else self.k1

        self.latent_dim = latent_dim if latent_dim is not None else max(1, self.d - 1)
        self.hidden_layers = hidden_layers
        self.hidden_size = hidden_size
        self.lambda_gp = lambda_gp
        self.rho = rho
        self.n_critic = n_critic
        self.latent_dist = latent_dist
        self.df = df
        self.kurtosis_weight = kurtosis_weight

        self.gpd_threshold_q = gpd_threshold_q
        self._gpd_params = None
        self.radial_threshold = None

        self.aitchison = AitchisonTransform(self.d)
        self.wagan = None
        self.is_trained = False

        self.standardized_data = None
        self.angles_data = None
        self.coords_data = None

        self.data_mean = np.mean(self.loss_data, axis=0)
        self.data_std = np.std(self.loss_data, axis=0)

    def _empirical_standardize(self):
        standardized = np.zeros_like(self.loss_data)

        for j in range(self.d):
            ranks = np.argsort(np.argsort(self.loss_data[:, j])) + 1
            cdf = ranks / (self.n + 1)
            standardized[:, j] = 1 / (1 - cdf)

        return standardized

    def _compute_angles(self, standardized):
        R = np.sum(standardized, axis=1)
        t = self.n / self.k1
        self.radial_threshold = t

        extreme_indices = R >= t
        angles = standardized[extreme_indices] / R[extreme_indices, np.newaxis]

        coords = []
        for angle in angles:
            coord = self.aitchison.to_coordinates(angle)
            coords.append(coord)

        self.standardized_data = standardized
        self.angles_data = angles
        self.coords_data = np.array(coords)

        return np.array(coords)

    def _fit_gpd_tails(self):
        gpd_params = []
        for j in range(self.d):
            col = self.loss_data[:, j]
            threshold = np.quantile(col, self.gpd_threshold_q)
            exceedances = col[col > threshold] - threshold

            if len(exceedances) >= 20:
                xi, loc, beta = genpareto.fit(exceedances, floc=0)
            else:
                xi, loc, beta = 0.0, 0.0, np.std(exceedances) if len(exceedances) > 1 else 1e-6

            gpd_params.append({'threshold': threshold, 'xi': xi, 'beta': beta})

        self._gpd_params = gpd_params

    def _quantile_with_tail_extrapolation(self, j, p):
        p = np.clip(p, 1e-6, 1 - 1e-9)

        if p <= self.gpd_threshold_q or self._gpd_params is None:
            return np.quantile(self.loss_data[:, j], p)

        params = self._gpd_params[j]
        threshold, xi, beta = params['threshold'], params['xi'], params['beta']

        q_cond = (p - self.gpd_threshold_q) / (1 - self.gpd_threshold_q)
        q_cond = np.clip(q_cond, 1e-9, 1 - 1e-9)

        tail_value = genpareto.ppf(q_cond, xi, loc=0, scale=beta)
        return threshold + tail_value

    def fit(self, epochs=5000, batch_size=64, verbose=True):
        standardized = self._empirical_standardize()
        coords = self._compute_angles(standardized)

        if len(coords) == 0:
            raise ValueError(f"No data points exceed the threshold. "
                           f"Try increasing k1 (currently {self.k1}).")

        self._fit_gpd_tails()

        
        R_train = np.sum(standardized, axis=1)
        self._bulk_mask = R_train < self.radial_threshold

        print(f"Using {len(coords)} extreme observations for training")
        print(f"Latent distribution: {self.latent_dist} with df={self.df}")
        print(f"Data statistics (log returns):")
        print(f"  Mean: {np.mean(self.data_mean):.6f}")
        print(f"  Std:  {np.mean(self.data_std):.6f}")
        print(f"  Min:  {np.min(self.loss_data):.4f}")
        print(f"  Max:  {np.max(self.loss_data):.4f}")
        for j, p in enumerate(self._gpd_params):
            print(f"  [Asset {j}] GPD tail fit: threshold={p['threshold']:.4f}, "
                  f"xi={p['xi']:.4f}, beta={p['beta']:.4f}")

        self.wagan = WAGAN_HeavyTail(
            d=self.d,
            latent_dim=self.latent_dim,
            hidden_layers=self.hidden_layers,
            hidden_size=self.hidden_size,
            lambda_gp=self.lambda_gp,
            rho=self.rho,
            n_critic=self.n_critic,
            latent_dist=self.latent_dist,
            df=self.df,
            kurtosis_weight=self.kurtosis_weight
        )

        self.wagan.fit(coords, epochs=epochs, batch_size=batch_size, verbose=verbose)
        self.is_trained = True

    def sample_extremes(self, n_samples):
        if not self.is_trained:
            raise ValueError("Model must be trained first.")

        samples = []
        attempts = 0
        max_attempts = n_samples * 10

        while len(samples) < n_samples and attempts < max_attempts:
            attempts += 1

            coords = self.wagan.generate(1)
            angle = self.aitchison.from_coordinates(coords[0])

            Y = np.random.pareto(1, 1) + 1
            R_new = self.radial_threshold * Y
            V = R_new * angle
            if np.max(V) > 1:
                sample = np.zeros(self.d)
                for j in range(self.d):
                    
                    if V[j] >= 1:
                        p = 1 - 1 / V[j]
                        sample[j] = self._quantile_with_tail_extrapolation(j, p)
                    else:
                        bulk_col = self.loss_data[self._bulk_mask, j]
                        sample[j] = bulk_col[np.random.randint(len(bulk_col))]

                samples.append(sample)

        if len(samples) < n_samples:
            print(f"Warning: Only generated {len(samples)} samples out of {n_samples}")

        return np.array(samples[:n_samples])

    def sample_angles(self, n_samples):
        if not self.is_trained:
            raise ValueError("Model must be trained first.")
        return self.wagan.generate_angles(n_samples)

    def get_test_tail(self, test_data, q=None):
        
        test_data = np.array(test_data)
        test_losses = -test_data
        n_t = len(test_losses)

        standardized_test = np.zeros_like(test_losses)
        for j in range(self.d):
            ranks = np.argsort(np.argsort(test_losses[:, j])) + 1
            cdf = ranks / (n_t + 1)
            standardized_test[:, j] = 1 / (1 - cdf)

        R_test = np.sum(standardized_test, axis=1)
        # stessa proporzione (k1/n) usata in training, applicata al test set
        k_test = max(1, int(round(self.k1 * n_t / self.n)))
        t_test = n_t / k_test

        extreme_mask = R_test >= t_test
        return test_losses[extreme_mask]


# =============================================================================
# CSV Data Loader
# =============================================================================

def load_csv_single_asset(file_path, price_col='Close', date_col='Date'):
    print(f"Loading data from: {file_path}")

    df = pd.read_csv(file_path)
    print(f"Raw data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    if date_col and date_col in df.columns:
        dates = pd.to_datetime(df[date_col])
        print(f"Date range: {dates.min()} to {dates.max()}")
    else:
        dates = None

    if price_col not in df.columns:
        raise ValueError(f"Column '{price_col}' not found.")

    prices = df[price_col].values

    log_returns = np.diff(np.log(prices + 1e-10))

    mask = ~(np.isnan(log_returns) | np.isinf(log_returns))
    log_returns = log_returns[mask]
    if dates is not None:
        dates = dates[1:][mask]

    log_returns = log_returns.reshape(-1, 1)

    print(f"\nLog returns shape: {log_returns.shape}")
    print(f"Log returns statistics:")
    print(f"  Mean: {np.mean(log_returns):.6f}")
    print(f"  Std:  {np.std(log_returns):.6f}")
    print(f"  Min:  {np.min(log_returns):.4f}")
    print(f"  Max:  {np.max(log_returns):.4f}")
    print(f"  Kurtosis: {kurtosis(log_returns.flatten(), fisher=True):.4f}")

    return log_returns, dates, prices


def load_multiple_csvs(file_paths, price_col='Close', date_col='Date'):
    """Carica e unisce più file CSV allineandoli per data."""
    print(f"Loading and merging data from {len(file_paths)} files...")

    dfs = []
    for i, path in enumerate(file_paths):
        df = pd.read_csv(path)

        asset_name = f'Asset_{i}'
        df = df[[date_col, price_col]].rename(columns={price_col: asset_name})
        df[date_col] = pd.to_datetime(df[date_col])
        dfs.append(df)

    merged_df = dfs[0]
    for df in dfs[1:]:
        merged_df = pd.merge(merged_df, df, on=date_col, how='inner')

    merged_df = merged_df.sort_values(date_col).reset_index(drop=True)

    print(f"Merged rows after inner join on '{date_col}': {len(merged_df)}")
    for i, path in enumerate(file_paths):
        print(f"  (verifica manualmente che questo numero non sia molto più")
        print(f"   piccolo delle righe originali di {path} — un inner join")
        print(f"   su calendari di mercato diversi (es. festività locali)")
        print(f"   può scartare molte osservazioni)")
        break  # stampa la nota una sola volta

    dates = merged_df[date_col].values
    prices = merged_df.drop(columns=[date_col]).values

    log_returns = np.diff(np.log(prices + 1e-10), axis=0)

    mask = ~np.any(np.isnan(log_returns) | np.isinf(log_returns), axis=1)
    log_returns = log_returns[mask]
    dates = dates[1:][mask]

    print(f"Merged Data Shape: {log_returns.shape}")
    print(f"Date range: {pd.to_datetime(dates.min()).date()} to {pd.to_datetime(dates.max()).date()}")

    return log_returns, dates, prices


# =============================================================================
# Enhanced Metrics
# =============================================================================

class EnhancedMetrics:
    @staticmethod
    def compute_all_metrics(data_true, data_gen):
        metrics = {}
        kurt_true = np.array([kurtosis(data_true[:, j]) for j in range(data_true.shape[1])])
        kurt_gen = np.array([kurtosis(data_gen[:, j]) for j in range(data_gen.shape[1])])
        metrics['kurtosis_true'] = kurt_true
        metrics['kurtosis_gen'] = kurt_gen
        skew_true = np.array([skew(data_true[:, j]) for j in range(data_true.shape[1])])
        skew_gen = np.array([skew(data_gen[:, j]) for j in range(data_gen.shape[1])])
        metrics['skewness_true'] = skew_true
        metrics['skewness_gen'] = skew_gen
        metrics['mean_true'] = np.mean(data_true, axis=0)
        metrics['mean_gen'] = np.mean(data_gen, axis=0)
        metrics['std_true'] = np.std(data_true, axis=0)
        metrics['std_gen'] = np.std(data_gen, axis=0)
        return metrics

    @staticmethod
    def compute_var_es_metrics(real_losses_or_returns, generated_losses, tail_fraction, alpha=0.01,
                           input_is_returns=True):
        real_flat = np.array(real_losses_or_returns).flatten()
        if input_is_returns:
            real_losses = -real_flat
        else:
            real_losses = real_flat

        sorted_real = np.sort(real_losses)[::-1]
        idx_real = max(1, int(alpha * len(sorted_real)))
        vaR_real = sorted_real[idx_real - 1]
        es_real = np.mean(sorted_real[:idx_real])

        # Proiezione dell'alpha globale in alpha "locale" per i dati generati
        # (che rappresentano SOLO la tail_fraction più estrema).
        adjusted_alpha = alpha / tail_fraction

        if adjusted_alpha > 1:
            print(f"\n[!] ATTENZIONE: alpha={alpha} cade al di fuori della coda generata (tail_fraction={tail_fraction:.4f}).")
            adjusted_alpha = 1.0

        gen_losses = np.array(generated_losses).flatten()
        sorted_gen = np.sort(gen_losses)[::-1]
        idx_gen = max(1, int(adjusted_alpha * len(sorted_gen)))

        vaR_gen = sorted_gen[idx_gen - 1]
        es_gen = np.mean(sorted_gen[:idx_gen])

        var_error = abs(vaR_gen - vaR_real) / (abs(vaR_real) + 1e-10)
        es_error = abs(es_gen - es_real) / (abs(es_real) + 1e-10)

        return {
            'real_var': vaR_real, 'gen_var': vaR_gen, 'var_error': var_error,
            'real_es': es_real, 'gen_es': es_gen, 'es_error': es_error
        }


# =============================================================================
# Visualization Functions
# =============================================================================

def plot_training_history(simulator):
    if simulator.wagan is None:
        print("Model not trained yet")
        return
    history = simulator.wagan.history
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].plot(history['critic_loss'], linewidth=2)
    axes[0, 0].set_title('Critic Loss', fontsize=14)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 1].plot(history['generator_loss'], linewidth=2, color='orange')
    axes[0, 1].set_title('Generator Loss', fontsize=14)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].grid(True, alpha=0.3)
    axes[1, 0].plot(history['margin_penalty'], linewidth=2, color='green', label='Margin')
    axes[1, 0].plot(history['kurtosis_loss'], linewidth=2, color='red', label='Kurtosis')
    axes[1, 0].set_title('Regularization Penalties', fontsize=14)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Penalty')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 1].plot(history['generator_loss'], label='Generator', linewidth=2)
    axes[1, 1].plot(history['kurtosis_loss'], label='Kurtosis Penalty', linewidth=2)
    axes[1, 1].set_title('Generator and Kurtosis Loss', fontsize=14)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Value')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_log_return_distribution(simulator, generated_data, test_data):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax = axes[0, 0]
    ax.hist(test_data[:, 0], bins=50, alpha=0.5, label='Test', density=True)
    ax.hist(generated_data[:, 0], bins=50, alpha=0.5, label='Generated', density=True)
    ax.set_title('Log Return Distribution (Dim 0)', fontsize=14)
    ax.set_xlabel('Log Return')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax = axes[0, 1]
    sorted_gen = np.sort(generated_data[:, 0])
    sorted_test = np.sort(test_data[:, 0])
    min_len = min(len(sorted_gen), len(sorted_test))
    ax.scatter(sorted_test[:min_len], sorted_gen[:min_len], alpha=0.5, s=20)
    ax.plot([min(sorted_test), max(sorted_test)],
           [min(sorted_test), max(sorted_test)], 'r--', linewidth=2)
    ax.set_title('QQ Plot (Dim 0)', fontsize=14)
    ax.set_xlabel('Test Quantiles')
    ax.set_ylabel('Generated Quantiles')
    ax.grid(True, alpha=0.3)
    ax = axes[1, 0]
    threshold = np.percentile(test_data[:, 0], 95)
    test_tail_plot = test_data[test_data[:, 0] > threshold, 0]
    gen_tail = generated_data[generated_data[:, 0] > threshold, 0]
    if len(test_tail_plot) > 5 and len(gen_tail) > 5:
        sorted_test_tail = np.sort(test_tail_plot)
        sorted_gen_tail = np.sort(gen_tail)
        min_len = min(len(sorted_test_tail), len(sorted_gen_tail))
        ax.scatter(sorted_test_tail[:min_len], sorted_gen_tail[:min_len],
                  alpha=0.5, s=20, color='red')
        ax.plot([min(sorted_test_tail), max(sorted_test_tail)],
               [min(sorted_test_tail), max(sorted_test_tail)], 'b--', linewidth=2)
        ax.set_title('Tail QQ Plot (Top 5%)', fontsize=14)
        ax.set_xlabel('Test Quantiles')
        ax.set_ylabel('Generated Quantiles')
        ax.grid(True, alpha=0.3)
    ax = axes[1, 1]
    data_to_plot = [test_data[:, 0], generated_data[:, 0]]
    bp = ax.boxplot(data_to_plot, labels=['Test', 'Generated'])
    ax.set_title('Box Plot Comparison (Dim 0)', fontsize=14)
    ax.set_ylabel('Log Return')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_extreme_paths(simulator, n_paths=10, n_points=100):
    if not simulator.is_trained:
        print("Model not trained yet")
        return
    paths = []
    for _ in range(n_paths):
        sample = simulator.sample_extremes(n_points)
        paths.append(sample)
    paths = np.array(paths)
    d_display = min(5, simulator.d)
    dims = np.random.choice(simulator.d, d_display, replace=False)
    fig, axes = plt.subplots(1, d_display, figsize=(15, 4))
    if d_display == 1:
        axes = [axes]
    colors = plt.cm.viridis(np.linspace(0, 1, n_paths))
    for idx, dim in enumerate(dims):
        ax = axes[idx]
        for path_idx in range(n_paths):
            ax.plot(paths[path_idx, :, dim],
                   color=colors[path_idx],
                   alpha=0.7,
                   linewidth=1.5)
        ax.set_title(f'Dimension {dim}', fontsize=12)
        ax.set_xlabel('Time')
        ax.set_ylabel('Loss' if idx == 0 else '')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.grid(True, alpha=0.3)
    plt.suptitle('Simulated Extreme Loss Paths (Log Returns)', fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_correlation_matrix(data, title="Correlation Matrix"):
    corr = np.corrcoef(data.T)
    fig, ax = plt.subplots(figsize=(10, 8))

    if data.shape[1] == 1:
        im = ax.imshow([[1]], cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_xticks([0])
        ax.set_yticks([0])
        ax.set_xticklabels(['Asset 1'])
        ax.set_yticklabels(['Asset 1'])
    else:
        im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_xticks(range(data.shape[1]))
        ax.set_yticks(range(data.shape[1]))

    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Variables')
    ax.set_ylabel('Variables')
    plt.colorbar(im)
    plt.tight_layout()
    plt.show()


# =============================================================================
# Main Execution - Multi-Asset
# =============================================================================
def main():
    print("=" * 70)
    print("WA-GAN - Multi-Asset Analysis (senza scale_factor)")
    print("=" * 70)

    file_paths = [
        "/content/^nkx_d-2.csv",
        "/content/^ndx_d copia.csv"
    ]

    try:
        log_returns, dates, prices = load_multiple_csvs(
            file_paths,
            price_col='Close',
            date_col='Date'
        )
    except FileNotFoundError:
        from google.colab import files
        print("\nCarica i file CSV richiesti...")
        uploaded = files.upload()
        file_paths = list(uploaded.keys())
        log_returns, dates, prices = load_multiple_csvs(file_paths)

    print("\n" + "=" * 50)
    print("DATA ANALYSIS")
    print("=" * 50)

    print(f"Number of log return observations: {len(log_returns)}")
    print(f"Number of assets/columns: {log_returns.shape[1]}")

    n_train = int(0.7 * len(log_returns))
    n_test = len(log_returns) - n_train

    train_data = log_returns[:n_train]
    test_data = log_returns[n_train:]

    print(f"Training: {n_train}, Test: {n_test}")

    print("\n" + "=" * 50)
    print("TRAINING WA-GAN")
    print("=" * 50)

    configs = [
        ('gaussian', None, 0),
        ('student', 5.0, 0.05),
        ('student', 2.5, 0.1),
        ('student', 1.5, 0.15),
    ]

    best_simulator = None
    best_kurtosis_diff = float('inf')

    for latent_dist, df, kurt_weight in configs:
        print(f"\n   Testing: {latent_dist} (df={df})")
        try:
            simulator = FinancialExtremeSimulator_HeavyTail(
                train_data,
                k1=int(np.sqrt(n_train)),
                k2=int(np.sqrt(n_train)),
                hidden_layers=3,
                hidden_size=128,
                lambda_gp=10.0,
                rho=1.0,
                n_critic=5,
                latent_dist=latent_dist,
                df=df if df is not None else 2.5,
                kurtosis_weight=kurt_weight
            )
            simulator.fit(epochs=1000, batch_size=64, verbose=False)
            n_extremes = 500
            generated = simulator.sample_extremes(n_extremes)

            
            test_tail = simulator.get_test_tail(test_data)

            metrics = EnhancedMetrics.compute_all_metrics(test_tail, generated)
            kurt_diff = np.mean(np.abs(metrics['kurtosis_true'] - metrics['kurtosis_gen']))
            mean_kurt_true = np.mean(metrics['kurtosis_true'])
            mean_kurt_gen = np.mean(metrics['kurtosis_gen'])
            print(f"   Mean Kurtosis - Test tail: {mean_kurt_true:.2f}, Generated: {mean_kurt_gen:.2f}")
            print(f"   Kurtosis Difference: {kurt_diff:.2f}")
            if kurt_diff < best_kurtosis_diff:
                best_kurtosis_diff = kurt_diff
                best_simulator = simulator
        except Exception as e:
            print(f"   Failed: {e}")

    if best_simulator is None:
        print("\nUsing default configuration...")
        simulator = FinancialExtremeSimulator_HeavyTail(
            train_data,
            k1=int(np.sqrt(n_train)),
            k2=int(np.sqrt(n_train)),
            latent_dist='student',
            df=2.5,
            kurtosis_weight=0.1
        )
        simulator.fit(epochs=2000, batch_size=64, verbose=True)
    else:
        print(f"\nBest configuration: {best_simulator.latent_dist} with df={best_simulator.df}")
        simulator = best_simulator

    print("\n" + "=" * 50)
    print("GENERATING SAMPLES")
    print("=" * 50)

    n_extremes = 1000
    generated_extremes = simulator.sample_extremes(n_extremes)
    generated_angles = simulator.sample_angles(n_extremes)

    print(f"Generated {len(generated_extremes)} samples")
    print(f"Mean: {np.mean(generated_extremes):.6f}")
    print(f"Std: {np.std(generated_extremes):.6f}")

    print("\n" + "=" * 50)
    print("EVALUATION")
    print("=" * 50)

    test_tail = simulator.get_test_tail(test_data)
    print(f"Test tail (soglia radiale, coerente col training): "
          f"{len(test_tail)} osservazioni su {len(test_data)} totali")

    metrics = EnhancedMetrics.compute_all_metrics(test_tail, generated_extremes)

    print(f"\nKurtosis (Test TAIL vs Generated):")
    print(f"  Test tail: {np.mean(metrics['kurtosis_true']):.2f} ± {np.std(metrics['kurtosis_true']):.2f}")
    print(f"  Generated: {np.mean(metrics['kurtosis_gen']):.2f} ± {np.std(metrics['kurtosis_gen']):.2f}")

    print("\n" + "=" * 50)
    print("VAR/ES EVALUATION (α=1%)")
    print("=" * 50)

    tail_frac = 1.0 / simulator.radial_threshold

    var_es = EnhancedMetrics.compute_var_es_metrics(
        test_data,
        generated_extremes,
        tail_fraction=tail_frac,
        alpha=0.01,
        input_is_returns=True
    )

    print(f"Real VaR (loss-space): {var_es['real_var']:.6f}")
    print(f"Generated VaR: {var_es['gen_var']:.6f}")
    print(f"VaR Error: {var_es['var_error']:.2%}")

    print(f"\nReal ES (loss-space): {var_es['real_es']:.6f}")
    print(f"Generated ES: {var_es['gen_es']:.6f}")
    print(f"ES Error: {var_es['es_error']:.2%}")

    print("\n" + "-" * 40)
    print("CONFRONTO CON GBM DIFFUSION")
    print("-" * 40)
    print("GBM Diffusion: VaR Error ~60%, ES Error ~15-16%")
    print(f"WA-GAN: VaR Error {var_es['var_error']:.2%}, ES Error {var_es['es_error']:.2%}")

    print("\n" + "=" * 50)
    print("VISUALIZATIONS")
    print("=" * 50)

    plot_training_history(simulator)
    plot_log_return_distribution(simulator, generated_extremes, test_data)
    plot_extreme_paths(simulator, n_paths=10, n_points=100)
    plot_correlation_matrix(generated_extremes, "Generated Extreme Losses Correlation")

    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)

    return simulator, generated_extremes, generated_angles


if __name__ == "__main__":
    simulator, extremes, angles = main()