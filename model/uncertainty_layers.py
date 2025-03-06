class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Weight mean and variance parameters
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))

        # Bias mean and variance parameters
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))

        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize means
        nn.init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_mu)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias_mu, -bound, bound)

        # Initialize sigmas (log variance)
        nn.init.constant_(self.weight_sigma, -6.0)  # exp(-6) is a small value
        nn.init.constant_(self.bias_sigma, -6.0)

    def forward(self, x):
        # Sample weights during training
        if self.training:
            weight = self.weight_mu + torch.exp(self.weight_sigma) * torch.randn_like(self.weight_sigma)
            bias = self.bias_mu + torch.exp(self.bias_sigma) * torch.randn_like(self.bias_sigma)
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)

    def kl_divergence(self):
        # KL divergence between N(mu, sigma^2) and N(0, 1)
        kl_weight = 0.5 * torch.sum(
            torch.exp(2 * self.weight_sigma) + self.weight_mu ** 2 - 1 - 2 * self.weight_sigma
        )
        kl_bias = 0.5 * torch.sum(
            torch.exp(2 * self.bias_sigma) + self.bias_mu ** 2 - 1 - 2 * self.bias_sigma
        )
        return kl_weight + kl_bias


class UncertaintyModule(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=128, output_dim=5):
        super().__init__()

        # Bayesian neural network layers
        self.bayes1 = BayesianLinear(input_dim, hidden_dim)
        self.bayes2 = BayesianLinear(hidden_dim, hidden_dim)
        self.bayes_out = BayesianLinear(hidden_dim, output_dim)

        # Dropout for Monte Carlo dropout
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.bayes1(x))
        x = self.dropout(x)
        x = F.relu(self.bayes2(x))
        x = self.dropout(x)
        x = self.bayes_out(x)
        return x

    def kl_divergence(self):
        # Sum KL divergence from all Bayesian layers
        return self.bayes1.kl_divergence() + self.bayes2.kl_divergence() + self.bayes_out.kl_divergence()

    def sample_predictions(self, x, num_samples=10):
        # Enable dropout in evaluation mode
        self.train()

        # MC dropout sampling
        samples = []
        for _ in range(num_samples):
            samples.append(self.forward(x))

        # Stack samples [num_samples, batch, output_dim]
        return torch.stack(samples)