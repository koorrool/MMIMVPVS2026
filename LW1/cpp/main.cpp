#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <random>
#include <iomanip>

using namespace std;

// ядро Епанченикова
double kernel(double z) { return (abs(z) <= 1.0) ? 0.75 * (1.0 - z * z) : 0.0; } // формула 4.7.4

// оценка регрессии, по формуле 4.7.3
double estimate(const vector<double>& xs, const vector<double>& ys, double x, double h, int skip_idx = -1) {
    double num = 0.0, den = 0.0;
    for (int i = 0; i < (int)xs.size(); ++i) {
        if (i == skip_idx) continue;
        double w = kernel((x - xs[i]) / h);
        num += w * ys[i];
        den += w;
    }
    return (den > 1e-12) ? num / den : 0.0;
}

// скользящий экзамен
double loocv_mse(const vector<double>& xs, const vector<double>& ys, double beta, double delta) {
    int n = xs.size();
    double h = delta / beta;  // из формулы 4.7.5: beta = c^-1 * delta * n^(1/5), h = c * n^(-1/5) = delta/beta
    double mse = 0.0;
    for (int i = 0; i < n; ++i) {
        double pred = estimate(xs, ys, xs[i], h, i);
        double err = ys[i] - pred;
        mse += err * err;
    }
    return mse / n;
}

int main() {
    // генерация данных
    const int N = 80;
    const double X_MIN = -3.0, X_MAX = 7.0;

    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> uni(X_MIN, X_MAX);
    normal_distribution<double> noise(0.0, 0.3);

    vector<double> xs(N), ys(N);
    for (int i = 0; i < N; ++i) {
        xs[i] = uni(gen);
        double x = xs[i];
        ys[i] = 0.1 * (x - 4) * cos(x) + 0.5 * x + noise(gen);
    }

    // считаем дельта
    vector<double> sorted_x = xs;
    sort(sorted_x.begin(), sorted_x.end());
    double delta = 0.0;
    for (int i = 0; i < N - 1; ++i) delta = max(delta, sorted_x[i + 1] - sorted_x[i]);
    cout << "Delta (max spacing) = " << delta << "\n\n";

    // ищем оптимальную beta
    double best_beta = 0.1, best_mse = 1e18;

    cout << fixed << setprecision(6);
    cout << "Beta\t\tLOO-MSE\n";
    cout << "----\t\t-------\n";

    for (int k = 1; k <= 20; ++k) {
        double beta = k * 0.1;
        double mse = loocv_mse(xs, ys, beta, delta);
        cout << beta << "\t\t" << mse << "\n";
        if (mse < best_mse) {
            best_mse = mse;
            best_beta = beta;
        }
    }

    cout << "\nOptimal beta = " << best_beta << "  (LOO-MSE = " << best_mse << ")\n";
}